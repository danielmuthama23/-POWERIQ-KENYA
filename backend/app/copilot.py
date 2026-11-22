"""AI Copilot: natural-language interface grounded in agent outputs and cited knowledge.

Routing is deterministic (keyword intents). If OPENAI_API_KEY is set, the OpenAI API writes the narrative strictly from the
structured evidence; otherwise (or on any failure) a local template narrative is used. The structured sections (measured facts,
model predictions, assumptions, recommendations) are always produced by POWERIQ itself, never by the LLM.
"""
from __future__ import annotations

import json
import re
import uuid
from collections import defaultdict, deque

import httpx

from .agents import AgentInput, AgentRegistry
from .rag import get_kb
from .security import AuditLog, get_settings
from .service import PowerIQ

SOURCES = ["geothermal", "hydro", "wind", "solar", "thermal"]
INTENTS: list[tuple[str, list[str]]] = [
    ("storage", [r"batter", r"storage", r"bess"]),
    ("scenario", [r"what happens if", r"what if", r"\b(decrease|reduce|drop|fall|lower|cut)\w*\b.*\d+\s*%", r"\bincrease\b.*\d+\s*%"]),
    ("report", [r"report", r"priority list", r"priorit"]),
    ("storage", [r"batter", r"storage", r"bess", r"minimi[sz]es? peak"]),
    ("curtail", [r"curtail", r"renewable.*(wast|spill)"]),
    ("loss", [r"\bloss", r"theft", r"unaccounted", r"non-technical"]),
    ("maintenance", [r"maintenance", r"failure", r"risk", r"fault", r"turbine", r"vibration"]),
    ("gap", [r"gap", r"deficit", r"shortfall", r"reserve", r"adequa", r"unserved", r"blackout"]),
    ("grid", [r"transmission", r"bottleneck", r"congest", r"\bline\b", r"n-1", r"substation"]),
    ("climate", [r"weather", r"climate", r"rain", r"water", r"wind speed", r"irradian"]),
    ("demand", [r"demand", r"consumption", r"peak load", r"forecast", r"next 24"]),
    ("assess", [r"overall", r"status", r"summary", r"assessment", r"how is the grid", r"national"]),
]
INTENT_AGENTS = {"report": ["predictive_maintenance"], "storage": ["energy_storage_optimization", "generation_intelligence"],
                 "curtail": ["generation_intelligence", "energy_storage_optimization"], "loss": ["electricity_loss_detection"],
                 "maintenance": ["predictive_maintenance"], "gap": ["generation_intelligence", "demand_forecasting", "grid_intelligence"],
                 "grid": ["grid_intelligence"], "climate": ["climate_intelligence"], "demand": ["demand_forecasting"]}


def classify(q: str) -> str | None:
    ql = q.lower()
    for name, pats in INTENTS:
        if any(re.search(p, ql) for p in pats):
            return name
    return None


def parse_scenario(q: str) -> dict | None:
    ql = q.lower()
    pct = re.search(r"(\d+(?:\.\d+)?)\s*%", ql)
    if not pct:
        return None
    v = float(pct.group(1))
    if v <= 0 or v > 100:
        return None
    src = next((s for s in SOURCES if s in ql), None)
    up = re.search(r"increase|rise|grow|surge|higher", ql) is not None
    if "demand" in ql and up:
        return {"name": f"Demand +{v:g}%", "demand_increase_pct": v, "horizon_hours": 24}
    if src and re.search(r"decreas|reduc|drop|fall|lower|cut|decline|loss", ql):
        return {"name": f"{src.title()} output -{v:g}%", "generation_reductions": [{"source": src, "pct": v}], "horizon_hours": 24}
    return None


_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=12))


def _llm_narrative(evidence: dict, question: str) -> tuple[str | None, str | None]:
    st = get_settings()
    if not st.openai_api_key:
        return None, "OPENAI_API_KEY not configured"
    system = ("You are the POWERIQ Kenya copilot. Answer ONLY from the provided evidence JSON. Use four labelled sections: Measured facts, "
              "Model predictions, Assumptions, Recommendations. Quote numbers exactly as given, keep citation ids like [doc#0] when relevant, "
              "state uncertainty and limitations, never invent procedures, measurements or regulations, and say clearly when evidence is missing. "
              "All recommendations are advisory and need human approval.")
    try:
        r = httpx.post("https://api.openai.com/v1/chat/completions", timeout=30,
                       headers={"Authorization": f"Bearer {st.openai_api_key}"},
                       json={"model": st.openai_model, "temperature": 0.1,
                             "messages": [{"role": "system", "content": system},
                                          {"role": "user", "content": f"Question: {question}\n\nEvidence JSON:\n{json.dumps(evidence, default=str)[:24000]}"}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"], None
    except Exception as e:  # network error, bad key, timeout, malformed response
        return None, f"{type(e).__name__}: {str(e)[:160]}"


def _local_narrative(sections: dict, headline: str) -> str:
    def block(title, items):
        return f"**{title}**\n" + ("\n".join(f"- {i}" for i in items) if items else "- None available.") + "\n"
    return "\n".join([headline, "", block("Measured facts", sections["measured_facts"]), block("Model predictions", sections["model_predictions"]),
                      block("Assumptions", sections["assumptions"]), block("Recommendations (advisory, require human approval)", sections["recommendations"])])


def answer(question: str, svc: PowerIQ, reg: AgentRegistry, audit: AuditLog, actor: str, role: str, conversation_id: str | None) -> dict:
    cid = conversation_id or uuid.uuid4().hex[:12]
    intent = classify(question)
    secs = {"measured_facts": [], "model_predictions": [], "assumptions": [], "recommendations": []}
    citations: list[dict] = []
    agents_used: list[str] = []
    limitations: list[str] = []
    uncertainty = ""
    headline = ""
    model_version = svc.fb.version
    scenario = None

    def absorb(out):
        nonlocal uncertainty, model_version
        agents_used.append(out.agent)
        for f in out.findings:
            key = {"measured": "measured_facts", "prediction": "model_predictions", "assumption": "assumptions"}.get(f.kind)
            if key:
                secs[key].append(f"{f.statement} [{f.source}]")
        secs["recommendations"].extend(out.recommendations)
        citations.extend(out.citations)
        limitations.extend(out.limitations)
        uncertainty = uncertainty or out.uncertainty
        model_version = out.model_version or model_version

    if intent == "scenario":
        spec = parse_scenario(question)
        if spec is None:
            headline = "I could not extract a scenario (source or demand change and a percentage) from that question. Try: 'What happens if geothermal output decreases by 15%?'"
            intent = "scenario_unparsed"
        else:
            scenario = svc.scenario(spec)
            agents_used.append("chief_energy_intelligence")
            sb, ss, dl = scenario["baseline"], scenario["scenario"], scenario["delta_vs_baseline"]
            secs["measured_facts"].append(f"Baseline next 24 h (forecast engine): unserved {sb['unserved_energy_mwh']:.0f} MWh, min reserve margin {sb['min_reserve_margin_pct']:.1f}%, thermal {sb['thermal_energy_mwh']:.0f} MWh.")
            secs["model_predictions"] += [f"Scenario '{scenario['scenario_name']}': unserved {ss['unserved_energy_mwh']:.0f} MWh ({dl['unserved_energy_mwh']:+.0f}), min reserve {ss['min_reserve_margin_pct']:.1f}% ({dl['min_reserve_margin_pct']:+.1f} pts), thermal energy {dl['thermal_energy_mwh']:+.0f} MWh, hydro {dl['hydro_energy_mwh']:+.0f} MWh."] + scenario["warnings"][:3]
            secs["assumptions"] += ["Merit-order re-dispatch: hydro first, then thermal up to available capacity; imports and geothermal fixed apart from the stated change.", scenario["label"]]
            if dl["unserved_energy_mwh"] > 1 or ss["min_reserve_margin_pct"] < 5:
                secs["recommendations"].append("Pre-position thermal/hydro reserves and review import options for the affected hours; confirm with system operations before action.")
            else:
                secs["recommendations"].append("No immediate adequacy action indicated by the scenario; monitor thermal fuel readiness since thermal output rises.")
            limitations.append(scenario["label"])
            uncertainty = "Scenario inherits forecast error; demand 80% interval at peak is in /demand."
            headline = f"Scenario result for '{scenario['scenario_name']}':"
            kb = get_kb().query("dc power flow limits illustrative network", k=1, category="grid")
            citations += [{k: x[k] for k in ("citation", "title", "source", "score")} for x in kb["results"]]
    elif intent == "assess":
        c = reg.chief.assess(question, None, actor, role, audit)
        agents_used += ["chief_energy_intelligence"] + c["agents_run"]
        for f in c["findings"]:
            secs["measured_facts" if f["kind"] == "measured" else "assumptions"].append(f["statement"])
        for n, o in c["specialists"].items():
            for f in o["findings"][:2]:
                key = {"measured": "measured_facts", "prediction": "model_predictions", "assumption": "assumptions"}.get(f["kind"])
                if key:
                    secs[key].append(f"{f['statement']} [{f['source']}]")
            citations += o["citations"][:1]
        secs["recommendations"] += c["recommendations"]
        secs["assumptions"] += [f"Missing information: {m}" for m in c["missing_information"][:3]]
        for cf in c["conflicts_resolved"]:
            secs["assumptions"].append("Conflict resolved: " + cf["resolution"])
        limitations += c["limitations"]
        uncertainty = "See each specialist's uncertainty statement in /agents."
        headline = f"Consolidated assessment: overall risk level **{c['overall_risk_level']}**."
    elif intent in INTENT_AGENTS:
        for name in INTENT_AGENTS[intent]:
            params = {}
            out = reg.run(name, AgentInput(question=question, params=params), actor, role, audit)
            if out.status == "ok":
                absorb(out)
            else:
                secs["assumptions"].append(f"Agent {name} unavailable: {out.error}")
        headline = {"report": "Maintenance priority report (advisory):", "storage": "Storage optimisation results:", "curtail": "Renewable curtailment analysis:",
                    "loss": "Electricity loss analysis:", "maintenance": "Maintenance risk analysis:", "gap": "Generation-demand adequacy analysis:",
                    "grid": "Grid bottleneck analysis:", "climate": "Climate and hydrology analysis:", "demand": "Demand forecast:"}[intent]
        if intent == "report":
            mv = svc.maintenance_view()
            rows = ["| Rank | Unit | Plant | Risk | Drivers |", "|---|---|---|---|---|"]
            for i, u in enumerate(mv["units"][:8], 1):
                rows.append(f"| {i} | {u['unit_id']} | {u['plant_name']} | {u['risk_probability']:.0%} | {'; '.join(u['drivers'][:2]) or 'model pattern'} |")
            headline += "\n\n" + "\n".join(rows)
    else:
        r = get_kb().query(question, k=3)
        if r["answerable"]:
            headline = "From the engineering knowledge base (reference notes, not official publications):"
            for x in r["results"]:
                secs["measured_facts"].append(f"{x['text'][:300].strip()} {x['citation']}")
                citations.append({k: x[k] for k in ("citation", "title", "source", "score")})
            limitations.append("Retrieved passages are author-written reference notes; verify against official EPRA/KenGen/Kenya Power/KETRACO documents.")
        else:
            headline = ("I don't have evidence to answer that. The knowledge base and analytics do not cover it, so I won't guess. "
                        "Try asking about demand, generation, maintenance risk, losses, curtailment, storage or a what-if scenario.")
            intent = "no_evidence"

    sections = {k: v for k, v in secs.items()}
    # de-duplicate while preserving order
    for k in sections:
        seen, uniq = set(), []
        for x in sections[k]:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        sections[k] = uniq
    seen_c, cit = set(), []
    for c in citations:
        if c["citation"] not in seen_c:
            seen_c.add(c["citation"])
            cit.append(c)
    evidence = {"intent": intent, **sections, "citations": cit, "uncertainty": uncertainty, "limitations": list(dict.fromkeys(limitations))}
    text, llm_err = (None, "no analytic evidence to narrate") if intent in ("no_evidence", "scenario_unparsed") else _llm_narrative(evidence, question)
    mode = "openai" if text else "local-template"
    narrative = text or _local_narrative(sections, headline)
    recs = []
    for r in sections["recommendations"]:
        existing = [x for x in audit.list_recommendations("pending", 200) if x["text"] == r]
        rid = existing[0]["id"] if existing else audit.add_recommendation("copilot", r, actor, {"question": question, "intent": intent})
        recs.append({"id": rid, "text": r, "status": "pending"})
    _history[cid].append({"q": question, "intent": intent})
    audit.append("copilot_query", actor, role, {"question": question, "intent": intent, "agents": agents_used, "mode": mode, "llm_error": llm_err})
    return {"conversation_id": cid, "question": question, "intent": intent, "answer": narrative, "mode": mode,
            "llm_status": "ok" if mode == "openai" else f"not used: {llm_err}", **sections, "recommendation_records": recs, "citations": cit,
            "timestamp": svc._now(), "model_version": model_version, "uncertainty": uncertainty or "n/a",
            "limitations": list(dict.fromkeys(limitations)) + ["Synthetic data; advisory only."], "agents_used": agents_used,
            "scenario": scenario, "history_length": len(_history[cid])}
