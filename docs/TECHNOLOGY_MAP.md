# POWERIQ Kenya: technology map and code layout

One folder, `poweriq/`. Nothing in the application was changed to produce this page; it only records what exists, where, and how far each technology goes. Statuses are deliberately honest.

## 1. Technology map

| Technology | Status | Where it lives | Notes |
|---|---|---|---|
| **MCP (Model Context Protocol)** | Implemented, basic | `backend/app/main.py` (`POST /api/v1/mcp`), tool set in `backend/app/agents.py` (`build_tools`) | JSON-RPC 2.0 with `initialize`, `tools/list`, `tools/call`; 14 **read-only** tools (overview, demand/generation forecast, maintenance risk, anomalies, losses, curtailment, grid status, N-1, climate, usage, scenario, storage optimisation, knowledge search). Analyst role required, every call audit-logged. Not yet tested against external MCP clients; the Fault Hub and Query Studio are not exposed as MCP tools yet. |
| **RAG (retrieval-augmented generation)** | Implemented, local | `backend/app/rag.py`, `backend/knowledge/*.md` (9 engineering documents) | Hybrid retrieval: word and character TF-IDF vectors plus a BM25-style keyword score, metadata filters, chunk overlap, citations returned with every answer. It is **not** neural-embedding search and **not** Azure AI Search or Foundry IQ; those report "not configured". |
| **AI agents** | Implemented | `backend/app/agents.py` | Demand, Generation, Maintenance, Grid, Storage, Loss, Climate agents plus a Chief agent that merges findings; each returns structured findings with confidence and evidence. Routing is deterministic, not LLM-driven. |
| **LLM (OpenAI)** | Optional, needs your key | `backend/app/copilot.py`, `backend/app/query_studio.py` | With `OPENAI_API_KEY` set on the server, the model writes narratives strictly from computed results and plans Query Studio queries. Without a key a local template/keyword planner is used. Only tested with mocked responses; `POST /query/test-openai` checks a real key. |
| **Machine learning** | Implemented | `backend/app/ml.py` | LightGBM demand/generation forecasts, XGBoost maintenance risk, IsolationForest anomaly detection, scikit-learn KMeans, statsmodels STL. |
| **Optimisation and grid physics** | Implemented | `optimize.py`, `network.py` | OR-Tools LP battery schedules; DC power flow, PTDF, N-1 screening (approximation, assumed line ratings). |
| **Simulation** | Implemented | `insights.py` | Monte Carlo loss-of-load, drought and wind-lull stress tests, Pareto theft-inspection planner. |
| **Fault Hub** | Implemented | `fault_hub.py`, `frontend/src/pages/FaultHub.tsx` | Located, prioritised malfunctions; operator and community reports; endpoint for an external detection model. Simulated live feed plus genuinely real-time posted reports. |
| **Maps** | Implemented | `PowerMap.tsx` (MapLibre 3D), `FaultMap.tsx` (MapLibre), `LocationMap.tsx` (Leaflet street/satellite) | Leaflet is also used in the published artifact. |
| **Security and audit** | Implemented | `security.py` | API-key roles (viewer, analyst, approver), rate limiting, hash-chained SQLite audit log, human approval queue. |
| **Frontend** | Implemented | `frontend/src` | React 18, TypeScript, Tailwind, Recharts, Vite, Vitest. |
| **Delivery** | Implemented | `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml`, `infra/azure/deploy.sh` | Single-container deploy serving the built frontend. |

### Not in the project (do not assume)
Work IQ or any Microsoft 365 / Foundry IQ connection; LSTM or Temporal Fusion Transformer forecasters; Kafka or Spark; PostgreSQL, TimescaleDB, PostGIS; MQTT, OPC UA or any live SCADA link; YOLO or any camera/satellite vision model (the Fault Hub only has an endpoint that receives detections from a model you train separately); Cesium; Power BI or Grafana connectors (Query Studio exports CSV). The MCP server is POWERIQ's own and could be added to an external agent as a client-side connection, but no such connection exists here.

## 2. Folder layout
```
poweriq/
  backend/   app/ (API, models, agents, RAG, MCP, Fault Hub)  knowledge/ (RAG documents)  tests/  scripts/
  frontend/  src/ (pages, components, api client)  config files
  e2e/       browser sweep
  examples/  send_detection.py (how your detection model posts to the Fault Hub)
  docs/      METRICS.md, this file
  infra/     Azure deploy script      .github/  CI
```

## 3. Code index (lines of code)
- `backend/app/__init__.py` (0 lines)
- `backend/app/agents.py` (430 lines) : Seven specialist agents + chief agent, tool registry (the MCP tool set)
- `backend/app/copilot.py` (220 lines) : AI Command Centre: intent routing, retrieval, optional OpenAI narrative
- `backend/app/fault_hub.py` (523 lines) : Fault Hub: malfunction detection feed, community reports, model-detection ingest, priority queue
- `backend/app/geo.py` (142 lines) : Hex grid and Kenya outline
- `backend/app/insights.py` (378 lines) : Kenya-problem modules (reserve risk, curtailment, theft, STL, findings)
- `backend/app/main.py` (734 lines) : FastAPI app: REST, MCP JSON-RPC endpoint, static frontend
- `backend/app/ml.py` (562 lines) : LightGBM/XGBoost/IsolationForest pipelines and metrics
- `backend/app/network.py` (49 lines) : DC power flow, PTDF, N-1
- `backend/app/optimize.py` (88 lines) : OR-Tools battery LP
- `backend/app/query_studio.py` (720 lines) : Copilot-style query planner/executor with scope guard and analysis
- `backend/app/rag.py` (175 lines) : Local hybrid retriever (TF-IDF word+char vectors, BM25-style score, citations)
- `backend/app/security.py` (236 lines) : API-key RBAC, rate limit, hash-chained audit log
- `backend/app/service.py` (572 lines) : State, dispatch/scenario engine, map layers
- `backend/app/synthetic.py` (516 lines) : Synthetic Kenya world generator
- `backend/tests/conftest.py` (30 lines)
- `backend/tests/test_api.py` (186 lines)
- `backend/tests/test_data_and_ml.py` (73 lines)
- `backend/tests/test_fault_hub.py` (76 lines)
- `backend/tests/test_insights.py` (46 lines)
- `backend/tests/test_optimize_network.py` (84 lines)
- `backend/tests/test_query_studio.py` (130 lines)
- `backend/tests/test_rag_agents.py` (80 lines)
- `frontend/src/App.tsx` (84 lines)
- `frontend/src/api.ts` (49 lines)
- `frontend/src/app.test.tsx` (102 lines)
- `frontend/src/index.css` (12 lines)
- `frontend/src/main.tsx` (7 lines)
- `frontend/src/setupTests.ts` (1 lines)
- `frontend/src/vite-env.d.ts` (1 lines)
- `frontend/src/components/FaultMap.tsx` (114 lines)
- `frontend/src/components/LocationMap.tsx` (92 lines)
- `frontend/src/components/PowerMap.tsx` (154 lines)
- `frontend/src/components/ui.tsx` (108 lines)
- `frontend/src/pages/Analytics.tsx` (138 lines)
- `frontend/src/pages/Command.tsx` (112 lines)
- `frontend/src/pages/DigitalTwin.tsx` (211 lines)
- `frontend/src/pages/FaultHub.tsx` (185 lines)
- `frontend/src/pages/Insights.tsx` (162 lines)
- `frontend/src/pages/Overview.tsx` (52 lines)
- `frontend/src/pages/QueryStudio.tsx` (131 lines)
- `frontend/src/pages/Storage.tsx` (42 lines)
- `e2e/ui_sweep.py` (37 lines)
- `examples/send_detection.py` (21 lines)
- `backend/knowledge/curtailment_and_storage.md` (10 lines)
- `backend/knowledge/data_dictionary.md` (10 lines)
- `backend/knowledge/dc_power_flow_limits.md` (10 lines)
- `backend/knowledge/geothermal_turbine_monitoring.md` (17 lines)
- `backend/knowledge/hydro_reservoir_management.md` (10 lines)
- `backend/knowledge/loss_classification.md` (12 lines)
- `backend/knowledge/operational_governance.md` (8 lines)
- `backend/knowledge/transformer_loading_temperature.md` (10 lines)
- `backend/knowledge/wind_power_curve.md` (8 lines)
