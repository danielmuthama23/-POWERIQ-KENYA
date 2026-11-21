# POWERIQ Kenya

**POWERIQ Kenya** is an AI-assisted decision-support platform for electricity intelligence. It combines synthetic grid data, forecasting, predictive maintenance, network analysis, scenario simulation, local knowledge retrieval, multi-agent assessment, and audit-ready governance in a unified dashboard.

> POWERIQ is a prototype running on reproducible synthetic data. It is not connected to live grid-control systems. Results are advisory and must not be used as a substitute for operator-approved operational decisions.

## Key Capabilities

### Intelligence and Forecasting

- National demand forecasting with LightGBM at 1-, 6-, 24-, and 72-hour horizons
- Generation forecasting and merit-order-style dispatch for geothermal, hydro, wind, solar, thermal, and imported energy
- 80% forecast intervals for demand
- XGBoost-based 14-day power-unit fault-risk modelling
- Isolation Forest telemetry anomaly detection
- Technical and non-technical feeder-loss analysis
- Climate, hydrology, renewable generation, and demand analytics

### Grid and Scenario Analysis

- Illustrative 18-bus DC power-flow model
- Power Transfer Distribution Factors and N-1 contingency screening
- Scenario models for generation cuts, outages, derates, demand changes, and battery operation
- OR-Tools battery optimisation and alternative schedule comparison
- National 72-hour Digital Twin map with 257 hex cells and time replay
- Maintenance, fault, congestion, loss, renewable, and grid-stress views

### AI and Knowledge Features

- Seven specialist agents and a Chief Energy Intelligence agent
- Structured findings, evidence, uncertainty, limitations, and recommendations
- Hybrid local RAG using word TF-IDF, character TF-IDF, BM25-style ranking, metadata filtering, and citations
- Optional OpenAI narratives for the AI Copilot and Query Studio
- Deterministic local fallback when OpenAI is unavailable or not configured
- Scope-controlled Query Studio with validated, read-only dataset queries
- Query charts, tables, CSV export, dashboard builder, session-pinned tiles, and computed analysis
- Read-only MCP JSON-RPC endpoint for supported POWERIQ tools

### Fault Hub

- Interactive malfunction map for power stations, substations, and transmission lines
- Synthetic live/replay alerts and operator-reported incidents
- Fault prioritisation by severity, exposure, certainty, and imminence
- Analyst acknowledgement and resolution workflows
- Community concern reporting and grouped report review
- External-detection ingestion for independently trained camera, satellite, or drone models
- Leaflet street/satellite location view and MapLibre fault overlays
- Crew dispatch requests that create approval recommendations; the system never dispatches crews or controls equipment

### Kenya-Specific Modules

- Kenya problem scorecard and evidence-led analysis
- Monte Carlo reserve-risk assessment
- Curtailment-to-storage analysis
- Drought and wind-stress testing
- Power-station fleet clustering
- Theft and non-technical-loss inspection planning
- STL-based demand structure analysis
- Findings routed to the human approval queue

### Governance and Security

- API-key authentication and viewer, analyst, and approver roles
- Per-key rate limiting and structured API errors
- SHA-256 hash-chained SQLite audit trail
- Tamper verification for audit events
- Human recommendation approval workflow
- Read-only agent and MCP tool boundaries
- OpenAPI documentation at `/docs`

## Dashboard

The responsive React dashboard provides grouped workspace navigation:

1. **Operations** — overview, generation, maintenance, grid, renewables, storage, losses, climate, Digital Twin, and Fault Hub
2. **Analysis** — Kenya problem analysis, reserve risk, curtailment, stress tests, fleet, theft planning, and demand structure
3. **Governance** — recommendations, audit trail, and integration status
4. **AI and Query** — AI Command Centre, Copilot, and Query Studio

A grouped workspace dropdown is available alongside the professional Operations sidebar. Relevant forecast windows, chart series, source classes, and analysis controls are presented as dropdowns or compact filter controls so users can correlate their selections across views.

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, TypeScript, Tailwind CSS, Recharts, MapLibre GL, Leaflet, Vite |
| Backend | Python, FastAPI, Pydantic, Uvicorn |
| Machine Learning | LightGBM, XGBoost, scikit-learn, Isolation Forest, KMeans, statsmodels |
| Optimisation | OR-Tools Linear Programming |
| Data Processing | Pandas, NumPy, SciPy |
| Search | Local TF-IDF and BM25-style retrieval |
| Persistence | SQLite for audit and approval records; in-memory prototype state |
| AI Integration | OpenAI API with structured local fallback |
| Testing | pytest and Vitest |

## Measured Model Performance

Performance below is measured on held-out windows of the project's synthetic dataset.

| Capability | Key Result |
|---|---|
| National demand | 24-hour MAPE 1.46%, compared with 2.27% for same-hour-last-week |
| Predictive maintenance | XGBoost F1 0.43, precision 0.30, recall 0.77; baseline F1 0.32 |
| Anomaly detection | Isolation Forest F1 0.79, compared with 0.74 for z-score baseline |
| Loss detection | Hybrid Isolation Forest and loss-threshold method achieved perfect held-out separation |

Detailed metrics are available in [`docs/METRICS.md`](docs/METRICS.md).

## Quick Start

### Backend

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests -q
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The backend trains and initializes its synthetic world on startup. Health status is available at:

```text
http://localhost:8000/health
```

API documentation is available at:

```text
http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm ci
npm run build
npm run dev -- --host 0.0.0.0
```

Open:

```text
http://localhost:5173/
```

The Vite development server proxies API calls to `http://localhost:8000`.

## Local Demo Keys

When `POWERIQ_ENV=local`, the following keys are available:

| Key | Role |
|---|---|
| `demo-viewer-key` | Read-only dashboards |
| `demo-analyst-key` | Analysis, scenarios, optimisation, AI, reporting, and audit viewing |
| `demo-approver-key` | Recommendation decisions |

Keys are session-only in the browser. Never place private credentials in frontend source code.

## OpenAI Configuration

To enable OpenAI narration and planning, set the backend environment variable:

```bash
OPENAI_API_KEY=your_backend_key
POWERIQ_OPENAI_MODEL=gpt-4o-mini
```

Copy `.env.example` to `.env` for local configuration. Restart the backend after changing environment variables.

The key is not exposed to the browser. Query Studio reports whether OpenAI is configured and provides an optional analyst-level connection test. If OpenAI is unavailable or rejected, the application continues using its local narrative and planning fallback.

## Docker

The repository includes Docker and Compose configuration:

```bash
cp .env.example .env
docker compose up --build
```

The application is served on port `8000` when run through Docker. Docker availability and runtime permissions depend on the host environment.

## Project Structure

```text
backend/app/       FastAPI routes, synthetic world, models, optimisation, agents, RAG, MCP
backend/knowledge/  Local cited reference documents
backend/tests/      Backend test suite
frontend/src/       React dashboard, pages, charts, maps, and API client
examples/           External fault-detection submission example
e2e/                Repeatable browser sweep
infra/azure/        Azure deployment script
docs/               Metrics and technology documentation
```

## Important Limitations

- Grid values, asset records, telemetry, weather, demand, and incident data are synthetic.
- Plant names may reference real facilities, but capacities and operating series are illustrative.
- The DC power-flow model uses assumed ratings and is not a validated AC power-flow or utility control model.
- Loss detection performance reflects injected synthetic effect sizes and may not generalize to real feeder data.
- Storage prices and merit-order dispatch are modelling proxies.
- Fault detections and community reports are accepted as submitted; the platform does not verify images or identities.
- Response targets and external tile services are not operational grid policies.

## Testing

```bash
cd backend
python -m pytest tests -q

cd ../frontend
npm test
npm run build
```
# -POWERIQ-KENYA
