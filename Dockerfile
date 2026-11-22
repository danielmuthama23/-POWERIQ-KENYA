# Stage 1: build the React frontend
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# Stage 2: Python API serving the built frontend
FROM python:3.13-slim AS api
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 POWERIQ_ENV=local POWERIQ_AUDIT_DB=/data/audit.db
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install -r requirements.txt
COPY backend/ ./
COPY --from=web /web/dist /app/frontend/dist
RUN useradd -m app && mkdir -p /data && chown app /data
USER app
EXPOSE 8000
# Models are trained at startup (about 25 s), so allow a generous start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 CMD curl -fs http://localhost:8000/health | grep -q '"ready":true' || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
