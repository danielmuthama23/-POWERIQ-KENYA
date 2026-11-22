"""Authentication, role-based access, rate limiting, tamper-evident audit log and recommendation approvals."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, Header, HTTPException, Request

ROLE_LEVEL = {"viewer": 1, "analyst": 2, "approver": 3}

# Demo keys exist ONLY in POWERIQ_ENV=local (the default). Production must supply POWERIQ_API_KEYS.
LOCAL_DEMO_KEYS = {"demo-viewer-key": "viewer", "demo-analyst-key": "analyst", "demo-approver-key": "approver"}


@dataclass
class Settings:
    env: str
    api_keys: dict[str, str]
    rate_limit_per_min: int
    audit_db: str
    openai_api_key: str | None
    openai_model: str
    agent_timeout_s: float


def load_settings() -> Settings:
    env = os.getenv("POWERIQ_ENV", "local")
    keys: dict[str, str] = {}
    raw = os.getenv("POWERIQ_API_KEYS", "")
    for pair in [p for p in raw.split(",") if p.strip()]:
        k, _, role = pair.strip().partition(":")
        if role not in ROLE_LEVEL:
            raise RuntimeError(f"POWERIQ_API_KEYS entry has invalid role: {role!r}")
        keys[k] = role
    if not keys:
        if env == "production":
            raise RuntimeError("POWERIQ_API_KEYS must be set when POWERIQ_ENV=production")
        keys = dict(LOCAL_DEMO_KEYS)
    return Settings(env=env, api_keys=keys, rate_limit_per_min=int(os.getenv("POWERIQ_RATE_LIMIT", "240")),
                    audit_db=os.getenv("POWERIQ_AUDIT_DB", "audit.db"), openai_api_key=os.getenv("OPENAI_API_KEY") or None,
                    openai_model=os.getenv("POWERIQ_OPENAI_MODEL", "gpt-4o-mini"),
                    agent_timeout_s=float(os.getenv("POWERIQ_AGENT_TIMEOUT", "45")))


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reset_settings() -> None:
    global _settings, _audit, _limiter
    _settings = None
    _audit = None
    _limiter = None


# ------------------------------------------------------------------ rate limiting
class RateLimiter:
    def __init__(self, per_min: int):
        self.per_min = per_min
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key_id: str) -> int | None:
        """Returns None if allowed, else seconds to wait."""
        now = time.monotonic()
        with self.lock:
            q = self.hits[key_id]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_min:
                return int(60 - (now - q[0])) + 1
            q.append(now)
        return None


_limiter: RateLimiter | None = None


def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(get_settings().rate_limit_per_min)
    return _limiter


# ------------------------------------------------------------------ audit log (hash chained)
class AuditLog:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS audit (
            seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, event_type TEXT, actor TEXT, role TEXT, payload TEXT,
            prev_hash TEXT, hash TEXT)""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS recommendations (
            id TEXT PRIMARY KEY, created TEXT, source TEXT, text TEXT, status TEXT, requested_by TEXT,
            decided_by TEXT, decided_at TEXT, note TEXT, context TEXT)""")
        self.db.commit()

    @staticmethod
    def _digest(prev: str, ts: str, et: str, actor: str, role: str, payload: str) -> str:
        return hashlib.sha256("|".join([prev, ts, et, actor, role, payload]).encode()).hexdigest()

    def append(self, event_type: str, actor: str, role: str, payload: dict[str, Any]) -> str:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + 3 * 3600)) + "+03:00"
        body = json.dumps(payload, sort_keys=True, default=str)
        with self.lock:
            row = self.db.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row[0] if row else "GENESIS"
            h = self._digest(prev, ts, event_type, actor, role, body)
            self.db.execute("INSERT INTO audit (ts,event_type,actor,role,payload,prev_hash,hash) VALUES (?,?,?,?,?,?,?)",
                            (ts, event_type, actor, role, body, prev, h))
            self.db.commit()
        return h

    def events(self, limit: int = 100, event_type: str | None = None) -> list[dict]:
        q = "SELECT seq,ts,event_type,actor,role,payload,prev_hash,hash FROM audit"
        args: tuple = ()
        if event_type:
            q += " WHERE event_type=?"
            args = (event_type,)
        q += " ORDER BY seq DESC LIMIT ?"
        with self.lock:
            rows = self.db.execute(q, args + (limit,)).fetchall()
        return [{"seq": r[0], "ts": r[1], "event_type": r[2], "actor": r[3], "role": r[4], "payload": json.loads(r[5]),
                 "prev_hash": r[6], "hash": r[7]} for r in rows]

    def verify(self) -> dict:
        with self.lock:
            rows = self.db.execute("SELECT seq,ts,event_type,actor,role,payload,prev_hash,hash FROM audit ORDER BY seq").fetchall()
        prev = "GENESIS"
        for r in rows:
            if r[6] != prev or self._digest(prev, r[1], r[2], r[3], r[4], r[5]) != r[7]:
                return {"valid": False, "broken_at_seq": r[0], "events_checked": len(rows)}
            prev = r[7]
        return {"valid": True, "events_checked": len(rows), "head_hash": prev,
                "note": "Hash chain is tamper-evident locally. Anchoring head_hash on Hedera testnet is NOT implemented in this build."}

    # recommendations ---------------------------------------------------------
    def add_recommendation(self, source: str, text: str, requested_by: str, context: dict | None = None) -> str:
        rid = "REC-" + uuid.uuid4().hex[:8].upper()
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + 3 * 3600)) + "+03:00"
        with self.lock:
            self.db.execute("INSERT INTO recommendations (id,created,source,text,status,requested_by,context) VALUES (?,?,?,?,?,?,?)",
                            (rid, ts, source, text, "pending", requested_by, json.dumps(context or {}, default=str)))
            self.db.commit()
        return rid

    def list_recommendations(self, status: str | None = None, limit: int = 100) -> list[dict]:
        q = "SELECT id,created,source,text,status,requested_by,decided_by,decided_at,note FROM recommendations"
        args: tuple = ()
        if status:
            q += " WHERE status=?"
            args = (status,)
        q += " ORDER BY created DESC LIMIT ?"
        with self.lock:
            rows = self.db.execute(q, args + (limit,)).fetchall()
        keys = ["id", "created", "source", "text", "status", "requested_by", "decided_by", "decided_at", "note"]
        return [dict(zip(keys, r)) for r in rows]

    def decide(self, rid: str, decision: str, by: str, note: str | None) -> dict | None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + 3 * 3600)) + "+03:00"
        with self.lock:
            cur = self.db.execute("UPDATE recommendations SET status=?, decided_by=?, decided_at=?, note=? WHERE id=? AND status='pending'",
                                  (decision, by, ts, note, rid))
            self.db.commit()
            if cur.rowcount == 0:
                return None
        return next((r for r in self.list_recommendations(limit=500) if r["id"] == rid), None)


_audit: AuditLog | None = None


def get_audit() -> AuditLog:
    global _audit
    if _audit is None:
        _audit = AuditLog(get_settings().audit_db)
    return _audit


# ------------------------------------------------------------------ auth dependency
@dataclass
class Principal:
    key_id: str
    role: str


def _key_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:10]


def authenticate(request: Request, x_api_key: str | None = Header(default=None), authorization: str | None = Header(default=None)) -> Principal:
    key = x_api_key or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else None)
    st = get_settings()
    if not key:
        get_audit().append("auth_failure", "anonymous", "none", {"reason": "missing_credentials", "path": request.url.path})
        raise HTTPException(401, detail={"code": "unauthenticated", "message": "Provide X-API-Key or Bearer token."})
    role = None
    for k, r in st.api_keys.items():
        if hmac.compare_digest(k.encode(), key.encode()):
            role = r
    if role is None:
        get_audit().append("auth_failure", _key_id(key), "none", {"reason": "invalid_credentials", "path": request.url.path})
        raise HTTPException(401, detail={"code": "invalid_credentials", "message": "Invalid API key."})
    p = Principal(_key_id(key), role)
    wait = get_limiter().check(p.key_id)
    if wait is not None:
        raise HTTPException(429, detail={"code": "rate_limited", "message": f"Rate limit exceeded. Retry in {wait}s."},
                            headers={"Retry-After": str(wait)})
    request.state.principal = p
    return p


def require(min_role: str):
    def dep(p: Principal = Depends(authenticate)) -> Principal:
        if ROLE_LEVEL[p.role] < ROLE_LEVEL[min_role]:
            get_audit().append("access_denied", p.key_id, p.role, {"required": min_role})
            raise HTTPException(403, detail={"code": "forbidden", "message": f"Requires role '{min_role}' or higher."})
        return p
    return dep
