import os
import tempfile
import warnings

warnings.filterwarnings("ignore")
_tmp = tempfile.mkdtemp()
os.environ["POWERIQ_AUDIT_DB"] = os.path.join(_tmp, "audit.db")
os.environ.pop("OPENAI_API_KEY", None)
os.environ["POWERIQ_ENV"] = "local"

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.service import get_service

VIEWER = {"X-API-Key": "demo-viewer-key"}
ANALYST = {"X-API-Key": "demo-analyst-key"}
APPROVER = {"X-API-Key": "demo-approver-key"}


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def svc(client):
    return get_service()
