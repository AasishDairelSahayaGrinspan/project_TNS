"""Tests for FastAPI predict service (port 8001)."""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_approval_json():
    payload = {
        "income": 60000,
        "credit_score": 720,
        "loan_amount": 15000,
        "employment_years": 5,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert "probability" in body
    assert isinstance(body["prediction"], int)
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_missing_field():
    r = client.post("/predict", json={"income": 50000})
    assert r.status_code == 422
