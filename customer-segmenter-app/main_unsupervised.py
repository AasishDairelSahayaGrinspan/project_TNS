"""FastAPI cluster assignment service (inference only)."""
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
KMEANS_PKL = APP_DIR / "kmeans.pkl"
SCALER_PKL = APP_DIR / "scaler.pkl"

if not KMEANS_PKL.exists():
    raise RuntimeError(f"missing model file: {KMEANS_PKL}")
if not SCALER_PKL.exists():
    raise RuntimeError(f"missing scaler file: {SCALER_PKL}")

kmeans = joblib.load(KMEANS_PKL)
scaler = joblib.load(SCALER_PKL)

PERSONAS = getattr(kmeans, "cluster_to_persona_", None) or {0: "Budget", 1: "Standard", 2: "Premium"}

app = FastAPI(title="customer-segmenter")


class PredictRequest(BaseModel):
    annual_income: float | None = None
    annual_income_k: float | None = None
    spending_score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    income = req.annual_income if req.annual_income is not None else req.annual_income_k
    if income is None:
        raise HTTPException(status_code=422, detail="annual_income is required")
    try:
        Xs = scaler.transform([[float(income), float(req.spending_score)]])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    cluster = int(kmeans.predict(Xs)[0])
    persona = str(PERSONAS.get(cluster, "Unknown"))
    return {"cluster": cluster, "persona": persona}
