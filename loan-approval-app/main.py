"""FastAPI loan approval predict service on port 8001."""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.pkl"
SCALER_PATH = APP_DIR / "scaler.pkl"

FEATURES = ["income", "credit_score", "loan_amount", "employment_years"]

app = FastAPI(title="Loan Approval API")

_model = joblib.load(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)


class LoanRequest(BaseModel):
    income: float
    credit_score: float
    loan_amount: float
    employment_years: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: LoanRequest):
    x = pd.DataFrame(
        [[req.income, req.credit_score, req.loan_amount, req.employment_years]],
        columns=FEATURES,
    )
    xs = _scaler.transform(x)
    pred = int(_model.predict(xs)[0])
    proba = float(_model.predict_proba(xs)[0].max())
    return {"prediction": pred, "probability": proba}
