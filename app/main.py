"""Phase 4 - churn prediction REST API (FastAPI).

Loads the production model + scaler + feature list at startup and exposes
individual and batch inference. Preprocessing mirrors training exactly.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DATA_DIR = os.getenv("DATA_DIR", "data")
NUM_FEATURES = ["tenure", "MonthlyCharges", "support_tickets"]

app = FastAPI(
    title="Churn Predictor API",
    description="Fintech churn prediction - MLOps pipeline",
    version="1.0.0",
)

# Loaded once at import (module) time; falls back to None so /health can report.
model = None
scaler = None
feature_names: List[str] = []
model_info: dict = {}


def _load_artifacts() -> None:
    global model, scaler, feature_names, model_info
    model = joblib.load(os.path.join(DATA_DIR, "production_model.pkl"))
    scaler = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))
    feature_names = pd.read_csv(
        os.path.join(DATA_DIR, "feature_names.csv"), header=None
    )[0].tolist()
    info_path = os.path.join(DATA_DIR, "production_model_info.json")
    model_info = json.load(open(info_path)) if os.path.exists(info_path) else {}


try:
    _load_artifacts()
except Exception as exc:  # pragma: no cover - surfaced via /health
    print(f"WARNING: could not load model artifacts: {exc}")


class CustomerFeatures(BaseModel):
    gender: str = Field(..., examples=["Male"])
    senior_citizen: int = Field(..., ge=0, le=1)
    tenure_months: int = Field(..., ge=0)
    contract_type: str = Field(..., examples=["Month-to-Month"])
    internet_service: Optional[str] = Field(None, examples=["Fiber"])
    monthly_charges: float = Field(..., ge=0)
    support_tickets_last_6m: int = Field(..., ge=0)
    customer_id: Optional[str] = None


class BatchRequest(BaseModel):
    customers: List[CustomerFeatures]


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    customer_id: Optional[str]
    churn_prediction: int
    churn_label: str
    churn_probability: float
    risk_level: str
    recommendation: str
    model_version: str


def _risk_level(prob: float) -> str:
    return "ALTO" if prob >= 0.6 else ("MEDIO" if prob >= 0.3 else "BAJO")


def _recommendation(risk: str) -> str:
    return {
        "ALTO": "Contacto proactivo de retencion + oferta personalizada",
        "MEDIO": "Incluir en campana de fidelizacion",
        "BAJO": "Sin accion inmediata; seguimiento estandar",
    }[risk]


def preprocess_customer(data: dict) -> pd.DataFrame:
    """Manual encoding aligned to the trained feature set.

    We do NOT use get_dummies here: with a single row it would drop the only
    observed category and lose the signal. Instead we set each trained dummy
    column explicitly.
    """
    contract = (data.get("contract_type") or "").strip().lower()
    internet = (data.get("internet_service") or "No").strip().lower()

    row = {name: 0 for name in feature_names}
    row["gender"] = 1 if data["gender"].strip().lower() == "male" else 0
    row["SeniorCitizen"] = data["senior_citizen"]
    row["tenure"] = data["tenure_months"]
    row["MonthlyCharges"] = data["monthly_charges"]
    row["support_tickets"] = data["support_tickets_last_6m"]

    if "Contract_One Year" in row and contract in ("one year", "one-year"):
        row["Contract_One Year"] = 1
    if "Contract_Two Year" in row and contract in ("two year", "two-year"):
        row["Contract_Two Year"] = 1
    if "InternetService_Fiber" in row and "fiber" in internet:
        row["InternetService_Fiber"] = 1
    if "InternetService_No" in row and internet in ("no", "none", ""):
        row["InternetService_No"] = 1

    df = pd.DataFrame([row])[feature_names]
    df[NUM_FEATURES] = scaler.transform(df[NUM_FEATURES])
    return df


def _predict_one(features: CustomerFeatures) -> PredictionResponse:
    X = preprocess_customer(features.model_dump())
    prob = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])
    risk = _risk_level(prob)
    return PredictionResponse(
        customer_id=features.customer_id,
        churn_prediction=pred,
        churn_label="Churn" if pred == 1 else "No Churn",
        churn_probability=round(prob, 4),
        risk_level=risk,
        recommendation=_recommendation(risk),
        model_version=str(model_info.get("version", "unknown")),
    )


@app.get("/")
def root():
    return {"service": "Churn Predictor API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "degraded",
            "model_loaded": model is not None}


@app.get("/model/info")
def get_model_info():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model_info


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _predict_one(features)


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(request: BatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return [_predict_one(c) for c in request.customers]
