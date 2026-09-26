"""
FastAPI model serving — loads ChurnModel/Production from MLflow registry.

Run:
  uvicorn serve:app --host 0.0.0.0 --port 8001
"""

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MLFLOW_URI = "http://127.0.0.1:5000"
MODEL_URI = "models:/ChurnModel/Production"

mlflow.set_tracking_uri(MLFLOW_URI)

app = FastAPI(title="Telco Churn Prediction API")
_model = None


def get_model():
    global _model
    if _model is None:
        _model = mlflow.pyfunc.load_model(MODEL_URI)
    return _model


class CustomerFeatures(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: float
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_label: str


@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    try:
        model = get_model()
        df = pd.DataFrame([features.model_dump()])
        pred = int(model.predict(df)[0])
        return PredictionResponse(
            churn_prediction=pred,
            churn_label="Churn" if pred == 1 else "No Churn",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
