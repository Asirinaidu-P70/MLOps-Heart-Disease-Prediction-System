from pathlib import Path
import json
from typing import Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

MODEL_PATH = Path("models/model.joblib")
FEATURES_PATH = Path("models/features.json")
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")

if MODEL_PATH.exists() and FEATURES_PATH.exists():
    model = joblib.load(MODEL_PATH)
    features = json.loads(FEATURES_PATH.read_text())
else:
    model = None
    features = []


class PredictionRequest(BaseModel):
    features: Dict[str, float]


@app.get("/app", include_in_schema=False)
def frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/")
def root():
    return {"message": "Heart Disease Prediction API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found. Run training first.")

    missing = [f for f in features if f not in request.features]
    if missing:
        raise HTTPException(status_code=400, detail={"missing_features": missing})

    row = {f: request.features[f] for f in features}
    X = pd.DataFrame([row], columns=features)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0, 1])

    return {
        "prediction": prediction,
        "probability_of_class_1": round(probability, 4),
        "message": "Model predicts higher risk" if prediction == 1 else "Model predicts lower risk",
        "note": "Demo/academic prediction only; not a medical diagnosis.",
    }
