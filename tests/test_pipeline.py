import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api import app, features, model

client = TestClient(app)


def load_example_request():
    return json.loads((PROJECT_ROOT / "example_request.json").read_text())


def test_model_files_can_be_loaded():
    model_path = PROJECT_ROOT / "models" / "model.joblib"
    features_path = PROJECT_ROOT / "models" / "features.json"

    loaded_model = joblib.load(model_path)
    loaded_features = json.loads(features_path.read_text())

    assert loaded_model is not None
    assert loaded_features
    assert loaded_features == features
    assert model is not None


def test_health_endpoint_reports_loaded_model():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_prediction_endpoint_returns_expected_structure_and_types():
    response = client.post("/predict", json=load_example_request())

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "prediction",
        "probability_of_class_1",
        "message",
        "note",
    }
    assert isinstance(body["prediction"], int)
    assert body["prediction"] in (0, 1)
    assert isinstance(body["probability_of_class_1"], float)
    assert 0.0 <= body["probability_of_class_1"] <= 1.0
    assert isinstance(body["message"], str)
    assert isinstance(body["note"], str)


def test_prediction_rejects_missing_features():
    response = client.post("/predict", json={"features": {"age": 52}})

    assert response.status_code == 400
    assert "missing_features" in response.json()["detail"]


def test_prediction_rejects_malformed_request():
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_loaded_model_accepts_example_feature_values():
    request = load_example_request()
    row = {name: request["features"][name] for name in features}
    prediction = model.predict(pd.DataFrame([row], columns=features))[0]

    assert int(prediction) in (0, 1)
