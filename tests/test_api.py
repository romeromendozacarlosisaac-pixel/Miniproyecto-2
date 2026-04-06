import sys
import os

sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_predict_returns_200():
    response = client.post("/predict", json={
        "Age": 52,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 125,
        "Cholesterol": 212,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 168,
        "ExerciseAngina": "N",
        "Oldpeak": 1.0,
        "ST_Slope": "Up"
    })
    assert response.status_code == 200


def test_predict_returns_probability():
    response = client.post("/predict", json={
        "Age": 52,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 125,
        "Cholesterol": 212,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 168,
        "ExerciseAngina": "N",
        "Oldpeak": 1.0,
        "ST_Slope": "Up"
    })
    data = response.json()
    assert "heart_disease_probability" in data
    assert "prediction" in data
    assert data["prediction"] in [0, 1]