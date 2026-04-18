from fastapi import FastAPI
from pydantic import BaseModel
from app.monitoring import generate_drift_report
from fastapi.responses import FileResponse
import joblib
import os
import pandas as pd


model = joblib.load("app/model.joblib")
app = FastAPI()

DATA_PATH = "app/current_data.csv"
REFERENCE_PATH = "app/reference_data.csv"


class Input(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str


@app.post("/predict")
def predict(data: Input):
    df = pd.DataFrame([data.model_dump()])
    
    #Asegurar mismo orden de columnas que entrenamiento
    if os.path.exists(REFERENCE_PATH):
        ref_cols = pd.read_csv(REFERENCE_PATH, nrows=1).columns
        df = df[ref_cols]
    
    proba = model.predict_proba(df)[0][1]
    
    #GUARDAR datos para monitoreo
    if os.path.exists(DATA_PATH):
        df.to_csv(DATA_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(DATA_PATH, index=False)
    
    return {
        "heart_disease_probability": round(proba, 4),
        "prediction": int(proba > 0.5)
    }
    
@app.get("/monitor")
def monitor():
    generate_drift_report()
    return {"message": "Drift report generated successfully"}

@app.get("/report")
def get_report():
    return FileResponse("drift_report.html")