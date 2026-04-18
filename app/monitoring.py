import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.metric_preset import DataDriftPreset


current = "app/current_data.csv"
test = "app/test_data.csv"


def generate_drift_report():
    reference = pd.read_csv("app/reference_data.csv")
    current = pd.read_csv(test) 
    
    column_mapping = ColumnMapping()
    
    column_mapping.target = "HeartDisease"

    column_mapping.numerical_features = [
        "Age", "RestingBP", "Cholesterol", "FastingBS",  "MaxHR", "Oldpeak"
    ]

    column_mapping.categorical_features = [
        "Sex", "ChestPainType", "RestingECG",
        "ExerciseAngina", "ST_Slope"
    ]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
    report.save_html("drift_report.html")