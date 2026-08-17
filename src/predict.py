import joblib
import pandas as pd
from pathlib import Path
from .config import MODEL_DIR

MODEL_PATH = MODEL_DIR / "best_model.joblib"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python run_project.py")
    return joblib.load(MODEL_PATH)

def predict_dataframe(df):
    model = load_model()
    out = df.copy()
    out["prediction"] = model.predict(df)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)
        if proba.shape[1] == 2:
            out["satisfaction_probability"] = proba[:, 1]
    return out
