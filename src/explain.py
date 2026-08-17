import json
from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_DIR, REPORT_DIR

def export_feature_importance():
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    pre = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    if not hasattr(estimator, "feature_importances_"):
        return "Selected model does not expose feature_importances_. Use SHAP for model-agnostic explanations."

    try:
        names = pre.get_feature_names_out()
    except Exception:
        names = [f"feature_{i}" for i in range(len(estimator.feature_importances_))]

    importance = pd.DataFrame({
        "feature": names,
        "importance": estimator.feature_importances_
    }).sort_values("importance", ascending=False)

    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    return importance.head(25)

if __name__ == "__main__":
    print(export_feature_importance())
