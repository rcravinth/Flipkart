import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import DATA_PATH, REPORT_DIR
from .data_utils import load_data, detect_target, clean_data

def run_eda():
    REPORT_DIR.mkdir(exist_ok=True)
    df = load_data(DATA_PATH)
    target = detect_target(df)
    df = clean_data(df, target)

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "target": target,
        "missing_values": {str(k): int(v) for k, v in df.isna().sum().items() if v > 0},
        "duplicates": int(df.duplicated().sum()),
        "target_distribution": {str(k): int(v) for k, v in df[target].value_counts(dropna=False).items()}
    }

    with open(REPORT_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    # Target distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=target, order=df[target].value_counts().index)
    plt.title("Customer Satisfaction Distribution")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "target_distribution.png", dpi=180)
    plt.close()

    # Numeric correlation
    numeric = df.select_dtypes(include=np.number)
    if numeric.shape[1] >= 2:
        plt.figure(figsize=(10, 7))
        sns.heatmap(numeric.corr(), cmap="coolwarm", center=0)
        plt.title("Numeric Feature Correlation")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=180)
        plt.close()

    return summary

if __name__ == "__main__":
    print(run_eda())
