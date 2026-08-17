import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from .config import DATA_PATH, MODEL_DIR, REPORT_DIR, RANDOM_STATE, TEST_SIZE, MAX_TEXT_FEATURES
from .data_utils import load_data, detect_target, detect_text_column, clean_data, add_time_features, add_text_length, make_binary_target

warnings.filterwarnings("ignore")

def prepare_features(df, target):
    df = add_time_features(df)
    text_col = detect_text_column(df, target)
    df = add_text_length(df, text_col)

    y = make_binary_target(df[target])
    X = df.drop(columns=[target]).copy()

    # Drop high-cardinality identifiers where possible.
    drop_cols = []
    for c in X.columns:
        n = X[c].nunique(dropna=True)
        if n > 0.98 * len(X) and n > 50:
            drop_cols.append(c)
    X = X.drop(columns=drop_cols, errors="ignore")

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols and c != text_col]

    transformers = []

    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False))
            ]),
            numeric_cols
        ))

    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_cols
        ))

    if text_col and text_col in X.columns:
        X[text_col] = X[text_col].fillna("").astype(str)
        transformers.append((
            "text",
            Pipeline([
                ("tfidf", TfidfVectorizer(
                    max_features=MAX_TEXT_FEATURES,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True
                ))
            ]),
            text_col
        ))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return X, y, preprocessor, text_col

def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
    }
    if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
        proba = model.predict_proba(X_test)[:, 1]
        result["roc_auc"] = roc_auc_score(y_test, proba)
    else:
        result["roc_auc"] = np.nan

    cm = confusion_matrix(y_test, pred)
    return result, pred, cm

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = load_data(DATA_PATH)
    target = detect_target(df)
    df = clean_data(df, target)

    X, y, preprocessor, text_col = prepare_features(df, target)

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=350, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=350, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE)
    }

    results = []
    fitted = {}

    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])
        pipe.fit(X_train, y_train)
        metrics, pred, cm = evaluate(name, pipe, X_test, y_test)
        results.append(metrics)
        fitted[name] = pipe

    metrics_df = pd.DataFrame(results).sort_values("f1_weighted", ascending=False)
    metrics_df.to_csv(REPORT_DIR / "model_metrics.csv", index=False)

    best_name = metrics_df.iloc[0]["model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

    pred = best_model.predict(X_test)
    pred_df = X_test.copy()
    pred_df["actual"] = y_test.values
    pred_df["prediction"] = pred
    if hasattr(best_model, "predict_proba") and len(best_model.classes_) == 2:
        pred_df["prediction_probability"] = best_model.predict_proba(X_test)[:, 1]
    pred_df.to_csv(REPORT_DIR / "predictions.csv", index=False)

    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)
    ax.set_title(f"Confusion Matrix — {best_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    if len(np.unique(y_test)) == 2 and hasattr(best_model, "predict_proba"):
        RocCurveDisplay.from_estimator(best_model, X_test, y_test)
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "roc_curve.png", dpi=180)
        plt.close()

    metadata = {
        "target_column": target,
        "text_column": text_col,
        "best_model": best_name,
        "rows_after_cleaning": int(len(df)),
        "feature_columns": list(X.columns),
        "classes": [str(x) for x in best_model.classes_]
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(metrics_df.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved: {MODEL_DIR / 'best_model.joblib'}")

if __name__ == "__main__":
    main()
