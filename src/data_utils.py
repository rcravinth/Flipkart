import re
import pandas as pd
import numpy as np
from .config import TARGET_COLUMN, TARGET_CANDIDATES, TEXT_CANDIDATES

def normalize_name(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

def find_column(df, candidates):
    normalized = {normalize_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    return None

def load_data(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def detect_target(df):
    if TARGET_COLUMN:
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"TARGET_COLUMN={TARGET_COLUMN!r} is not present. Columns: {list(df.columns)}")
        return TARGET_COLUMN
    target = find_column(df, TARGET_CANDIDATES)
    if target is None:
        raise ValueError(
            "Could not automatically identify the target column. "
            "Set TARGET_COLUMN in src/config.py. "
            f"Available columns: {list(df.columns)}"
        )
    return target

def detect_text_column(df, target):
    candidate = find_column(df, TEXT_CANDIDATES)
    if candidate and candidate != target:
        return candidate
    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    object_cols = [c for c in object_cols if c != target]
    if not object_cols:
        return None
    scores = {}
    for c in object_cols:
        sample = df[c].dropna().astype(str).head(1000)
        scores[c] = sample.str.len().mean() if len(sample) else 0
    return max(scores, key=scores.get) if scores else None

def clean_data(df, target):
    out = df.copy()
    out = out.drop_duplicates()
    out = out.dropna(subset=[target])
    # Trim string values.
    for c in out.select_dtypes(include=["object", "string"]).columns:
        out[c] = out[c].astype("string").str.strip()
    return out

def add_time_features(df):
    out = df.copy()
    for c in list(out.columns):
        if "date" in normalize_name(c) or "time" in normalize_name(c):
            parsed = pd.to_datetime(out[c], errors="coerce")
            if parsed.notna().mean() >= 0.5:
                out[c + "_year"] = parsed.dt.year
                out[c + "_month"] = parsed.dt.month
                out[c + "_day"] = parsed.dt.day
                out[c + "_hour"] = parsed.dt.hour
                out[c + "_dayofweek"] = parsed.dt.dayofweek
    return out

def add_text_length(df, text_col):
    out = df.copy()
    if text_col:
        out["text_length"] = out[text_col].fillna("").astype(str).str.len()
        out["word_count"] = out[text_col].fillna("").astype(str).str.split().str.len()
    return out

def make_binary_target(y):
    """
    Converts numeric 1-5 satisfaction scores into:
    0 = unsatisfied (1-3)
    1 = satisfied (4-5)

    For non-numeric labels, preserves the original labels.
    """
    numeric = pd.to_numeric(y, errors="coerce")
    if numeric.notna().mean() >= 0.95 and numeric.nunique() <= 10:
        return (numeric >= 4).astype(int)
    return y.astype(str)
