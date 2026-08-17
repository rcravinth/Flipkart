import streamlit as st
import pandas as pd

from src.predict import load_model

st.set_page_config(page_title="Flipkart CSAT Predictor", layout="wide")
st.title("Flipkart Customer Service Satisfaction Predictor")
st.caption("ML prediction interface for customer-service satisfaction.")

try:
    model = load_model()
except Exception as e:
    st.error(str(e))
    st.stop()

uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Input preview")
    st.dataframe(df.head(20), use_container_width=True)

    try:
        pred = model.predict(df)
        result = df.copy()
        result["prediction"] = pred

        if hasattr(model, "predict_proba") and len(model.classes_) == 2:
            result["satisfaction_probability"] = model.predict_proba(df)[:, 1]

        st.success("Prediction completed")
        st.dataframe(result.head(100), use_container_width=True)

        st.download_button(
            "Download predictions",
            result.to_csv(index=False).encode("utf-8"),
            "predictions.csv",
            "text/csv"
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("The uploaded CSV must contain the same feature columns used during model training.")
