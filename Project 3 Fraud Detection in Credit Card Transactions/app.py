import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("Credit Card Fraud Detection")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model/xgb_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler


try:
    model, scaler = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    st.error(f"Model not found. Run train_model.py first.\n\n{e}")
    artifacts_loaded = False


def prepare_input(df, scaler):
    df = df.copy()
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])
    df[["Amount", "Time"]] = scaler.transform(df[["Amount", "Time"]])
    df["iso_score"] = 0.0
    df["lof_score"] = 0.0
    return df


if artifacts_loaded:
    tab1, tab2 = st.tabs(["Manual Input", "Upload CSV"])

    with tab1:
        st.subheader("Enter Transaction Features")

        col1, col2, col3 = st.columns(3)
        inputs = {}

        with col1:
            inputs["Time"] = st.number_input("Time", value=0.0)
            for i in range(1, 10):
                inputs[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

        with col2:
            for i in range(10, 19):
                inputs[f"V{i}"] = st.number_input(f"V{i}", value=0.0)

        with col3:
            for i in range(19, 29):
                inputs[f"V{i}"] = st.number_input(f"V{i}", value=0.0)
            inputs["Amount"] = st.number_input("Amount", value=0.0)

        if st.button("Predict Fraud", key="manual"):
            row = pd.DataFrame([inputs])
            row_prepared = prepare_input(row, scaler)
            prob = model.predict_proba(row_prepared)[0][1]
            label = "Fraud" if prob >= 0.5 else "Not Fraud"

            st.markdown("---")
            if label == "Fraud":
                st.error(f"Prediction: **{label}**")
            else:
                st.success(f"Prediction: **{label}**")
            st.metric("Fraud Probability", f"{prob:.4f}")

    with tab2:
        st.subheader("Upload CSV File")
        uploaded = st.file_uploader("Upload a CSV with transaction features", type=["csv"])

        if uploaded is not None:
            df_upload = pd.read_csv(uploaded)
            st.write("Preview:", df_upload.head())

            if st.button("Predict Fraud", key="csv"):
                df_prepared = prepare_input(df_upload, scaler)
                probs = model.predict_proba(df_prepared)[:, 1]
                labels = ["Fraud" if p >= 0.5 else "Not Fraud" for p in probs]

                df_upload["Prediction"] = labels
                df_upload["Fraud Probability"] = probs

                st.dataframe(df_upload[["Prediction", "Fraud Probability"]].head(50))

                fraud_count = labels.count("Fraud")
                st.info(f"Fraudulent transactions detected: {fraud_count} / {len(labels)}")

    st.markdown("---")
    st.subheader("Model Evaluation")

    col_roc, col_cm = st.columns(2)

    with col_roc:
        if os.path.exists("static/roc_curve.png"):
            st.image("static/roc_curve.png", caption="ROC Curve", use_container_width=True)
        else:
            st.warning("ROC Curve not found. Run train_model.py.")

    with col_cm:
        if os.path.exists("static/confusion_matrix.png"):
            st.image("static/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
        else:
            st.warning("Confusion Matrix not found. Run train_model.py.")
