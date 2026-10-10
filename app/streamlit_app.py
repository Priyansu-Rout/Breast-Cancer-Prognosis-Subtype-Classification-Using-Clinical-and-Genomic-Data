import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.breast_cancer_ml.config.settings import settings

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prognosis Dashboard",
    page_icon="🎗️",
    layout="wide"
)

# Medical Disclaimer
st.sidebar.warning(
    "**MEDICAL DISCLAIMER**\n\n"
    "This project is intended for research and educational purposes only. "
    "It is not a medical diagnostic tool and should not be used to make "
    "clinical or treatment decisions."
)

def load_model(group, model_type="survival"):
    """Loads the trained model and preprocessor."""
    if model_type == "survival":
        path = Path(f"models/survival/{group}/random_survival_forest.joblib")
    else:
        path = Path(f"models/classification/{group}/xgboost.joblib")

    if not path.exists():
        return None
    return joblib.load(path)

def load_label_encoder(group):
    path = Path(f"models/classification/{group}/label_encoder.joblib")
    return joblib.load(path) if path.exists() else None

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Overview",
    "Data Explorer",
    "Survival Prediction",
    "Subtype Prediction",
    "Model Comparison",
    "Biomarkers"
])

# --- Page: Overview ---
if page == "Overview":
    st.title("🎗️ Integrated Clinical & Genomic Breast Cancer Prognosis")
    st.markdown("""
    ### Project Overview
    This research project investigates whether integrating **Clinical characteristics** with **Molecular gene-expression**
    and **Mutation information** improves breast-cancer prognosis and molecular subtype classification.

    **Core Research Question:**
    *Does Clinical + Genomic data outperform Clinical-only or Genomic-only models?*

    #### Key Features:
    - **Survival Analysis**: Using CoxPH and Random Survival Forests.
    - **Subtype Classification**: Using XGBoost and Random Forests.
    - **Explainable AI**: SHAP-based biomarker discovery.
    - **MLOps**: Fully reproducible pipeline managed by DVC.
    """)

    # Display a sample figure
    km_path = Path("reports/figures/integrated_km.png")
    if km_path.exists():
        st.image(str(km_path), caption="Integrated Model Risk Stratification (Kaplan-Meier)", use_column_width=True)

# --- Page: Data Explorer ---
elif page == "Data Explorer":
    st.title("📊 Data Explorer")
    raw_path = Path(settings.get("data.raw_path"))
    if raw_path.exists():
        df = pd.read_csv(raw_path).head(100)
        st.write("Preview of METABRIC Dataset (First 100 rows):")
        st.dataframe(df)
    else:
        st.error("Dataset not found. Please run the pipeline first.")

# --- Page: Survival Prediction ---
elif page == "Survival Prediction":
    st.title("📈 Survival Risk Prediction")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Patient Inputs")
        group = st.selectbox("Select Model Arm", ["clinical", "genomic", "integrated"])

        # In a real app, we'd list the actual features from the model
        # For this demo, we use a generic input form
        age = st.slider("Age at Diagnosis", 20, 90, 50)
        tumor_size = st.slider("Tumor Size (mm)", 1, 100, 20)
        stage = st.selectbox("Tumor Stage", ["I", "II", "III", "IV"])
        grade = st.selectbox("Histologic Grade", ["1", "2", "3"])

        predict_btn = st.button("Predict Risk")

    with col2:
        st.subheader("Prediction Result")
        if predict_btn:
            model = load_model(group, "survival")
            if model:
                # Create a dummy feature vector matching the model's expected input
                # In production, this would map precisely to the X.csv columns
                X_processed = Path(f"data/processed/test_{group}/X.csv")
                if X_processed.exists():
                    sample_df = pd.read_csv(X_processed).iloc[[0]]
                    # We mock the modification of the sample with user inputs
                    # (Simplified for the UI demo)
                    risk_score = model.predict(sample_df)[0]

                    st.metric("Risk Score", f"{risk_score:.4f}")

                    # Mock categorization
                    category = "High Risk" if risk_score > 0.5 else "Low Risk" # Simplified threshold
                    st.info(f"Patient Category: **{category}**")

                    st.warning("Note: This is a research prediction based on the RSF model. It is not a clinical diagnosis.")
                else:
                    st.error("Feature matrices not found. Please run `dvc repro`.")
            else:
                st.error("Model artifact not found.")

# --- Page: Subtype Prediction ---
elif page == "Subtype Prediction":
    st.title("🧬 Molecular Subtype Prediction")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Patient Inputs")
        group = st.selectbox("Select Model Arm", ["clinical", "genomic", "integrated"])
        predict_btn = st.button("Predict Subtype")

    with col2:
        st.subheader("Prediction Result")
        if predict_btn:
            model = load_model(group, "classification")
            le = load_label_encoder(group)

            if model and le:
                X_processed = Path(f"data/processed/test_{group}/X.csv")
                if X_processed.exists():
                    sample_df = pd.read_csv(X_processed).iloc[[0]]
                    prob = model.predict_proba(sample_df)[0]
                    pred_idx = np.argmax(prob)
                    subtype = le.inverse_transform([pred_idx])[0]

                    st.success(f"Predicted Subtype: **{subtype}**")

                    # Probability Distribution
                    prob_df = pd.DataFrame({
                        "Subtype": le.classes_,
                        "Probability": prob
                    }).sort_values(by="Probability", ascending=False)

                    st.bar_chart(prob_df.set_index("Subtype"))
                else:
                    st.error("Feature matrices not found.")
            else:
                st.error("Model or Label Encoder not found.")

# --- Page: Model Comparison ---
elif page == "Model Comparison":
    st.title("⚖️ Model Comparison Dashboard")

    tab1, tab2 = st.tabs(["Survival (C-index)", "Classification (F1/AUC)"])

    with tab1:
        comp_path = Path("reports/metrics/survival_comparison.csv")
        if comp_path.exists():
            df = pd.read_csv(comp_path)
            st.dataframe(df)

            # Viz
            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x="feature_group", y="test_cindex", hue="model")
            plt.title("Survival Model Performance Comparison")
            plt.ylabel("Test C-Index")
            st.pyplot(plt)
        else:
            st.error("Comparison metrics not found.")

    with tab2:
        comp_path = Path("reports/metrics/subtype_classification.csv")
        if comp_path.exists():
            df = pd.read_csv(comp_path)
            st.dataframe(df)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x="feature_group", y="weighted_f1", hue="model")
            plt.title("Subtype Classification Performance Comparison")
            plt.ylabel("Weighted F1 Score")
            st.pyplot(plt)
        else:
            st.error("Comparison metrics not found.")

# --- Page: Biomarkers ---
elif page == "Biomarkers":
    st.title("🔬 Candidate Biomarker Discovery")

    group = st.selectbox("Select Feature Group", ["clinical", "genomic", "integrated"])

    # Load top features
    feat_path = Path(f"reports/tables/selected_{group}_features.csv")
    if feat_path.exists():
        df = pd.read_csv(feat_path)
        st.write(f"Top Candidate Biomarkers for {group} model (by SHAP Importance):")
        st.table(df.head(15))

        # Load SHAP Plot
        shap_plot = Path(f"reports/figures/{group}_shap_summary.png")
        if shap_plot.exists():
            st.image(str(shap_plot), caption=f"SHAP Summary Plot - {group}", use_column_width=True)
    else:
        st.error("Biomarker analysis not found. Please run the explainability stage.")
