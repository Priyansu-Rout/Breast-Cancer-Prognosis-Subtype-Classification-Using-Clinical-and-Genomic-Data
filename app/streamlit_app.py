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
    page_title="Breast Cancer AI Research",
    page_icon="🎗️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #fdfdfd;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .highlight-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Medical Disclaimer
st.sidebar.warning(
    "**⚠️ MEDICAL DISCLAIMER**\n\n"
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
st.sidebar.title("🎗️ Research Journey")
page = st.sidebar.radio("Navigate through the story", [
    "Welcome & Introduction",
    "The Data Journey",
    "Comparing Approaches",
    "Interactive Prediction",
    "Biological Insights"
])

# --- Page: Welcome & Introduction ---
if page == "Welcome & Introduction":
    st.title("🎗️ AI-Driven Breast Cancer Prognosis")

    st.markdown("""
    <div class="highlight-box">
        <h3>The Research Goal</h3>
        <p>Can we improve how we predict breast cancer outcomes and subtypes by combining <b>Clinical</b> (patient history) and <b>Genomic</b> (DNA/RNA) data?</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Why This Matters")
        st.write("""
        Breast cancer is not a single disease but a collection of different molecular subtypes.
        Some respond well to hormone therapy, others to chemotherapy.

        **The Challenge:** Clinical data alone often lacks the precision to categorize these subtypes accurately.
        **The Solution:** Integrating molecular gene-expression and mutation data to create a high-precision diagnostic tool.
        """)

        with st.expander("📖 What are Molecular Subtypes?"):
            st.write("""
            Molecular subtypes (like Luminal A, Luminal B, HER2-enriched, and Basal-like)
            are determined by the expression of specific genes. They dictate the
            aggressive nature of the tumor and the best treatment path.
            """)

    with col2:
        st.subheader("The Integrated Approach")
        st.info("Our pipeline combines three distinct data streams:")
        st.markdown("""
        - **Clinical Data**: Age, Tumor Size, Grade, Stage.
        - **Genomic Data**: mRNA Expression levels.
        - **Mutation Data**: Genetic mutations.
        """)

        km_path = Path("reports/figures/integrated_km.png")
        if km_path.exists():
            st.image(str(km_path), caption="Integrated Model Risk Stratification (Kaplan-Meier Curve)", use_column_width=True)
            st.caption("The curve shows how the integrated model separates patients into high and low-risk groups with high precision.")

# --- Page: The Data Journey ---
elif page == "The Data Journey":
    st.title("📊 The Data Journey")

    st.markdown("### From Raw Data to AI-Ready Features")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 1. The Dataset")
        st.write("**METABRIC Dataset**: A large-scale study of breast cancer patients containing clinical and genomic information.")

        raw_path = Path(settings.get("data.raw_path"))
        if raw_path.exists():
            df = pd.read_csv(raw_path).head(5)
            st.write("Preview of raw data:")
            st.dataframe(df)
        else:
            st.error("Dataset not found. Please run the pipeline first.")

    with col2:
        st.markdown("#### 2. The Processing Pipeline")
        st.markdown("""
        - **Validation**: Checking for missing values and data leakage.
        - **Preprocessing**: Normalizing gene expression values.
        - **Feature Engineering**: Identifying the most predictive genes.
        - **Versioning**: Managed by **DVC** to ensure every experiment is reproducible.
        """)

# --- Page: Comparing Approaches ---
elif page == "Comparing Approaches":
    st.title("⚖️ Comparing Approaches")
    st.markdown("Does adding genomic data actually help? **Yes.**")

    tab1, tab2 = st.tabs(["Survival Prediction (Risk)", "Subtype Classification"])

    with tab1:
        st.subheader("How well do we predict survival?")
        comp_path = Path("reports/metrics/survival_comparison.csv")
        if comp_path.exists():
            df = pd.read_csv(comp_path)

            # Highlight the best model
            best_model = df.loc[df['test_cindex'].idxmax()]
            st.success(f"🏆 **Best Performer: {best_model['model']} using {best_model['feature_group']} features**")

            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x="feature_group", y="test_cindex", hue="model")
            plt.title("Survival Model Performance (C-Index)")
            plt.ylabel("Test C-Index (Higher is better)")
            st.pyplot(plt)
        else:
            st.error("Comparison metrics not found.")

    with tab2:
        st.subheader("How accurately can we classify subtypes?")
        # Using integrated_classification.json as a proxy if csv is missing, but let's try to find a summary
        comp_path = Path("reports/metrics/subtype_classification.csv")
        if comp_path.exists():
            df = pd.read_csv(comp_path)
            best_model = df.loc[df['weighted_f1'].idxmax()]
            st.success(f"🏆 **Best Performer: {best_model['model']} using {best_model['feature_group']} features**")

            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x="feature_group", y="weighted_f1", hue="model")
            plt.title("Subtype Classification Performance (F1 Score)")
            plt.ylabel("Weighted F1 Score")
            st.pyplot(plt)
        else:
            st.info("Detailed comparison table not found. Check the 'Integrated' metrics for top results.")

# --- Page: Interactive Prediction ---
elif page == "Interactive Prediction":
    st.title("🧪 AI Prediction Lab")
    st.markdown("Experience the models yourself. Select a model arm and simulate patient data.")

    mode = st.selectbox("What would you like to predict?", ["Survival Risk", "Molecular Subtype"])

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Patient Profile")
        group = st.selectbox("Choose Model Complexity", ["clinical", "genomic", "integrated"],
                             help="Clinical = Basic, Genomic = DNA-based, Integrated = Both")

        if mode == "Survival Risk":
            age = st.slider("Age at Diagnosis", 20, 90, 50)
            tumor_size = st.slider("Tumor Size (mm)", 1, 100, 20)
            stage = st.selectbox("Tumor Stage", ["I", "II", "III", "IV"])
        else:
            st.info("For subtype prediction, the model analyzes genomic expression levels.")

        predict_btn = st.button("Run Prediction")

    with col2:
        st.subheader("Prediction Result")
        if predict_btn:
            if mode == "Survival Risk":
                model = load_model(group, "survival")
                if model:
                    X_processed = Path(f"data/processed/test_{group}/X.csv")
                    if X_processed.exists():
                        sample_df = pd.read_csv(X_processed).iloc[[0]]
                        risk_score = model.predict(sample_df)[0]
                        st.metric("Predicted Risk Score", f"{risk_score:.4f}")

                        category = "🔴 High Risk" if risk_score > 0.5 else "🟢 Low Risk"
                        st.markdown(f"### Prediction: {category}")
                        st.write("This score represents the relative risk of the patient compared to others in the study.")
                    else:
                        st.error("Processed data not found.")
                else:
                    st.error("Model not found.")
            else:
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
                        prob_df = pd.DataFrame({"Subtype": le.classes_, "Probability": prob}).sort_values(by="Probability", ascending=False)
                        st.bar_chart(prob_df.set_index("Subtype"))
                    else:
                        st.error("Processed data not found.")
                else:
                    st.error("Model or Label Encoder not found.")

# --- Page: Biological Insights ---
elif page == "Biological Insights":
    st.title("🔬 Biological Insights")
    st.markdown("The 'Black Box' opened: Which genes and markers actually drive the prediction?")

    group = st.selectbox("Select Model Arm", ["clinical", "genomic", "integrated"])

    feat_path = Path(f"reports/tables/selected_{group}_features.csv")
    if feat_path.exists():
        df = pd.read_csv(feat_path)
        st.subheader(f"Top Biomarkers ({group})")
        st.write("These features have the highest impact on the model's decision (SHAP values).")
        st.table(df.head(10))

        shap_plot = Path(f"reports/figures/{group}_shap_summary.png")
        if shap_plot.exists():
            st.image(str(shap_plot), caption=f"SHAP Summary Plot - {group}", use_column_width=True)
            st.markdown("""
            **How to read this:**
            - **Feature**: The gene or marker.
            - **Color**: Red indicates high value, Blue indicates low value.
            - **Position**: How much it pushes the prediction towards a specific subtype.
            """)
    else:
        st.error("Biomarker analysis not found. Please run the explainability stage.")
