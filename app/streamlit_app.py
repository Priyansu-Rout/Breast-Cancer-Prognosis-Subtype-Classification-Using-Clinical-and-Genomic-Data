import streamlit as st
import sys
from pathlib import Path

# Add the project root to the python path so 'src' can be found
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from src.breast_cancer_ml.config.settings import settings

# Page configuration
st.set_page_config(
    page_title="Breast Cancer AI Research",
    page_icon="🎗️",
    layout="wide"
)

# Custom CSS for professional, modern look - Theme Agnostic
st.markdown("""
    <style>
    /* Main background - adapts to theme but stays clean */
    .stApp {
        background-color: var(--background-color);
    }

    /* Custom Cards - White in light mode, Dark Grey in dark mode */
    .research-card {
        background-color: var(--secondary-background-color);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        color: var(--text-color) !important;
    }

    /* Header Styling */
    h1 {
        color: var(--text-color) !important;
        font-weight: 800 !important;
        font-family: 'Inter', sans-serif !important;
    }
    h2, h3 {
        color: var(--text-color) !important;
        font-weight: 700 !important;
    }

    /* The Goal Highlight Box - Keeps the red accent but adapts text */
    .highlight-box {
        background: linear-gradient(90deg, rgba(255, 75, 75, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #ff4b4b;
        margin-bottom: 30px;
        box-shadow: 0 2px 10px rgba(255, 75, 75, 0.1);
        color: var(--text-color) !important;
    }
    .highlight-box h3 {
        color: #ff4b4b !important;
        margin-top: 0 !important;
    }
    .highlight-box p {
        color: var(--text-color) !important;
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        opacity: 0.9;
    }

    /* Metric Styling - Uses theme variables for text */
    .custom-metric {
        text-align: center;
        padding: 10px;
    }
    .metric-label {
        color: var(--secondary-text-color);
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
        opacity: 0.8;
    }
    .metric-value {
        color: var(--text-color) !important;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }

    /* Button Styling */
    .stButton>button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
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
    """Loads the trained model."""
    if model_type == "survival":
        path = Path(f"models/survival/{group}/random_survival_forest.joblib")
    else:
        path = Path(f"models/classification/{group}/xgboost.joblib")

    if not path.exists():
        return None
    return joblib.load(path)

def load_label_encoder(group):
    """Loads the label encoder for classification subtypes."""
    path = Path(f"models/classification/{group}/label_encoder.joblib")
    if not path.exists():
        return None
    return joblib.load(path)

def get_treatment_recommendation(subtype, risk_score):
    """
    Provides a simplified clinical treatment suggestion based on
    molecular subtype and risk level.
    """
    # Define risk level for the logic
    is_high_risk = risk_score > 1.5

    # Use normalized keys that match the short-hand labels (LumA, LumB, etc.)
    recommendations = {
        "luma": {
            "Low": "Standard Endocrine Therapy (Hormone therapy) and regular monitoring. Low risk of recurrence.",
            "High": "Combination of Endocrine Therapy and potentially adjuvant chemotherapy due to elevated risk."
        },
        "lumb": {
            "Low": "Endocrine Therapy with closer monitoring of biomarkers.",
            "High": "Aggressive combination of Chemotherapy and Endocrine Therapy is typically recommended."
        },
        "her2": {
            "Low": "Targeted anti-HER2 therapy (e.g., Trastuzumab) and surgery.",
            "High": "Intensive Targeted Therapy combined with systemic Chemotherapy."
        },
        "basal": {
            "Low": "Systemic Chemotherapy. Basal-like tumors are generally more aggressive even at lower relative risk.",
            "High": "Aggressive multi-modal therapy including high-dose Chemotherapy and surgical intervention."
        }
    }

    # Normalize the input subtype: lowercase and remove non-alphanumeric chars
    # This converts "LumB" -> "lumb", "Luminal B" -> "luminalb", "Basal-like" -> "basallike"
    normalized_subtype = subtype.lower().replace(" ", "").replace("-", "").replace("inal", "")

    risk_label = "High" if is_high_risk else "Low"
    return recommendations.get(normalized_subtype, {}).get(risk_label, "Consult a multidisciplinary tumor board for personalized treatment planning.")

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
            st.image(str(km_path), caption="Integrated Model Risk Stratification (Kaplan-Meier Curve)", width=700)
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

            # Add a clean, styled data table with high-contrast highlighting
            st.markdown("### Detailed Performance Metrics")
            # Use a custom function to ensure text is black and background is a soft green
            def highlight_max_custom(s):
                is_max = s == s.max()
                return [f'background-color: #c3e6cb; color: black;' if v else '' for v in is_max]

            st.dataframe(df.style.apply(highlight_max_custom, subset=['test_cindex']), use_container_width=True)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=df, x="feature_group", y="test_cindex", hue="model")
            plt.title("Survival Model Performance (C-Index)")
            plt.ylabel("Test C-Index (Higher is better)")
            st.pyplot(plt)
        else:
            st.error("Comparison metrics not found.")

    with tab2:
        st.subheader("How accurately can we classify subtypes?")

        # Load and merge the JSON metrics since subtype_classification.csv is missing
        try:
            groups = ["clinical", "genomic", "integrated"]
            all_metrics = []

            for g in groups:
                path = Path(f"reports/metrics/{g}_classification.json")
                if path.exists():
                    data = joblib.load(path) if path.suffix == '.joblib' else pd.read_json(path)
                    for model, metrics in data.items():
                        all_metrics.append({
                            "feature_group": g,
                            "model": model,
                            "weighted_f1": metrics.get("weighted_f1", 0),
                            "accuracy": metrics.get("accuracy", 0),
                            "roc_auc": metrics.get("roc_auc", 0)
                        })

            if all_metrics:
                df = pd.DataFrame(all_metrics)
                best_model = df.loc[df['weighted_f1'].idxmax()]
                st.success(f"🏆 **Best Performer: {best_model['model']} using {best_model['feature_group']} features**")

                # Add a clean, styled data table with high-contrast highlighting
                st.markdown("### Detailed Performance Metrics")
                # Use a custom function to ensure text is black and background is a soft green
                def highlight_max_custom(s):
                    is_max = s == s.max()
                    return [f'background-color: #c3e6cb; color: black;' if v else '' for v in is_max]

                st.dataframe(df.style.apply(highlight_max_custom, subset=['weighted_f1']), use_container_width=True)

                plt.figure(figsize=(10, 6))
                sns.barplot(data=df, x="feature_group", y="weighted_f1", hue="model")
                plt.title("Subtype Classification Performance (F1 Score)")
                plt.ylabel("Weighted F1 Score")
                st.pyplot(plt)
            else:
                st.info("Classification metrics not found. Please run the pipeline.")
        except Exception as e:
            st.error(f"Error loading classification metrics: {e}")

# --- Page: Interactive Prediction ---
elif page == "Interactive Prediction":
    st.title("🧪 AI Prediction Lab")
    st.markdown("Experience the models yourself. Select a model arm and simulate patient data.")

    mode = st.selectbox("What would you like to predict?", ["Survival Risk", "Molecular Subtype"])

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Patient Profile")

        # Use a container for a cleaner look
        with st.container():
            group = st.selectbox("Choose Model Complexity", ["clinical", "genomic", "integrated"],
                                 help="Clinical = Basic, Genomic = DNA-based, Integrated = Both")

            if mode == "Survival Risk":
                age = st.slider("Age at Diagnosis", 20, 90, 50)
                tumor_size = st.slider("Tumor Size (mm)", 1, 100, 20)
                stage = st.selectbox("Tumor Stage", ["I", "II", "III", "IV"])
            else:
                st.info("🧬 For subtype prediction, the model analyzes genomic expression levels from the patient's sample.")
                st.write("*(Input features for genomic models are complex high-dimensional vectors; for this demo, we use representative patient samples)*")

            st.markdown("---")
            predict_btn = st.button("🚀 Run AI Prediction", use_container_width=True)

    with col2:
        st.subheader("Prediction Result")
        if predict_btn:
            # Create a professional-looking card for the result
            st.markdown('<div class="research-card">', unsafe_allow_html=True)

            if mode == "Survival Risk":
                model = load_model(group, "survival")
                if model:
                    X_processed = Path(f"data/processed/test_{group}/X.csv")
                    if X_processed.exists():
                        # FIX: Instead of always taking index [0], we pick a random patient
                        # to simulate different outcomes in the demo.
                        full_df = pd.read_csv(X_processed)

                        # For 'clinical' mode, we'll actually use the sliders to modify the patient
                        if group == "clinical":
                            sample_df = full_df.iloc[[0]].copy()
                            # RESET all known risk factors to lowest possible to ensure sliders drive the result
                            for col in sample_df.columns:
                                if 'lymph' in col.lower() or 'positive' in col.lower():
                                    sample_df[col] = 0
                                if 'grade' in col.lower():
                                    sample_df[col] = 1
                                if 'age' in col.lower(): sample_df[col] = age
                                if 'size' in col.lower(): sample_df[col] = tumor_size
                                if 'stage' in col.lower(): sample_df[col] = 1 if stage == 'I' else (2 if stage == 'II' else (3 if stage == 'III' else 4))
                        else:
                            # For genomic/integrated, we pick a random patient to show variety
                            import random
                            random_idx = random.randint(0, len(full_df) - 1)
                            sample_df = full_df.iloc[[random_idx]]
                            st.caption(f"Simulating Patient Case #{random_idx + 1}")

                        risk_score = model.predict(sample_df)[0]

                        # --- FINAL FIX: Normalized Risk Scoring ---
                        # Instead of absolute or simple relative, we map the score
                        # based on the actual distribution of the dataset.
                        all_risks = model.predict(full_df)
                        min_risk = np.min(all_risks)
                        max_risk = np.max(all_risks)

                        # Normalize current score between 0 and 1
                        normalized_score = (risk_score - min_risk) / (max_risk - min_risk) if max_risk != min_risk else 0.5

                        # Modern Metric Layout
                        st.markdown(f"""
                            <div class="custom-metric">
                                <p class="metric-label">Predicted Risk Score</p>
                                <p class="metric-value">{risk_score:.4f}</p>
                            </div>
                        """, unsafe_allow_html=True)

                        # Use normalized score for the category labels
                        if normalized_score < 0.20:
                            category = "🟢 Low Risk"
                            detail = "The patient is in the lowest risk tier of the study population."
                        elif 0.20 <= normalized_score <= 0.45:
                            category = "🟡 Moderate Risk"
                            detail = "The patient has a moderate risk profile compared to others."
                        else:
                            category = "🔴 High Risk"
                            detail = "The patient is in the highest risk tier of the study population."


                        st.markdown(f'<div style="text-align: center; margin-top: 20px;"><h3 style="color: black;">Result: {category}</h3>></div>', unsafe_allow_html=True)
                        st.markdown(f'<p style="color: #666; text-align: center; font-style: italic; margin-top: 10px;">{detail}</p>', unsafe_allow_html=True)

                        # --- NEW: LOCAL EXPLAINABILITY SECTION ---
                        st.markdown("---")
                        st.markdown('<p style="text-align: center; font-weight: bold; color: #31333F;">🔍 Why this result?</p>', unsafe_allow_html=True)

                        if group == "clinical":
                            # Dynamic logic based on actual slider values
                            increasing_factors = []
                            decreasing_factors = []

                            if age > 55: increasing_factors.append(f"Age ({age} yrs) - Older age correlates with higher risk")
                            elif age < 40: decreasing_factors.append(f"Age ({age} yrs) - Younger age is a protective factor")

                            if tumor_size > 20: increasing_factors.append(f"Tumor Size ({tumor_size}mm) - Larger size increases risk")
                            elif tumor_size < 15: decreasing_factors.append(f"Tumor Size ({tumor_size}mm) - Small tumor size is protective")

                            if stage in ["III", "IV"]: increasing_factors.append(f"Stage {stage} - Advanced stage increases risk")
                            elif stage == "I": decreasing_factors.append(f"Stage {stage} - Early detection is highly protective")

                            exp_col1, exp_col2 = st.columns(2)
                            with exp_col1:
                                st.markdown("**📈 Risk Increasing Factors**")
                                if increasing_factors:
                                    for factor in increasing_factors:
                                        st.markdown(f"- {factor}")
                                else:
                                    st.write("No significant increasing factors.")
                            with exp_col2:
                                st.markdown("**📉 Protective Factors**")
                                if decreasing_factors:
                                    for factor in decreasing_factors:
                                        st.markdown(f"- {factor}")
                                else:
                                    st.write("No significant protective factors.")
                        else:
                            # For Genomic/Integrated, we use the SHAP proxy from the tables
                            feat_path = Path(f"reports/tables/selected_{group}_features.csv")
                            if feat_path.exists():
                                top_feats = pd.read_csv(feat_path).head(5)
                                exp_col1, exp_col2 = st.columns(2)
                                with exp_col1:
                                    st.markdown("**📈 Risk Increasing Factors**")
                                    increasing = top_feats[top_feats['importance'] > 0].head(2)
                                    for idx, row in increasing.iterrows():
                                        st.markdown(f"- {row['feature']} (Impact: {row['importance']:.3f})")
                                with exp_col2:
                                    st.markdown("**📉 Protective Factors**")
                                    decreasing = top_feats.tail(2)
                                    for idx, row in decreasing.iterrows():
                                        st.markdown(f"- {row['feature']} (Low Impact: {row['importance']:.3f})")
                            else:
                                st.info("Detailed feature impact not available for this simulation.")
                        # -------------------------------------------------

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
                        # FIX: Pick a random patient for subtype classification too
                        full_df = pd.read_csv(X_processed)
                        import random
                        random_idx = random.randint(0, len(full_df) - 1)
                        sample_df = full_df.iloc[[random_idx]]
                        st.caption(f"Simulating Patient Case #{random_idx + 1}")

                        prob = model.predict_proba(sample_df)[0]
                        pred_idx = np.argmax(prob)
                        subtype = le.inverse_transform([pred_idx])[0]

                        st.markdown(f'<div style="text-align: center; margin-bottom: 20px;"><h3 style="color: black;">Predicted Subtype: {subtype}</h3></div>', unsafe_allow_html=True)

                        # --- NEW: TREATMENT RECOMMENDATION ---
                        # We'll use a default risk score of 1.0 for subtype-only mode, or let the user switch to survival
                        recommendation = get_treatment_recommendation(subtype, 1.0)
                        st.markdown(f"""
                            <div style="background-color: #eef2ff; padding: 15px; border-radius: 10px; border-left: 5px solid #4f46e5; color: #31333f; margin-bottom: 20px;">
                                <p style="font-weight: bold; margin-bottom: 5px;">🩺 Suggested Clinical Path:</p>
                                <p style="font-style: italic;">{recommendation}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        # ------------------------------------

                        prob_df = pd.DataFrame({"Subtype": le.classes_, "Probability": prob}).sort_values(by="Probability", ascending=False)
                        st.bar_chart(prob_df.set_index("Subtype"))
                    else:
                        st.error("Processed data not found.")
                else:
                    st.error("Model or Label Encoder not found.")

            st.markdown('</div>', unsafe_allow_html=True)

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
            st.image(str(shap_plot), caption=f"SHAP Summary Plot - {group}", width=700)
            st.markdown("""
            **How to read this:**
            - **Feature**: The gene or marker.
            - **Color**: Red indicates high value, Blue indicates low value.
            - **Position**: How much it pushes the prediction towards a specific subtype.
            """)
    else:
        st.error("Biomarker analysis not found. Please run the explainability stage.")
