import sys
from pathlib import Path

import streamlit as st

# Add the project root to the python path so 'src' can be found
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

@st.cache_resource
def load_model(group, model_type="survival"):
    """Loads the trained model pipeline."""
    if model_type == "survival":
        path = Path(f"models/survival/{group}/random_survival_forest.joblib")
    else:
        path = Path(f"models/classification/{group}/xgboost.joblib")

    if not path.exists():
        return None
    return joblib.load(path)

@st.cache_resource
def load_label_encoder(group):
    """Loads the label encoder for classification subtypes."""
    path = Path(f"models/classification/{group}/label_encoder.joblib")
    if not path.exists():
        return None
    return joblib.load(path)

@st.cache_data
def get_training_reference_risks(group):
    """Computes and caches reference risk distribution on the training cohort."""
    model = load_model(group, "survival")
    if model is None:
        return None
    train_path = Path(f"data/processed/train_{group}/X.csv")
    if not train_path.exists():
        return None
    train_df = pd.read_csv(train_path)
    return model.predict(train_df)

@st.cache_data
def load_test_cohort(group):
    """Loads test feature matrix and associated ground-truth outcomes."""
    x_path = Path(f"data/processed/test_{group}/X.csv")
    y_surv_path = Path(f"data/processed/test_{group}/y_survival.csv")
    y_sub_path = Path(f"data/processed/test_{group}/y_subtype.csv")
    if not (x_path.exists() and y_surv_path.exists() and y_sub_path.exists()):
        return None, None, None
    X = pd.read_csv(x_path)
    y_surv = pd.read_csv(y_surv_path)
    y_sub = pd.read_csv(y_sub_path)
    return X, y_surv, y_sub

@st.cache_data
def get_training_feature_medians(group="clinical"):
    """Returns medians of training features for reference benchmarking."""
    train_path = Path(f"data/processed/train_{group}/X.csv")
    if not train_path.exists():
        return {}
    df = pd.read_csv(train_path)
    return df.median().to_dict()

def get_biological_subtype_info(subtype):
    """
    Returns objective biological characteristics for breast cancer molecular subtypes.
    Strictly educational and research-oriented — no treatment recommendations.
    """
    subtype_clean = str(subtype).strip()
    lookup = {
        "luma": "LumA",
        "luminal a": "LumA",
        "lumb": "LumB",
        "luminal b": "LumB",
        "her2": "Her2",
        "basal": "Basal",
        "claudin-low": "claudin-low",
        "normal": "Normal",
        "nc": "NC",
    }
    key = lookup.get(subtype_clean.lower(), subtype_clean)

    info_dict = {
        "LumA": {
            "name": "Luminal A",
            "biology": "Estrogen receptor-positive (ER+) and progesterone receptor-positive (PR+), HER2-negative with low Ki-67 proliferation index.",
            "prognosis": "Generally exhibits favorable long-term disease-specific survival in historical cohorts.",
            "biomarkers": "High ESR1, PGR, GATA3; low MKI67 proliferation signature."
        },
        "LumB": {
            "name": "Luminal B",
            "biology": "Estrogen receptor-positive (ER+), with higher histological grade or proliferation markers, and variable HER2 status.",
            "prognosis": "Higher risk of distant relapse compared to Luminal A, driven by active cell-cycle proliferation.",
            "biomarkers": "ESR1+, elevated AURKA and MKI67 expression; variable ERBB2 amplification."
        },
        "Her2": {
            "name": "HER2-Enriched",
            "biology": "Overexpression and amplification of the ERBB2 (HER2) oncogene and adjacent 17q amplicon genes.",
            "prognosis": "Aggressive baseline natural history with high proliferative activity.",
            "biomarkers": "Elevated ERBB2, GRB7; typically ER-negative and PR-negative."
        },
        "Basal": {
            "name": "Basal-Like",
            "biology": "Frequently exhibits the triple-negative clinical phenotype (ER-, PR-, HER2-), prominent genomic instability, and high TP53 mutation rate.",
            "prognosis": "Associated with earlier relapse kinetics and high mitotic indices in benchmark studies.",
            "biomarkers": "High basal cytokeratins (KRT5, KRT14), EGFR overexpression, frequent TP53 alterations."
        },
        "claudin-low": {
            "name": "Claudin-Low",
            "biology": "Characterized by reduced expression of cell-cell adhesion and tight junction genes (CLDN3, CLDN4, CLDN7, CDH1) with mesenchymal and stem-like characteristics.",
            "prognosis": "Intermediate-to-poor prognosis accompanied by pronounced immune lymphocytic infiltration.",
            "biomarkers": "Downregulated claudins and E-cadherin; enriched for EMT and immune signatures."
        },
        "Normal": {
            "name": "Normal-Like",
            "biology": "Expression profile mirroring non-malignant breast parenchyma and adipose tissue.",
            "prognosis": "Intermediate trajectory; frequently reflects tumor samples with substantial non-neoplastic stromal content.",
            "biomarkers": "Enriched for adipose and non-epithelial normal mammary tissue signatures."
        },
        "NC": {
            "name": "Unclassified (NC)",
            "biology": "Samples that did not attain definitive confidence thresholds in PAM50 centroid assignment.",
            "prognosis": "Heterogeneous molecular profile across cluster boundaries.",
            "biomarkers": "Mixed expression characteristics."
        }
    }
    return info_dict.get(key, {
        "name": subtype_clean,
        "biology": "Molecular subtype categorized by PAM50 gene expression profiling.",
        "prognosis": "Refer to METABRIC cohort survival benchmarks.",
        "biomarkers": "Multi-gene PAM50 expression panel."
    })

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
                return ['background-color: #c3e6cb; color: black;' if v else '' for v in is_max]

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
                    return ['background-color: #c3e6cb; color: black;' if v else '' for v in is_max]

                st.dataframe(df.style.apply(highlight_max_custom, subset=['weighted_f1']), use_container_width=True)

                plt.figure(figsize=(10, 6))
                sns.barplot(data=df, x="feature_group", y="weighted_f1", hue="model")
                plt.title("Subtype Classification Performance (F1 Score)")
                plt.ylabel("Weighted F1 Score")
                st.pyplot(plt)
            else:
                st.info("Classification metrics not found. Please run the pipeline.")
        except (FileNotFoundError, KeyError, ValueError, OSError) as e:
            st.error(f"Error loading classification metrics: {e}")

# --- Page: Interactive Prediction ---
elif page == "Interactive Prediction":
    st.title("🧪 AI Prediction Lab")
    st.markdown(
        "Explore model predictions across **Clinical**, **Genomic**, and **Integrated** feature arms. "
        "Benchmark real patient cases from the test cohort against observed outcomes, or simulate clinical parameters."
    )

    st.markdown("""
        <div style="background-color: rgba(255, 75, 75, 0.08); border-left: 4px solid #ff4b4b; padding: 12px 16px; border-radius: 8px; margin-bottom: 20px;">
            <p style="margin: 0; font-size: 0.95rem;">
                <b>🔬 Research Benchmark Tool:</b> Risk scores represent relative hazard rankings from survival models; subtype predictions represent multi-class PAM50 probabilities. This tool does not provide clinical diagnoses or treatment regimens.
            </p>
        </div>
    """, unsafe_allow_html=True)

    from scipy.stats import percentileofscore

    task_col1, task_col2 = st.columns(2)
    with task_col1:
        mode = st.radio(
            "Select Prediction Task",
            ["Prognostic Survival Risk", "Molecular Subtype Classification"],
            horizontal=True
        )
    with task_col2:
        group = st.selectbox(
            "Select Feature Complexity Arm",
            ["clinical", "genomic", "integrated"],
            format_func=lambda x: {
                "clinical": "Clinical Arm (Patient demographics & histopathology)",
                "genomic": "Genomic Arm (mRNA expression & somatic mutations)",
                "integrated": "Integrated Arm (Clinical + Genomic + Mutations)"
            }[x]
        )

    # Input method selection
    input_method = st.radio(
        "Patient Profile Source",
        ["Benchmark Test Cohort Case (All Arms)", "Custom Clinical Parameter Simulation (Clinical Arm Only)"],
        horizontal=True
    )

    col1, col2 = st.columns([1, 1], gap="large")

    sample_df = None
    patient_info = {}
    is_simulation = False

    with col1:
        st.subheader("Patient Input Profile")

        if input_method == "Custom Clinical Parameter Simulation (Clinical Arm Only)":
            if group != "clinical":
                st.warning(
                    f"⚠️ The **{group.capitalize()}** model arm requires 490+ high-dimensional gene expression vectors. "
                    "Custom parameter simulation is available for the **Clinical** arm. "
                    "Please switch to the **Clinical** arm above, or choose 'Benchmark Test Cohort Case' to test this arm."
                )
            else:
                is_simulation = True
                st.markdown("*Adjust clinical parameters to evaluate risk:*")
                sim_age = st.slider("Age at Diagnosis (years)", min_value=20, max_value=95, value=60)
                sim_size = st.slider("Tumor Size (mm)", min_value=1, max_value=150, value=25)
                sim_grade = st.selectbox(
                    "Neoplasm Histologic Grade",
                    [1, 2, 3],
                    format_func=lambda g: f"Grade {g} ({'Well' if g==1 else ('Moderately' if g==2 else 'Poorly')} differentiated)",
                    index=1
                )
                sim_nodes = st.slider("Positive Lymph Nodes Examined", min_value=0, max_value=35, value=1)
                sim_stage = st.selectbox(
                    "Tumor Stage",
                    [0, 1, 2, 3, 4],
                    format_func=lambda s: f"Stage {s}",
                    index=2
                )

                c_opt1, c_opt2, c_opt3 = st.columns(3)
                with c_opt1:
                    sim_chemo = st.selectbox("Chemotherapy", ["No", "Yes"], index=0)
                with c_opt2:
                    sim_hormone = st.selectbox("Hormone Therapy", ["No", "Yes"], index=1)
                with c_opt3:
                    sim_radio = st.selectbox("Radiotherapy", ["No", "Yes"], index=1)

                train_medians = get_training_feature_medians("clinical")
                magea8_val = train_medians.get("magea8", -0.1313)

                sample_df = pd.DataFrame([{
                    "age_at_diagnosis": float(sim_age),
                    "chemotherapy": 1.0 if sim_chemo == "Yes" else 0.0,
                    "neoplasm_histologic_grade": float(sim_grade),
                    "hormone_therapy": 1.0 if sim_hormone == "Yes" else 0.0,
                    "lymph_nodes_examined_positive": float(sim_nodes),
                    "radio_therapy": 1.0 if sim_radio == "Yes" else 0.0,
                    "tumor_size": float(sim_size),
                    "tumor_stage": float(sim_stage),
                    "magea8": float(magea8_val)
                }])

                patient_info = {
                    "Age": f"{sim_age} years",
                    "Tumor Size": f"{sim_size} mm",
                    "Grade": f"Grade {sim_grade}",
                    "Positive Nodes": sim_nodes,
                    "Stage": f"Stage {sim_stage}",
                    "Chemotherapy": sim_chemo,
                    "Hormone Therapy": sim_hormone,
                    "Radiotherapy": sim_radio
                }
        else:
            # Benchmark Test Case Selector
            X_test, y_surv_test, y_sub_test = load_test_cohort(group)
            if X_test is not None:
                max_cases = len(X_test)
                case_num = st.number_input(
                    f"Select Patient Case (1 to {max_cases})",
                    min_value=1,
                    max_value=max_cases,
                    value=1,
                    step=1
                )
                case_idx = case_num - 1
                sample_df = X_test.iloc[[case_idx]].copy()

                # Also load clinical test data for displaying patient summary
                X_clin_test, _, _ = load_test_cohort("clinical")
                if X_clin_test is not None:
                    clin_row = X_clin_test.iloc[case_idx]
                    patient_info = {
                        "Age": f"{clin_row['age_at_diagnosis']:.1f} years",
                        "Tumor Size": f"{clin_row['tumor_size']:.1f} mm",
                        "Histologic Grade": f"Grade {int(clin_row['neoplasm_histologic_grade'])}",
                        "Positive Lymph Nodes": int(clin_row['lymph_nodes_examined_positive']),
                        "Tumor Stage": f"Stage {int(clin_row['tumor_stage'])}",
                        "Chemotherapy": "Yes" if clin_row['chemotherapy'] == 1 else "No",
                        "Hormone Therapy": "Yes" if clin_row['hormone_therapy'] == 1 else "No",
                        "Radiotherapy": "Yes" if clin_row['radio_therapy'] == 1 else "No"
                    }

                st.markdown(f"**Patient Case #{case_num} Clinical Profile:**")
                summary_df = pd.DataFrame(list(patient_info.items()), columns=["Clinical Characteristic", "Patient Value"])
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                if y_surv_test is not None and y_sub_test is not None:
                    months = y_surv_test.iloc[case_idx]["overall_survival_months"]
                    event_code = y_surv_test.iloc[case_idx]["overall_survival"]
                    status_str = "Living (Censored)" if event_code == 1 else "Deceased (Event Observed)"
                    true_sub = y_sub_test.iloc[case_idx]["pam50_+_claudin-low_subtype"]

                    st.markdown("**Recorded Clinical Outcome (Ground Truth):**")
                    gt_col1, gt_col2 = st.columns(2)
                    with gt_col1:
                        st.metric("Follow-up Duration", f"{months:.1f} mos", status_str)
                    with gt_col2:
                        st.metric("Actual Subtype", true_sub)
            else:
                st.error("Test dataset files not found.")

    with col2:
        st.subheader("Model Evaluation & Output")
        if sample_df is not None:
            if mode == "Prognostic Survival Risk":
                surv_model = load_model(group, "survival")
                train_ref_risks = get_training_reference_risks(group)

                if surv_model is not None and train_ref_risks is not None:
                    risk_score = float(surv_model.predict(sample_df)[0])
                    pct = float(percentileofscore(train_ref_risks, risk_score))

                    if pct < 33.33:
                        risk_tier = "🟢 Low Risk Tier"
                        badge_color = "#28a745"
                    elif pct <= 66.67:
                        risk_tier = "🟡 Moderate Risk Tier"
                        badge_color = "#ffc107"
                    else:
                        risk_tier = "🔴 High Risk Tier"
                        badge_color = "#dc3545"

                    st.markdown(f"""
                        <div class="research-card" style="border-top: 5px solid {badge_color};">
                            <div class="custom-metric">
                                <p class="metric-label">{group.capitalize()} Model Predicted Risk Score</p>
                                <p class="metric-value">{risk_score:.4f}</p>
                            </div>
                            <div style="text-align: center; margin-top: 10px;">
                                <h3 style="margin: 0; color: inherit;">{risk_tier}</h3>
                                <p style="font-size: 1.05rem; margin-top: 6px; opacity: 0.85;"><b>{pct:.1f}th Percentile</b> within METABRIC Reference Cohort</p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.progress(min(max(pct / 100.0, 0.0), 1.0))
                    st.caption(f"Risk Percentile: {pct:.1f}% of reference cohort patients had a lower predicted mortality hazard.")

                    st.info(
                        "📌 **Prognostic Score Interpretation:**\n"
                        "The risk score reflects the relative hazard ranking produced by the Random Survival Forest. "
                        "It is a **relative risk score**, not an absolute calibrated probability of death. "
                        "Higher scores denote shorter expected survival relative to the study population."
                    )

                    # Local comparison vs training median
                    if group == "clinical" and is_simulation:
                        st.markdown("#### Clinical Feature Deviation vs Cohort Median")
                        clin_medians = get_training_feature_medians("clinical")
                        dev_data = []
                        if clin_medians:
                            for feat, label in [
                                ("age_at_diagnosis", "Age (years)"),
                                ("tumor_size", "Tumor Size (mm)"),
                                ("lymph_nodes_examined_positive", "Positive Lymph Nodes"),
                                ("neoplasm_histologic_grade", "Histologic Grade"),
                                ("tumor_stage", "Tumor Stage")
                            ]:
                                val = sample_df.iloc[0][feat]
                                med = clin_medians.get(feat, np.nan)
                                diff = val - med
                                diff_str = f"+{diff:.1f}" if diff > 0 else (f"{diff:.1f}" if diff < 0 else "0.0")
                                dev_data.append({
                                    "Feature": label,
                                    "Patient Value": f"{val:.1f}" if isinstance(val, float) else str(val),
                                    "Cohort Median": f"{med:.1f}" if isinstance(med, float) else str(med),
                                    "Difference": diff_str
                                })
                            st.dataframe(pd.DataFrame(dev_data), use_container_width=True, hide_index=True)

                else:
                    st.error("Survival model or training reference could not be loaded.")

            else:
                # Molecular Subtype Classification
                clf_model = load_model(group, "classification")
                le = load_label_encoder(group)

                if clf_model is not None and le is not None:
                    probs = clf_model.predict_proba(sample_df)[0]
                    pred_idx = int(np.argmax(probs))
                    pred_subtype = le.inverse_transform([pred_idx])[0]
                    pred_conf = float(probs[pred_idx]) * 100.0

                    info = get_biological_subtype_info(pred_subtype)

                    st.markdown(f"""
                        <div class="research-card" style="border-top: 5px solid #4f46e5;">
                            <div class="custom-metric">
                                <p class="metric-label">Predicted Molecular Subtype</p>
                                <p class="metric-value">{pred_subtype}</p>
                            </div>
                            <div style="text-align: center; margin-top: 5px;">
                                <p style="font-size: 1.1rem; margin: 0; opacity: 0.85;"><b>Confidence: {pred_conf:.1f}%</b></p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown("#### Probability Distribution Across Subtypes")
                    prob_df = pd.DataFrame({
                        "Subtype": le.classes_,
                        "Probability": probs
                    }).sort_values(by="Probability", ascending=True)

                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.barh(prob_df["Subtype"], prob_df["Probability"], color="#4f46e5")
                    ax.set_xlim(0, 1.05)
                    ax.set_xlabel("Predicted Probability")
                    ax.grid(axis='x', alpha=0.3)
                    for bar in bars:
                        w = bar.get_width()
                        if w > 0.02:
                            ax.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w:.1%}", va='center', fontsize=9)
                    st.pyplot(fig)
                    plt.close(fig)

                    st.markdown(f"#### Biological Profile: **{info['name']}**")
                    st.markdown(f"- **Key Characteristics:** {info['biology']}")
                    st.markdown(f"- **Prognostic Context:** {info['prognosis']}")
                    st.markdown(f"- **Biomarker Profile:** `{info['biomarkers']}`")

                else:
                    st.error("Classification model or label encoder not found.")

            # Biomarker context expander
            biomarker_path = Path(f"reports/tables/selected_{group}_features.csv")
            if biomarker_path.exists():
                with st.expander(f"📚 View Top Global Predictive Features ({group.capitalize()} Arm)"):
                    top_f = pd.read_csv(biomarker_path).head(10)
                    st.dataframe(top_f, use_container_width=True, hide_index=True)
                    st.caption("Feature importances derived from SHAP global analysis on the METABRIC cohort.")

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
