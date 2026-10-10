import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from app.streamlit_app import (
    get_biological_subtype_info,
    get_training_feature_medians,
    get_training_reference_risks,
    load_label_encoder,
    load_model,
    load_test_cohort,
)


def test_models_and_encoders_load():
    for group in ["clinical", "genomic", "integrated"]:
        surv_model = load_model(group, "survival")
        assert surv_model is not None, f"Survival model missing for {group}"

        clf_model = load_model(group, "classification")
        assert clf_model is not None, f"Classification model missing for {group}"

        le = load_label_encoder(group)
        assert le is not None, f"Label encoder missing for {group}"
        assert len(le.classes_) > 0

def test_training_reference_risks():
    for group in ["clinical", "genomic", "integrated"]:
        ref_risks = get_training_reference_risks(group)
        assert ref_risks is not None, f"Reference risks missing for {group}"
        assert len(ref_risks) > 0
        assert not np.isnan(ref_risks).any()

def test_test_cohort_loading():
    for group in ["clinical", "genomic", "integrated"]:
        X, y_surv, y_sub = load_test_cohort(group)
        assert X is not None
        assert y_surv is not None
        assert y_sub is not None
        assert len(X) == len(y_surv) == len(y_sub)
        assert "overall_survival_months" in y_surv.columns
        assert "overall_survival" in y_surv.columns
        assert "pam50_+_claudin-low_subtype" in y_sub.columns

def test_clinical_simulation_prediction():
    medians = get_training_feature_medians("clinical")
    sample_sim = pd.DataFrame([{
        "age_at_diagnosis": 55.0,
        "chemotherapy": 0.0,
        "neoplasm_histologic_grade": 2.0,
        "hormone_therapy": 1.0,
        "lymph_nodes_examined_positive": 2.0,
        "radio_therapy": 1.0,
        "tumor_size": 22.0,
        "tumor_stage": 2.0,
        "magea8": float(medians.get("magea8", -0.1313))
    }])

    surv_model = load_model("clinical", "survival")
    ref_risks = get_training_reference_risks("clinical")

    risk_score = float(surv_model.predict(sample_sim)[0])
    pct = float(percentileofscore(ref_risks, risk_score))

    assert 0.0 <= pct <= 100.0
    assert not np.isnan(risk_score)

def test_no_substring_corruption_in_clinical_features():
    """Regression test: age slider must not corrupt magea8 feature."""
    medians = get_training_feature_medians("clinical")
    sim_age = 75.0
    sample_df = pd.DataFrame([{
        "age_at_diagnosis": float(sim_age),
        "chemotherapy": 0.0,
        "neoplasm_histologic_grade": 3.0,
        "hormone_therapy": 1.0,
        "lymph_nodes_examined_positive": 5.0,
        "radio_therapy": 1.0,
        "tumor_size": 40.0,
        "tumor_stage": 3.0,
        "magea8": float(medians.get("magea8", -0.1313))
    }])

    # magea8 must stay at its biological level, not patient age
    assert sample_df.loc[0, "magea8"] < 20.0
    assert sample_df.loc[0, "age_at_diagnosis"] == 75.0

def test_subtype_biological_info():
    info = get_biological_subtype_info("LumA")
    assert info["name"] == "Luminal A"
    assert "ER+" in info["biology"] or "Estrogen receptor" in info["biology"]

    info_her2 = get_biological_subtype_info("Her2")
    assert info_her2["name"] == "HER2-Enriched"
