import pandas as pd
import json
import logging
from pathlib import Path
from src.breast_cancer_ml.config.settings import settings, logger

def detect_leakage():
    """
    Detects potential data leakage variables.
    Identifies columns that are derived from the target or occur post-outcome.
    """
    raw_path = Path(settings.get("data.raw_path"))
    df = pd.read_csv(raw_path)
    cols = df.columns.tolist()

    survival_time = settings.get("targets.survival_time")
    survival_event = settings.get("targets.survival_event")
    subtype = settings.get("targets.subtype")

    leakage_candidates = []

    # 1. Survival-related leakage
    survival_keywords = ['survival', 'death', 'prognosis', 'outcome']
    for col in cols:
        if any(kw in col.lower() for kw in survival_keywords):
            if col not in [survival_time, survival_event]:
                leakage_candidates.append({
                    "column": col,
                    "reason": "Survival-derived or outcome-related variable",
                    "excluded_for": "survival_prediction"
                })

    # 2. Subtype-related leakage
    subtype_keywords = ['subtype', 'cluster', 'classifier', 'pam50']
    for col in cols:
        if any(kw in col.lower() for kw in subtype_keywords):
            if col != subtype:
                leakage_candidates.append({
                    "column": col,
                    "reason": "Molecular subtype proxy or cluster label",
                    "excluded_for": "subtype_classification"
                })

    # Save report
    report_dir = Path(settings.get("data.data_dictionary_path"))
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "leakage_report.json", "w") as f:
        json.dump(leakage_candidates, f, indent=4)

    logger.info(f"Leakage detection complete. Found {len(leakage_candidates)} potential leakage variables.")
    return leakage_candidates

if __name__ == "__main__":
    detect_leakage()
