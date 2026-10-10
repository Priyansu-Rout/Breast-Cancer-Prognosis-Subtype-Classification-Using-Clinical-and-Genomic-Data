import pandas as pd
import json
import logging
from pathlib import Path
from src.breast_cancer_ml.config.settings import settings, logger

def discover_feature_groups():
    """
    Programmatically discovers Clinical, Genomic, and Mutation feature groups.
    """
    raw_path = Path(settings.get("data.raw_path"))
    df = pd.read_csv(raw_path)
    cols = df.columns.tolist()

    # 1. Clinical Feature Discovery
    # Based on common medical terms and dataset inspection
    clinical_keywords = [
        'age', 'surgery', 'cancer_type', 'cellularity', 'chemotherapy',
        'er_status', 'her2_status', 'grade', 'menopausal', 'hormone',
        'laterality', 'lymph_nodes', 'npi', 'tumor_size', 'tumor_stage', 'radio_therapy'
    ]
    clinical_features = [col for col in cols if any(kw in col.lower() for kw in clinical_keywords)]
    # Remove targets from clinical
    targets = settings.get("targets").values()
    clinical_features = [f for f in clinical_features if f not in targets]

    # 2. Mutation Feature Discovery
    # Based on the _mut suffix seen in dataset inspection
    mutation_features = [col for col in cols if col.endswith('_mut')]

    # 3. Gene Expression Discovery
    # Everything else that isn't clinical, mutation, target, or ID
    id_cols = ['patient_id']
    excluded = set(clinical_features + mutation_features + list(targets) + id_cols)
    expression_features = [col for col in cols if col not in excluded]

    # Save reports
    report_dir = Path(settings.get("data.data_dictionary_path"))
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "clinical_features.json", "w") as f:
        json.dump(clinical_features, f, indent=4)
    with open(report_dir / "mutation_features.json", "w") as f:
        json.dump(mutation_features, f, indent=4)
    with open(report_dir / "expression_features.json", "w") as f:
        json.dump(expression_features, f, indent=4)

    logger.info(f"Feature discovery complete: {len(clinical_features)} Clinical, "
                f"{len(expression_features)} Genomic, {len(mutation_features)} Mutation.")

    return {
        "clinical": clinical_features,
        "genomic": expression_features,
        "mutation": mutation_features
    }

if __name__ == "__main__":
    discover_feature_groups()
