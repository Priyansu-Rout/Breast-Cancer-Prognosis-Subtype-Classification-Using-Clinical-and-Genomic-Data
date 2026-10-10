import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.breast_cancer_ml.config.settings import settings, logger
from src.breast_cancer_ml.features.discover import discover_feature_groups

def build_features():
    """
    Orchestrates feature construction for Clinical, Genomic, and Integrated models.
    Generates separate feature matrices and targets.
    """
    processed_dir = Path(settings.get("data.processed_path"))
    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    # 1. Discover Groups
    groups = discover_feature_groups()
    clinical_cols = groups["clinical"]
    genomic_cols = groups["genomic"]
    mutation_cols = groups["mutation"]

    # 2. Define Targets
    targets = settings.get("targets")
    time_col = targets["survival_time"]
    event_col = targets["survival_event"]
    subtype_col = targets["subtype"]

    # Helper to save matrices
    def save_set(df, name, cols, is_test=False):
        prefix = "test" if is_test else "train"
        # Ensure we only use columns that actually exist in the df
        valid_cols = [c for c in cols if c in df.columns]
        X = df[valid_cols]
        # Survival targets
        y_surv = df[[time_col, event_col]]
        # Subtype target
        y_sub = df[[subtype_col]]

        out_path = processed_dir / f"{prefix}_{name}"
        X.to_csv(out_path / "X.csv", index=False)
        y_surv.to_csv(out_path / "y_survival.csv", index=False)
        y_sub.to_csv(out_path / "y_subtype.csv", index=False)

    # 3. Build Group-Specific Sets
    feature_sets = {
        "clinical": clinical_cols,
        "genomic": genomic_cols + mutation_cols,
        "integrated": clinical_cols + genomic_cols + mutation_cols
    }

    for name, cols in feature_sets.items():
        (processed_dir / f"train_{name}").mkdir(parents=True, exist_ok=True)
        (processed_dir / f"test_{name}").mkdir(parents=True, exist_ok=True)
        save_set(train_df, name, cols, is_test=False)
        save_set(test_df, name, cols, is_test=True)

    logger.info("Feature matrices built for clinical, genomic, and integrated groups.")
    return True

if __name__ == "__main__":
    build_features()
