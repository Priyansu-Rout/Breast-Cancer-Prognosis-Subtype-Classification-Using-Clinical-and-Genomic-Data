import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from src.breast_cancer_ml.config.settings import settings, logger

def validate_dataset():
    """
    Strict data validation stage for METABRIC dataset.
    Verifies schema, missingness, and target availability.
    """
    raw_path = Path(settings.get("data.raw_path"))
    if not raw_path.exists():
        logger.error(f"Dataset not found at {raw_path}")
        raise FileNotFoundError(f"Dataset missing: {raw_path}")

    logger.info(f"Loading dataset from {raw_path} for validation...")
    df = pd.read_csv(raw_path)

    # 1. Basic Stats
    stats = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "duplicates": df.duplicated().sum(),
        "duplicate_cols": df.columns[df.columns.duplicated()].tolist()
    }
    logger.info(f"Dataset shape: {df.shape}. Duplicates: {stats['duplicates']}")

    # 2. Target Verification
    targets = settings.get("targets")
    missing_targets = [t for t in targets.values() if t not in df.columns]
    if missing_targets:
        logger.error(f"Missing target columns: {missing_targets}")
        raise KeyError(f"Critical targets missing from dataset: {missing_targets}")

    # 3. Data Quality Report
    quality_report = {
        "stats": {k: int(v) if isinstance(v, (np.integer, np.int64)) else v for k, v in stats.items()},
        "missingness": {k: int(v) if isinstance(v, (np.integer, np.int64)) else v for k, v in df.isna().sum().to_dict().items()},
        "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "targets_found": targets
    }


    # Save reports
    report_dir = Path(settings.get("data.data_dictionary_path"))
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "data_quality.json", "w") as f:
        json.dump(quality_report, f, indent=4)

    # Generate Data Dictionary
    data_dict = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.values,
        "missing_count": df.isna().sum().values,
        "missing_pct": (df.isna().sum().values / len(df)) * 100
    })
    data_dict.to_csv(report_dir / "data_dictionary.csv", index=False)

    logger.info("Data validation successful.")
    return True

if __name__ == "__main__":
    validate_dataset()
