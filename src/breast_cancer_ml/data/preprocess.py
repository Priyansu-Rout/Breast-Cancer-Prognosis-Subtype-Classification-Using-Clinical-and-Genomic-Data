import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.breast_cancer_ml.config.settings import settings, logger

def preprocess_data():
    """
    End-to-end preprocessing pipeline.
    Handles missing values, encoding, and splitting.
    """
    raw_path = Path(settings.get("data.raw_path"))
    processed_dir = Path(settings.get("data.processed_path"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {raw_path} for preprocessing...")
    df = pd.read_csv(raw_path)

    # 1. Handle Missing Values
    # We use a threshold from config to drop columns with too many NaNs
    max_missing = settings.get("features.max_missing_fraction", 0.3)
    missing_pct = df.isna().mean()
    cols_to_keep = missing_pct[missing_pct <= max_missing].index.tolist()

    logger.info(f"Dropping {len(df.columns) - len(cols_to_keep)} columns due to missingness > {max_missing}")
    df = df[cols_to_keep]

    # Simple imputation for remaining NaNs: median for numeric, mode for categorical
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    # 2. Mutation Encoding & Categorical Handling
    # Convert any mutation columns that aren't already binary
    mut_cols = [col for col in df.columns if col.endswith('_mut')]
    for col in mut_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: 1 if str(x).lower() != 'wild-type' and str(x).lower() != '0' and str(x).lower() != 'nan' else 0)

    # NEW: Encode categorical clinical features (One-Hot Encoding)
    # This prevents "could not convert string to float" errors in the models
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    # We exclude the target columns from encoding
    targets = settings.get("targets").values()
    categorical_cols = [col for col in categorical_cols if col not in targets]

    if categorical_cols:
        logger.info(f"One-hot encoding categorical columns: {categorical_cols}")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


    # 3. Data Splitting
    # We split the entire dataset once to ensure the test set is totally independent
    test_size = settings.get("split.test_size", 0.2)
    random_state = settings.get("split.random_state", 42)

    # For stratified split, we need a target. We use subtype as primary stratification.
    subtype_target = settings.get("targets.subtype")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[subtype_target] if subtype_target in df.columns else None
    )

    # Save split datasets
    train_df.to_csv(processed_dir / "train.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    logger.info(f"Preprocessing complete. Train size: {len(train_df)}, Test size: {len(test_df)}")
    return True

if __name__ == "__main__":
    preprocess_data()
