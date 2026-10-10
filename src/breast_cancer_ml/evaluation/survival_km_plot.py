import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

def plot_integrated_km():
    # 1. Load Data
    data_paths = [
        Path("METABRIC_RNA_Mutation.csv"),
        Path("data/raw/METABRIC_RNA_Mutation.csv"),
        Path("data/METABRIC_RNA_Mutation.csv")
    ]

    df = None
    for p in data_paths:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            print(f"Loaded data from {p}")
            break

    if df is None:
        print("Error: Could not find METABRIC_RNA_Mutation.csv")
        return

    # --- DATASET FACTS ---
    duration_col = 'overall_survival_months'
    status_col = 'overall_survival'

    df = df.copy()
    df[duration_col] = df[duration_col].clip(lower=0.01)
    # Event = 1 if deceased (status == 0), 0 if living
    df['event'] = (df[status_col] == 0).astype(int)

    # 2. Feature Selection
    clinical_cols = [
        'age_at_diagnosis', 'tumor_size', 'neoplasm_histologic_grade',
        'lymph_nodes_examined_positive', 'tumor_stage', 'er_status',
        'pr_status', 'her2_status', 'type_of_breast_surgery',
        'chemotherapy', 'hormone_therapy', 'radio_therapy'
    ]

    all_cols = df.columns.tolist()
    try:
        start_idx = all_cols.index('brca1')
        end_idx = next(i for i, col in enumerate(all_cols) if col.endswith('_mut'))
        gene_cols = all_cols[start_idx:end_idx]
    except (ValueError, StopIteration):
        gene_cols = [c for c in all_cols if c not in clinical_cols + [duration_col, status_col, 'event']]

    features = clinical_cols + gene_cols

    # 3. Out-of-Fold Risk Score Computation using lifelines
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    risk_scores = np.zeros(len(df))

    numeric_features = df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df[features].select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

    for train_idx, test_idx in skf.split(df, df['event']):
        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]

        # Process features
        X_train_proc = preprocessor.fit_transform(df_train[features])
        X_test_proc = preprocessor.transform(df_test[features])

        # Create a temporary DataFrame for lifelines
        train_df = pd.DataFrame(X_train_proc, columns=[f"feat_{i}" for i in range(X_train_proc.shape[1])])
        train_df['duration'] = df_train[duration_col].values
        train_df['event'] = df_train['event'].values

        try:
            # Use CoxPHFitter from lifelines - much more stable
            cph = CoxPHFitter(penalizer=0.1) # Penalizer prevents the NaN crash
            cph.fit(train_df, duration_col='duration', event_col='event')

            test_df = pd.DataFrame(X_test_proc, columns=[f"feat_{i}" for i in range(X_test_proc.shape[1])])
            risk_scores[test_idx] = cph.predict_partial_hazard(test_df)
        except Exception as e:
            print(f"Fold failed: {e}. Using fallback.")
            risk_scores[test_idx] = 0

    # 4. Sanity Checks
    print("\n--- SANITY CHECKS ---")
    print(f"Duration Min/Max: {df[duration_col].min():.2f} / {df[duration_col].max():.2f}")
    print(f"Events: {df['event'].sum()} / Total: {len(df)}")
    print(f"Risk Score NaNs: {np.isnan(risk_scores).sum()}")
    print(f"Unique Risk Values: {len(np.unique(risk_scores))}")

    median_risk = np.median(risk_scores)
    high_risk_mask = risk_scores >= median_risk
    low_risk_mask = ~high_risk_mask

    print(f"High Risk Group Size: {high_risk_mask.sum()}")
    print(f"Low Risk Group Size: {low_risk_mask.sum()}")

    assert df[duration_col].max() > 50, f"Duration max is too low: {df[duration_col].max()}"

    # 5. Plotting
    plt.figure(figsize=(10, 7))
    kmf = KaplanMeierFitter()

    kmf.fit(df[duration_col][low_risk_mask], df['event'][low_risk_mask], label=f"Low Risk (n={low_risk_mask.sum()})")
    kmf.plot_survival_function(ci_show=True, color='blue')

    kmf.fit(df[duration_col][high_risk_mask], df['event'][high_risk_mask], label=f"High Risk (n={high_risk_mask.sum()})")
    kmf.plot_survival_function(ci_show=True, color='red')

    results = logrank_test(df[duration_col][low_risk_mask], df[duration_col][high_risk_mask],
                           df['event'][low_risk_mask], df['event'][high_risk_mask])

    plt.title(f"Kaplan-Meier Survival Curve • Integrated Model\n(p-value: {results.p_value:.4e})")
    plt.xlabel("Months")
    plt.ylabel("Survival probability")
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()

    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "integrated_km.png")
    print(f"\nPlot saved to {output_dir / 'integrated_km.png'}")
    plt.show()

if __name__ == "__main__":
    plot_integrated_km()
