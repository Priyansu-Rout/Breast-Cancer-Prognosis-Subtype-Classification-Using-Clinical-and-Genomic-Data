import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sksurv.ensemble import RandomSurvivalForest

def plot_km(risk_scores, duration, events, title, ax):
    """Shared function to plot Kaplan-Meier curves."""
    median_risk = np.median(risk_scores)
    high_mask = risk_scores >= median_risk
    low_mask = ~high_mask

    kmf = KaplanMeierFitter()
    kmf.fit(duration[low_mask], events[low_mask], label=f"Low Risk (n={low_mask.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color='blue')
    kmf.fit(duration[high_mask], events[high_mask], label=f"High Risk (n={high_mask.sum()})")
    kmf.plot_survival_function(ax=ax, ci_show=True, color='red')

    results = logrank_test(duration[low_mask], duration[high_mask], events[low_mask], events[high_mask])
    ax.set_title(f"{title}\n(p-value: {results.p_value:.4e})")
    ax.set_xlabel("Months")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()

def compute_oof_risk_fast(df, features, duration_col, event_col):
    """Computes risk scores using Random Survival Forest (FAST)."""
    X = df[features]
    y_time = df[duration_col].values
    y_event = df[event_col].values

    y = np.array([(bool(e), t) for e, t in zip(y_event, y_time)], dtype=[('event', 'bool'), ('time', 'float')])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    risk_scores = np.zeros(len(df))

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

    for train_idx, test_idx in skf.split(X, y_event):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y[train_idx]

        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        model = RandomSurvivalForest(n_estimators=50, min_samples_split=10, random_state=42, n_jobs=-1)
        model.fit(X_train_proc, y_train)
        risk_scores[test_idx] = model.predict(X_test_proc)

    return risk_scores

def run_survival_comparison():
    data_paths = [Path("METABRIC_RNA_Mutation.csv"), Path("data/raw/METABRIC_RNA_Mutation.csv"), Path("data/METABRIC_RNA_Mutation.csv")]
    df = None
    for p in data_paths:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            print(f"Loaded data from {p}")
            break

    if df is None:
        print("Error: Dataset not found.")
        return

    DURATION_COL = 'overall_survival_months'
    STATUS_COL = 'overall_survival'
    df = df.copy()
    df[DURATION_COL] = df[DURATION_COL].clip(lower=0.01)
    df['event'] = (df[STATUS_COL] == 0).astype(int)

    duration = df[DURATION_COL].values
    events = df['event'].values

    clinical_cols = ['age_at_diagnosis', 'tumor_size', 'neoplasm_histologic_grade', 'lymph_nodes_examined_positive', 'tumor_stage', 'er_status', 'pr_status', 'her2_status', 'type_of_breast_surgery', 'chemotherapy', 'hormone_therapy', 'radio_therapy']
    all_cols = df.columns.tolist()
    try:
        start_idx = all_cols.index('brca1')
        end_idx = next(i for i, col in enumerate(all_cols) if col.endswith('_mut'))
        gene_cols = all_cols[start_idx:end_idx]
    except (ValueError, StopIteration):
        gene_cols = [c for c in all_cols if c not in clinical_cols + [DURATION_COL, STATUS_COL, 'event']]
    mutation_cols = [c for c in all_cols if c.endswith('_mut')]

    model_configs = {
        "Clinical Model": clinical_cols,
        "Genomic Model": gene_cols,
        "Integrated Model": clinical_cols + gene_cols + mutation_cols
    }

    output_dir = Path("reports/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

    for i, (name, features) in enumerate(model_configs.items()):
        print(f"Processing {name}...")
        risk = compute_oof_risk_fast(df, features, DURATION_COL, 'event')
        print(f"  - Duration Max: {duration.max():.2f} | Events: {events.sum()} | Unique Risks: {len(np.unique(risk))}")

        median_r = np.median(risk)
        plot_km(risk, duration, events, name, axes[i])

        plt.figure(figsize=(8, 6))
        kmf_ind = KaplanMeierFitter()
        kmf_ind.fit(duration[risk < median_r], events[risk < median_r], label=f"Low Risk (n={(risk < median_r).sum()})")
        kmf_ind.plot_survival_function(ci_show=True, color='blue')
        kmf_ind.fit(duration[risk >= median_r], events[risk >= median_r], label=f"High Risk (n={(risk >= median_r).sum()})")
        kmf_ind.plot_survival_function(ci_show=True, color='red')

        res_ind = logrank_test(duration[risk < median_r], duration[risk >= median_r], events[risk < median_r], events[risk >= median_r])
        plt.title(f"{name}\n(p-value: {res_ind.p_value:.4e})")
        plt.xlabel("Months")
        plt.ylabel("Survival probability")
        plt.ylim(0, 1.02)
        plt.grid(True, alpha=0.3)
        plt.legend()

        fname = {"Clinical Model": "clinical_km.png", "Genomic Model": "genomic_km.png", "Integrated Model": "integrated_km.png"}[name]
        plt.savefig(output_dir / fname)
        plt.close()

    plt.tight_layout()
    plt.savefig(output_dir / "survival_comparison_km.png")
    print(f"\nSUCCESS: All plots saved to {output_dir}")
    plt.show()

if __name__ == "__main__":
    run_survival_comparison()
