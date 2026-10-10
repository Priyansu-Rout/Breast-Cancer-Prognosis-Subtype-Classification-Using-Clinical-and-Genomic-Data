import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from pathlib import Path
from typing import Dict, Any

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from src.breast_cancer_ml.config.settings import settings, logger

class SurvivalEvaluator:
    """
    Consolidates survival results and generates risk-stratified Kaplan-Meier plots.
    """
    def __init__(self):
        self.metrics_dir = Path("reports/metrics")
        self.figs_dir = Path("reports/figures")
        self.figs_dir.mkdir(parents=True, exist_ok=True)

    def compare_survival_models(self):
        """
        Reads results for clinical, genomic, and integrated models and creates a comparison table.
        """
        groups = ["clinical", "genomic", "integrated"]
        comparison_data = []

        for group in groups:
            metrics_file = self.metrics_dir / f"{group}_survival.json"
            if not metrics_file.exists():
                logger.warning(f"Metrics file {metrics_file} not found. Skipping.")
                continue

            with open(metrics_file, "r") as f:
                res = json.load(f)
                for model_name, metrics in res.items():
                    comparison_data.append({
                        "feature_group": group,
                        "model": model_name,
                        "cv_cindex": metrics["cv_cindex"],
                        "test_cindex": metrics["test_cindex"]
                    })

        df_comp = pd.DataFrame(comparison_data)
        df_comp.to_csv("reports/metrics/survival_comparison.csv", index=False)
        logger.info("Survival comparison table generated.")
        return df_comp

    def plot_kaplan_meier(self, group_name: str):
        """
        Generates Kaplan-Meier survival curves based on risk scores.
        """
        processed_dir = Path(settings.get("data.processed_path"))
        X_test = pd.read_csv(processed_dir / f"test_{group_name}/X.csv")
        y_test = pd.read_csv(processed_dir / f"test_{group_name}/y_survival.csv")

        # Load the best model (RSF usually performs better)
        import joblib
        model_path = Path(f"models/survival/{group_name}/random_survival_forest.joblib")
        if not model_path.exists():
            logger.error(f"Model {model_path} not found. Cannot plot KM.")
            return

        model = joblib.load(model_path)
        # Predict risk scores (higher = higher risk)
        risk_scores = model.predict(X_test)

        # Stratify by median risk
        threshold = np.median(risk_scores)
        high_risk = risk_scores >= threshold
        low_risk = risk_scores < threshold

        kmf = KaplanMeierFitter()

        plt.figure(figsize=(10, 6))

        # Plot High Risk
        kmf.fit(y_test.iloc[:, 1], event_observed=y_test.iloc[:, 0], label="High Risk")
        kmf.plot_survival_function()

        # Plot Low Risk
        kmf.fit(y_test.iloc[:, 1], event_observed=y_test.iloc[:, 0], label="Low Risk")
        kmf.plot_survival_function()

        # Log-rank test
        # Ensure we are passing the correct columns: time (index 1), event (index 0)
        # high_risk and low_risk are boolean masks
        time_high = y_test.iloc[:, 1][high_risk]
        event_high = y_test.iloc[:, 0][high_risk]
        time_low = y_test.iloc[:, 1][low_risk]
        event_low = y_test.iloc[:, 0][low_risk]

        try:
            results = logrank_test(
                time_high, event_high,
                time_low, event_low
            )
            p_val = results.p_value
        except Exception as e:
            logger.warning(f"Log-rank test failed for {group_name}: {e}")
            p_val = np.nan

        plt.title(f"Kaplan-Meier Survival Curve - {group_name.capitalize()} Model\n(p-value: {p_val:.4f})")

        plt.xlabel("Months")
        plt.ylabel("Survival Probability")
        plt.grid(True)
        plt.savefig(self.figs_dir / f"{group_name}_km.png")
        plt.close()

        logger.info(f"KM plot generated for {group_name} with p-value {p_val:.4f}")

if __name__ == "__main__":
    evaluator = SurvivalEvaluator()
    evaluator.compare_survival_models()
    for group in ["clinical", "genomic", "integrated"]:
        evaluator.plot_kaplan_meier(group)
