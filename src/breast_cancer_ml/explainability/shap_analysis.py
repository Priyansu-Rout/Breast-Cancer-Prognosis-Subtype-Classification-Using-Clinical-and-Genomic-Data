import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.breast_cancer_ml.config.settings import settings, logger

class ExplainabilityManager:
    """
    Handles model explainability and candidate biomarker discovery using SHAP.
    """
    def __init__(self):
        self.figs_dir = Path("reports/figures")
        self.tables_dir = Path("reports/tables")
        self.figs_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def analyze_survival_biomarkers(self, group_name: str):
        """
        Identifies candidate prognostic biomarkers for survival models.
        """
        processed_dir = Path(settings.get("data.processed_path"))
        X_test = pd.read_csv(processed_dir / f"test_{group_name}/X.csv")

        model_path = Path(f"models/survival/{group_name}/random_survival_forest.joblib")
        if not model_path.exists():
            logger.error(f"Model {model_path} not found for {group_name}.")
            return

        # Load pipeline (Scaler + RSF)
        pipeline = joblib.load(model_path)
        model = pipeline.named_steps['model']
        scaler = pipeline.named_steps['scaler']

        # Transform X_test for SHAP
        X_scaled = scaler.transform(X_test)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_test.columns)

        # SHAP KernelExplainer is slow; for RSF we use TreeExplainer if compatible
        # RSF from sksurv is a forest of trees, but not exactly a sklearn forest.
        # We use a sample for KernelExplainer for stability.
        try:
            # Use a subset for SHAP to avoid memory crashes on genomic data
            X_sample = X_scaled_df.sample(n=min(100, len(X_scaled_df)), random_state=42)

            # Survival models predict risk. We wrap the predict function.
            explainer = shap.KernelExplainer(model.predict, X_sample)
            shap_values = explainer.shap_values(X_sample)

            # Global Importance
            importance_df = pd.DataFrame({
                "feature": X_test.columns,
                "importance": np.abs(shap_values).mean(axis=0)
            }).sort_values(by="importance", ascending=False)

            # Save top biomarkers
            biomarker_path = self.tables_dir / f"selected_{group_name}_features.csv"
            importance_df.head(50).to_csv(biomarker_path, index=False)

            # Plots
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title(f"SHAP Summary - {group_name} Survival")
            plt.savefig(self.figs_dir / f"{group_name}_shap_summary.png", bbox_inches='tight')
            plt.close()

            logger.info(f"Biomarker analysis complete for {group_name}. Results saved to {biomarker_path}")

        except Exception as e:
            logger.error(f"SHAP analysis failed for {group_name}: {e}")

    def generate_patient_explanation(self, group_name: str, patient_index: int = 0):
        """
        Generates a human-readable explanation for a single patient's risk.
        """
        processed_dir = Path(settings.get("data.processed_path"))
        X_test = pd.read_csv(processed_dir / f"test_{group_name}/X.csv")

        model_path = Path(f"models/survival/{group_name}/random_survival_forest.joblib")
        pipeline = joblib.load(model_path)

        # Get risk score
        patient_data = X_test.iloc[[patient_index]]
        risk_score = pipeline.predict(patient_data)[0]

        # In a real scenario, we would use the pre-calculated SHAP values for this patient
        # For this demo, we'll report the risk score and a placeholder for top features
        explanation = {
            "patient_id": patient_index,
            "risk_score": float(risk_score),
            "risk_category": "High" if risk_score > np.median(pipeline.predict(X_test)) else "Low",
            "top_risk_increasing": ["Feature A", "Feature B"], # Placeholders for SHAP values
            "top_risk_decreasing": ["Feature C", "Feature D"]
        }

        return explanation

if __name__ == "__main__":
    explainer = ExplainabilityManager()
    for group in ["clinical", "genomic", "integrated"]:
        explainer.analyze_survival_biomarkers(group)
