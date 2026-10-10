import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from src.breast_cancer_ml.config.settings import settings, logger

class ClassificationEvaluator:
    """
    Consolidates classification results and generates a comparison report.
    """
    def __init__(self):
        self.metrics_dir = Path("reports/metrics")

    def compare_classification_models(self):
        """
        Reads results for clinical, genomic, and integrated models.
        """
        groups = ["clinical", "genomic", "integrated"]
        comparison_data = []

        for group in groups:
            metrics_file = self.metrics_dir / f"{group}_classification.json"
            if not metrics_file.exists():
                logger.warning(f"Metrics file {metrics_file} not found. Skipping.")
                continue

            with open(metrics_file, "r") as f:
                res = json.load(f)
                for model_name, metrics in res.items():
                    comparison_data.append({
                        "feature_group": group,
                        "model": model_name,
                        "accuracy": metrics["accuracy"],
                        "weighted_f1": metrics["weighted_f1"],
                        "roc_auc": metrics.get("roc_auc", np.nan)
                    })

        df_comp = pd.DataFrame(comparison_data)
        df_comp.to_csv("reports/metrics/subtype_classification.csv", index=False)
        logger.info("Classification comparison table generated.")
        return df_comp

if __name__ == "__main__":
    evaluator = ClassificationEvaluator()
    evaluator.compare_classification_models()
