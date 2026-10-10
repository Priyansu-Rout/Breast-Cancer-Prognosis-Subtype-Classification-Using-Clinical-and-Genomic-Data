import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Using scikit-survival for industry-standard survival analysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from src.breast_cancer_ml.config.settings import settings, logger

class SurvivalModelTrainer:
    """
    Handles training and evaluation of survival models for different feature groups.
    Includes strict 5-fold CV with pipeline-based scaling.
    """
    def __init__(self, group_name: str):
        self.group_name = group_name
        self.models = {
            "cox": CoxPHSurvivalAnalysis(alpha=0.1), # Added L2 regularization to prevent singular matrix
            "random_survival_forest": RandomSurvivalForest(
                n_estimators=100,
                min_samples_split=10,
                random_state=settings.get("split.random_state")
            )
        }


    def _prepare_sksurv_data(self, df_x: pd.DataFrame, df_y: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts pandas DataFrames to scikit-survival structured arrays.
        """
        # Target: structured array with boolean 'event' and float 'time'
        y = np.array(
            [(bool(event), float(time)) for event, time in zip(df_y.iloc[:, 0], df_y.iloc[:, 1])],
            dtype=[('event', 'bool'), ('time', 'float')]
        )
        X = df_x.values
        return X, y

    def train_and_evaluate(self):
        """
        Runs 5-fold CV for all configured survival models.
        """
        processed_dir = Path(settings.get("data.processed_path"))
        X_train_raw = pd.read_csv(processed_dir / f"train_{self.group_name}/X.csv")
        y_train_raw = pd.read_csv(processed_dir / f"train_{self.group_name}/y_survival.csv")
        X_test_raw = pd.read_csv(processed_dir / f"test_{self.group_name}/X.csv")
        y_test_raw = pd.read_csv(processed_dir / f"test_{self.group_name}/y_survival.csv")

        # Convert to sksurv format
        X_train, y_train = self._prepare_sksurv_data(X_train_raw, y_train_raw)
        X_test, y_test = self._prepare_sksurv_data(X_test_raw, y_test_raw)

        kf = KFold(n_splits=settings.get("cv.folds", 5), shuffle=True, random_state=settings.get("split.random_state"))
        results = {}

        for model_name, model in self.models.items():
            logger.info(f"Training {model_name} for {self.group_name} group...")

            cv_indices = []
            for train_idx, val_idx in kf.split(X_train):
                # Split
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                # Pipeline: Scaling -> Model (Critical: Scale fit only on train fold)
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])

                pipeline.fit(X_tr, y_tr)

                # Evaluate fold
                pred = pipeline.predict(X_val)
                c_index = concordance_index_censored(y_val['event'], y_val['time'], pred)[0]
                cv_indices.append(c_index)

            # Final fit on all training data for test set evaluation
            final_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            final_pipeline.fit(X_train, y_train)

            test_pred = final_pipeline.predict(X_test)
            test_cindex = concordance_index_censored(y_test['event'], y_test['time'], test_pred)[0]

            # Save model artifact
            model_path = Path(f"models/survival/{self.group_name}")
            model_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(final_pipeline, model_path / f"{model_name}.joblib")

            results[model_name] = {
                "cv_cindex": np.mean(cv_indices),
                "test_cindex": test_cindex
            }

        # Save metrics
        metrics_path = Path("reports/metrics")
        metrics_path.mkdir(parents=True, exist_ok=True)
        with open(metrics_path / f"{self.group_name}_survival.json", "w") as f:
            import json
            json.dump(results, f, indent=4)

        logger.info(f"Survival results for {self.group_name}: {results}")
        return results

if __name__ == "__main__":
    import sys
    # Support CLI usage: python -m breast_cancer_ml.models.survival clinical
    group = sys.argv[1] if len(sys.argv) > 1 else "integrated"
    trainer = SurvivalModelTrainer(group)
    trainer.train_and_evaluate()
