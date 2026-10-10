import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Any

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.breast_cancer_ml.config.settings import settings, logger

class SubtypeClassifierTrainer:
    """
    Handles training and evaluation of molecular subtype classification models.
    """
    def __init__(self, group_name: str):
        self.group_name = group_name
        self.models = {
            "logistic_regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=settings.get("split.random_state")),
            "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=settings.get("split.random_state"))
        }

    def train_and_evaluate(self):
        """
        Runs 5-fold CV for all classification models.
        """
        processed_dir = Path(settings.get("data.processed_path"))
        X_train_raw = pd.read_csv(processed_dir / f"train_{self.group_name}/X.csv")
        y_train_raw = pd.read_csv(processed_dir / f"train_{self.group_name}/y_subtype.csv")
        X_test_raw = pd.read_csv(processed_dir / f"test_{self.group_name}/X.csv")
        y_test_raw = pd.read_csv(processed_dir / f"test_{self.group_name}/y_subtype.csv")

        # Label encoding for targets (subtype is categorical)
        # we use the first column of y_subtype
        y_train = pd.get_dummies(y_train_raw.iloc[:, 0]).columns.tolist() # just to get classes
        target_col = y_train_raw.iloc[:, 0]
        target_test = y_test_raw.iloc[:, 0]

        # Map classes to integers for XGBoost
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_enc = le.fit_transform(target_col)
        y_test_enc = le.transform(target_test)

        kf = KFold(n_splits=settings.get("cv.folds", 5), shuffle=True, random_state=settings.get("split.random_state"))
        results = {}

        for model_name, model in self.models.items():
            logger.info(f"Training {model_name} for {self.group_name} group...")

            cv_scores = []
            for train_idx, val_idx in kf.split(X_train_raw):
                X_tr, X_val = X_train_raw.iloc[train_idx], X_train_raw.iloc[val_idx]
                y_tr, y_val = y_train_enc[train_idx], y_train_enc[val_idx]

                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', model)
                ])

                pipeline.fit(X_tr, y_tr)
                val_pred = pipeline.predict(X_val)
                cv_scores.append(f1_score(y_val, val_pred, average='weighted'))

            # Final fit on all training data
            final_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            final_pipeline.fit(X_train_raw, y_train_enc)

            test_pred = final_pipeline.predict(X_test_raw)
            test_probs = final_pipeline.predict_proba(X_test_raw)

            # Metrics
            metrics = {
                "accuracy": accuracy_score(y_test_enc, test_pred),
                "weighted_f1": f1_score(y_test_enc, test_pred, average='weighted'),
                "precision": precision_score(y_test_enc, test_pred, average='weighted'),
                "recall": recall_score(y_test_enc, test_pred, average='weighted')
            }

            # ROC-AUC for multi-class (OvR)
            try:
                metrics["roc_auc"] = roc_auc_score(y_test_enc, test_probs, multi_class='ovr')
            except:
                metrics["roc_auc"] = np.nan

            # Save model
            model_path = Path(f"models/classification/{self.group_name}")
            model_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(final_pipeline, model_path / f"{model_name}.joblib")
            joblib.dump(le, model_path / "label_encoder.joblib")

            results[model_name] = {
                "cv_f1": np.mean(cv_scores),
                **metrics
            }

        # Save metrics
        metrics_path = Path("reports/metrics")
        metrics_path.mkdir(parents=True, exist_ok=True)
        with open(metrics_path / f"{self.group_name}_classification.json", "w") as f:
            import json
            json.dump(results, f, indent=4)

        logger.info(f"Classification results for {self.group_name}: {results}")
        return results

if __name__ == "__main__":
    import sys
    group = sys.argv[1] if len(sys.argv) > 1 else "integrated"
    trainer = SubtypeClassifierTrainer(group)
    trainer.train_and_evaluate()
