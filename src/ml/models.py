import logging
import tempfile
from pathlib import Path
from typing import Any, Dict

import mlflow
import mlflow.sklearn
import pandas as pd
import polars as pl
from deltalake import DeltaTable
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.common.config import (
    DELAY_THRESHOLD,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from src.common.utils import safe_run

_LOGGER = logging.getLogger(__name__)


class MLModelTrainer:
    def __init__(
        self,
        features_path: Path,
        test_size: float = 0.2,
        random_state: int = 42,
        delay_threshold: int = DELAY_THRESHOLD,
        mlflow_tracking_uri: str = MLFLOW_TRACKING_URI,
        mlflow_experiment_name: str = MLFLOW_EXPERIMENT_NAME,
        sample_fraction: float = 1.0,
    ):
        self.features_path = features_path
        self.test_size = test_size
        self.random_state = random_state
        self.delay_threshold = delay_threshold
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.sample_fraction = sample_fraction
        self.feature_columns = [
            "hour",
            "day_of_week",
            "month",
            "season",
            "DISTANCE",
            "DEP_DELAY",
            "OP_UNIQUE_CARRIER",
            "ORIGIN",
            "DEST",
        ]
        self.target_regression = "ARR_DELAY"
        self.target_classification = "is_delayed"
        self.models: Dict[str, Any] = {}

    @safe_run(default_return_value=pd.DataFrame())
    def load_features(self) -> pd.DataFrame:
        lf = pl.scan_delta(str(self.features_path))
        if 0 < self.sample_fraction < 1:
            lf = (
                lf.collect()
                .sample(fraction=self.sample_fraction, seed=self.random_state)
                .lazy()
            )
        df = lf.collect()
        _LOGGER.info(
            f"Loaded ML feature table from {self.features_path} ({len(df)} rows)"
        )
        return df.to_pandas()

    def prepare_data(self) -> Dict[str, pd.DataFrame]:
        df = self.load_features()
        if (
            self.target_regression not in df.columns
            or self.target_classification not in df.columns
        ):
            raise ValueError("ML feature table must contain ARR_DELAY and is_delayed")

        df = df.dropna(
            subset=self.feature_columns
            + [self.target_regression, self.target_classification]
        )
        if df.empty:
            raise ValueError("ML feature table contains no usable rows")

        X = pd.get_dummies(df[self.feature_columns], drop_first=True)
        y_reg = df[self.target_regression]
        y_clf = df[self.target_classification].astype(int)

        stratify = None
        class_counts = y_clf.value_counts()
        if len(class_counts) > 1 and class_counts.min() >= 2:
            stratify = y_clf

        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = (
            train_test_split(
                X,
                y_reg,
                y_clf,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify,
            )
        )

        _LOGGER.info(
            f"Prepared ML training data: {len(X_train)} train rows, {len(X_test)} test rows"
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_reg_train": y_reg_train,
            "y_reg_test": y_reg_test,
            "y_clf_train": y_clf_train,
            "y_clf_test": y_clf_test,
        }

    def train_regression_models(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}

        regression_models: list[tuple[str, Any]] = [
            ("linear_regression", LinearRegression()),
            (
                "random_forest",
                RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            ),
        ]
        for name, model in regression_models:
            _LOGGER.info(f"Training regression model: {name}")
            model.fit(data["X_train"], data["y_reg_train"])
            pred = model.predict(data["X_test"])
            metrics = {
                "r2": float(r2_score(data["y_reg_test"], pred)),
                "mae": float(mean_absolute_error(data["y_reg_test"], pred)),
                "rmse": float(mean_squared_error(data["y_reg_test"], pred) ** 0.5),
            }
            results[name] = metrics
            self.models[f"regression_{name}"] = model
            _LOGGER.info(f"{name} regression metrics = {metrics}")

        return results

    def train_classification_models(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}

        classification_models: list[tuple[str, Any]] = [
            (
                "logistic_regression",
                LogisticRegression(max_iter=1000, random_state=self.random_state),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=100, random_state=self.random_state
                ),
            ),
        ]
        for name, model in classification_models:
            _LOGGER.info(f"Training classification model: {name}")
            model.fit(data["X_train"], data["y_clf_train"])
            pred = model.predict(data["X_test"])
            metrics = {
                "accuracy": float(accuracy_score(data["y_clf_test"], pred)),
                "f1": float(f1_score(data["y_clf_test"], pred, zero_division=0)),
            }
            if hasattr(model, "predict_proba") and len(set(data["y_clf_test"])) > 1:
                proba = model.predict_proba(data["X_test"])[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(data["y_clf_test"], proba))
            results[name] = metrics
            self.models[f"classification_{name}"] = model
            _LOGGER.info(f"{name} classification metrics = {metrics}")

        return results

    def _feature_importance_frame(self, feature_names: pd.Index) -> pd.DataFrame:
        rows = []
        for model_name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                for feature, importance in zip(
                    feature_names, model.feature_importances_
                ):
                    rows.append(
                        {
                            "model": model_name,
                            "feature": feature,
                            "importance": float(importance),
                        }
                    )
        return pd.DataFrame(rows).sort_values(
            ["model", "importance"], ascending=[True, False]
        )

    def _gold_version(self) -> int:
        return DeltaTable(str(self.features_path)).version()

    def log_to_mlflow(
        self,
        results: Dict[str, Dict[str, Dict[str, float]]],
        feature_names: pd.Index,
        input_example: pd.DataFrame,
    ) -> None:
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(self.mlflow_experiment_name)
        gold_version = self._gold_version()

        with mlflow.start_run(run_name=f"flight-delay-gold-v{gold_version}"):
            mlflow.log_params(
                {
                    "test_size": self.test_size,
                    "random_state": self.random_state,
                    "delay_threshold": self.delay_threshold,
                    "gold_features_version": gold_version,
                    "sample_fraction": self.sample_fraction,
                }
            )
            mlflow.set_tag("gold_features_path", str(self.features_path))

            for task, task_results in results.items():
                for model_name, metrics in task_results.items():
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{task}_{model_name}_{metric_name}", value)

            for model_name, model in self.models.items():
                signature = infer_signature(input_example, model.predict(input_example))
                mlflow.sklearn.log_model(
                    model,
                    artifact_path=model_name,
                    input_example=input_example,
                    signature=signature,
                )

            importance = self._feature_importance_frame(feature_names)
            if not importance.empty:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    path = Path(tmp_dir) / "feature_importance.csv"
                    importance.to_csv(path, index=False)
                    mlflow.log_artifact(str(path), artifact_path="feature_importance")

    def run(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        data = self.prepare_data()
        results = {
            "regression": self.train_regression_models(data),
            "classification": self.train_classification_models(data),
        }
        input_example = data["X_train"].head(5)
        self.log_to_mlflow(results, data["X_train"].columns, input_example)
        return results
