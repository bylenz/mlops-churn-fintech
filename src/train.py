"""Phase 2 - training with MLflow tracking.

Trains Logistic Regression, Random Forest and XGBoost, logs params and
metrics to MLflow, then selects the best model by F1 and exports it plus
its metadata for the registry phase.
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

DATA_DIR = "data"
DEFAULT_EXPERIMENT = "churn-prediction-fintech"


def load_data(data_dir: str):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze("columns")
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze("columns")
    return X_train, X_test, y_train, y_test


def train_and_log(model, model_name, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tags({"model_type": model_name, "dataset": "telco_churn"})
        mlflow.log_params(params)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model", input_example=X_train.head(3))
        return run.info.run_id, metrics, model


def build_models(y_train):
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return {
        "logistic": (
            "LogisticRegression",
            LogisticRegression,
            {
                "C": 1.0,
                "max_iter": 500,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "random_state": 42,
            },
        ),
        "random_forest": (
            "RandomForest",
            RandomForestClassifier,
            {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            },
        ),
        "xgboost": (
            "XGBoost",
            XGBClassifier,
            {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": round(float(scale_pos_weight), 2),
                "random_state": 42,
                "eval_metric": "logloss",
                "verbosity": 0,
            },
        ),
    }


def run(model_choice: str, experiment: str, data_dir: str = DATA_DIR) -> None:
    tracking_uri = f"file://{os.path.abspath('mlruns')}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    print(f"Tracking URI: {tracking_uri} | Experiment: {experiment}")

    X_train, X_test, y_train, y_test = load_data(data_dir)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    specs = build_models(y_train)
    selected = specs.keys() if model_choice == "all" else [model_choice]

    results = {}
    fitted = {}
    for key in selected:
        display_name, cls, params = specs[key]
        run_id, metrics, model = train_and_log(
            cls(**params), display_name, params, X_train, X_test, y_train, y_test
        )
        results[display_name] = metrics
        fitted[display_name] = (model, run_id)
        print(f"{display_name} -> F1={metrics['f1']} | ROC-AUC={metrics['roc_auc']}")

    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model, best_run_id = fitted[best_name]
    best_metrics = results[best_name]
    print(f"\nBest model: {best_name} (F1={best_metrics['f1']})")

    joblib.dump(best_model, os.path.join(data_dir, "best_model.pkl"))
    with open(os.path.join(data_dir, "best_model_meta.json"), "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "best_run_id": best_run_id,
                "metrics": best_metrics,
            },
            f,
            indent=2,
        )
    print("Saved best_model.pkl and best_model_meta.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train churn models with MLflow")
    parser.add_argument(
        "--model",
        default="all",
        choices=["logistic", "random_forest", "xgboost", "all"],
    )
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()
    run(args.model, args.experiment, args.data_dir)
