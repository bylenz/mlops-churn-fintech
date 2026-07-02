"""Phase 3 - model registry and lifecycle.

Registers the best run in the MLflow Model Registry, moves it through
None -> Staging -> Production with an F1 gate, and exports the
production artifacts consumed by the API.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score

DATA_DIR = "data"
REGISTERED_MODEL_NAME = "churn-predictor"
PRODUCTION_F1_THRESHOLD = 0.50
EXPERIMENT_NAME = "churn-prediction-fintech"


def run(data_dir: str = DATA_DIR) -> None:
    mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")
    client = MlflowClient()

    with open(os.path.join(data_dir, "best_model_meta.json")) as f:
        meta = json.load(f)
    best_run_id = meta["best_run_id"]
    best_model_name = meta["best_model"]
    print(f"Best model: {best_model_name} | run_id: {best_run_id}")

    # 1. Register
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    print(f"Registered {mv.name} v{mv.version} (stage={mv.current_stage})")

    # 2. Describe + tag
    client.update_registered_model(
        name=REGISTERED_MODEL_NAME,
        description="Churn prediction model for Fintech customers. Telco Customer Churn dataset.",
    )
    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=mv.version,
        description=(
            f"Trained with {best_model_name}. "
            f"F1={meta['metrics']['f1']:.4f} | ROC-AUC={meta['metrics']['roc_auc']:.4f}"
        ),
    )
    client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "dataset", "telco_churn")
    client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, "model_type", best_model_name)

    # 3. None -> Staging
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME, version=mv.version,
        stage="Staging", archive_existing_versions=False,
    )
    print(f"v{mv.version} -> Staging")

    # 4. Validate in Staging
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze("columns")
    staging_model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/Staging")
    staging_f1 = f1_score(y_test, staging_model.predict(X_test), zero_division=0)
    print(f"Staging F1: {staging_f1:.4f} | threshold: {PRODUCTION_F1_THRESHOLD}")

    # 5. Promote to Production if it passes
    if staging_f1 < PRODUCTION_F1_THRESHOLD:
        print(f"Validation failed: F1={staging_f1:.4f} < {PRODUCTION_F1_THRESHOLD}")
        sys.exit(1)

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME, version=mv.version,
        stage="Production", archive_existing_versions=True,
    )
    print(f"v{mv.version} -> Production")

    # 6. Export for deployment
    prod_model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/Production")
    joblib.dump(prod_model, os.path.join(data_dir, "production_model.pkl"))
    prod_info = {
        "model_name": REGISTERED_MODEL_NAME,
        "version": str(mv.version),
        "stage": "Production",
        "run_id": best_run_id,
        "model_type": best_model_name,
        "model_uri": f"models:/{REGISTERED_MODEL_NAME}/Production",
    }
    with open(os.path.join(data_dir, "production_model_info.json"), "w") as f:
        json.dump(prod_info, f, indent=2)
    print("Saved production_model.pkl and production_model_info.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register and promote best model")
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()
    run(args.data_dir)
