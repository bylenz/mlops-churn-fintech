"""Phase 1 - EDA & data preparation.

Turns the raw Telco churn CSV into versioned, model-ready artifacts:
X_train/X_test/y_train/y_test CSVs, a fitted StandardScaler and the
ordered feature-name list. Reproducible and CI-invocable.
"""
from __future__ import annotations

import argparse
import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_FILE = "telco_customer_churn_mlops.csv"
NUM_FEATURES = ["tenure", "MonthlyCharges", "support_tickets"]

COLUMN_MAP = {
    "customer_id": "customerID",
    "senior_citizen": "SeniorCitizen",
    "tenure_months": "tenure",
    "contract_type": "Contract",
    "internet_service": "InternetService",
    "monthly_charges": "MonthlyCharges",
    "support_tickets_last_6m": "support_tickets",
    "churn": "Churn",
}


def load_raw(data_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, RAW_FILE))
    return df.rename(columns=COLUMN_MAP)


def preprocess(df: pd.DataFrame):
    """Return (X, y) fully encoded, un-scaled."""
    df = df.copy()
    df = df.drop(columns=["customerID", "tenure_bin"], errors="ignore")

    # 30 NaN in InternetService are assumed to mean "no service".
    df["InternetService"] = df["InternetService"].fillna("No")

    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df["gender"] = LabelEncoder().fit_transform(df["gender"])

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def run(data_dir: str = "data") -> None:
    df = load_raw(data_dir)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"NaN in InternetService: {df['InternetService'].isnull().sum()}")

    X, y = preprocess(df)
    churn_rate = y.mean()
    print(f"Churn rate: {churn_rate:.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUM_FEATURES] = scaler.fit_transform(X_train[NUM_FEATURES])
    X_test[NUM_FEATURES] = scaler.transform(X_test[NUM_FEATURES])
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    os.makedirs(data_dir, exist_ok=True)
    X_train.to_csv(os.path.join(data_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(data_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(data_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(data_dir, "y_test.csv"), index=False)
    joblib.dump(scaler, os.path.join(data_dir, "scaler.pkl"))
    pd.Series(X_train.columns.tolist()).to_csv(
        os.path.join(data_dir, "feature_names.csv"), index=False, header=False
    )
    print("Artifacts saved to", data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare churn dataset")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    run(args.data_dir)
