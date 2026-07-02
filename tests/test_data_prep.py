import pandas as pd

from src.data_prep import COLUMN_MAP, NUM_FEATURES, preprocess


def _raw():
    return pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4"],
        "gender": ["Male", "Female", "Male", "Female"],
        "senior_citizen": [0, 1, 0, 1],
        "tenure_months": [12, 3, 60, 24],
        "contract_type": ["Month-to-Month", "Two Year", "One Year", "Month-to-Month"],
        "internet_service": ["Fiber", None, "DSL", "Fiber"],
        "monthly_charges": [70.0, 90.0, 45.0, 80.0],
        "support_tickets_last_6m": [2, 6, 0, 3],
        "churn": ["No", "Yes", "No", "Yes"],
    }).rename(columns=COLUMN_MAP)


def test_preprocess_target_is_binary():
    X, y = preprocess(_raw())
    assert set(y.unique()) <= {0, 1}
    assert y.tolist() == [0, 1, 0, 1]


def test_preprocess_no_nan_and_numeric():
    X, y = preprocess(_raw())
    assert not X.isnull().any().any()
    assert all(c in X.columns for c in NUM_FEATURES)
    # customerID must be dropped
    assert "customerID" not in X.columns


def test_internet_service_nan_imputed():
    X, _ = preprocess(_raw())
    # row 2 had None -> becomes "No" service dummy
    assert "InternetService_No" in X.columns
    assert X.loc[1, "InternetService_No"] == 1
