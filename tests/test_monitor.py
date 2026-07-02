import numpy as np

from src.monitor import calculate_psi, check_alerts


def test_psi_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1000)
    assert calculate_psi(x, x) < 0.01


def test_psi_high_for_shifted_distribution():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 1000)
    shifted = rng.normal(3, 1, 1000)
    assert calculate_psi(ref, shifted) > 0.25


def test_check_alerts_flags_low_f1():
    weekly = [{"week": 1, "f1": 0.4, "roc_auc": 0.5, "actual_churn_rate": 0.3}]
    import pandas as pd
    drift = pd.DataFrame([{"psi_level": "OK", "psi": 0.0}])
    alerts = check_alerts(weekly, weekly[0], drift)
    assert any(a["level"] == "CRITICAL" for a in alerts)
