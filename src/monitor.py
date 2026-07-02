"""Phase 5 - production monitoring.

Simulates weekly production traffic with gradual drift, detects data drift
(KS test + PSI), tracks model degradation, raises alerts and writes a JSON
report. Optionally publishes metrics to a Prometheus Pushgateway and/or
CloudWatch.
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

DATA_DIR = "data"
NUM_DRIFT_COLS = ["tenure", "MonthlyCharges", "support_tickets"]

THRESHOLDS = {
    "f1_min": 0.60,
    "roc_auc_min": 0.65,
    "churn_rate_delta": 0.10,
    "psi_critical": 0.25,
}


def simulate_production_week(X_base, y_base, model, week, drift_factor=0.0):
    X_week = X_base.copy()
    if drift_factor > 0:
        for col in NUM_DRIFT_COLS:
            if col in X_week.columns:
                X_week[col] = X_week[col] + np.random.normal(drift_factor, 0.1, len(X_week))

    y_pred = model.predict(X_week)
    y_prob = model.predict_proba(X_week)[:, 1]

    label_noise = np.random.random(len(X_base)) < (0.05 * week * drift_factor)
    y_actual = y_base.values.copy()
    y_actual[label_noise] = 1 - y_actual[label_noise]

    return X_week, {
        "week": week,
        "accuracy": round(float(accuracy_score(y_actual, y_pred)), 4),
        "f1": round(float(f1_score(y_actual, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_actual, y_prob)), 4),
        "gini": round(float(2 * roc_auc_score(y_actual, y_prob) - 1), 4),
        "predicted_churn_rate": round(float(y_pred.mean()), 4),
        "actual_churn_rate": round(float(y_actual.mean()), 4),
    }


def calculate_psi(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    exp_pct = np.histogram(expected, breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual, breakpoints)[0] / len(actual)
    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def detect_drift(X_reference, X_current):
    rows = []
    for col in X_reference.select_dtypes(include=[np.number]).columns:
        ref_vals = X_reference[col].dropna()
        cur_vals = X_current[col].dropna()
        _, p_value = stats.ks_2samp(ref_vals, cur_vals)
        psi = calculate_psi(ref_vals.values, cur_vals.values)
        rows.append({
            "feature": col,
            "mean_ref": round(float(ref_vals.mean()), 4),
            "mean_actual": round(float(cur_vals.mean()), 4),
            "ks_pvalue": round(float(p_value), 4),
            "psi": round(psi, 4),
            "psi_level": "CRITICO" if psi > 0.25 else ("MODERADO" if psi > 0.10 else "OK"),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def check_alerts(weekly_metrics, baseline, drift_df):
    alerts = []
    for m in weekly_metrics:
        if m["f1"] < THRESHOLDS["f1_min"]:
            alerts.append({"week": m["week"], "level": "CRITICAL",
                           "alert": f"F1 degradado: {m['f1']:.3f} < {THRESHOLDS['f1_min']}",
                           "action": "Reentrenar modelo urgente"})
        if m["roc_auc"] < THRESHOLDS["roc_auc_min"]:
            alerts.append({"week": m["week"], "level": "WARNING",
                           "alert": f"ROC-AUC bajo: {m['roc_auc']:.3f} < {THRESHOLDS['roc_auc_min']}",
                           "action": "Revisar features y datos de entrada"})
        delta = abs(m["actual_churn_rate"] - baseline["actual_churn_rate"])
        if delta > THRESHOLDS["churn_rate_delta"]:
            alerts.append({"week": m["week"], "level": "WARNING",
                           "alert": f"Tasa churn cambio: delta={delta:.3f}",
                           "action": "Analizar cambio de comportamiento de clientes"})
    n_psi_critical = int((drift_df["psi_level"] == "CRITICO").sum())
    if n_psi_critical > 0:
        alerts.append({"week": weekly_metrics[-1]["week"], "level": "CRITICAL",
                       "alert": f"PSI critico en {n_psi_critical} feature(s)",
                       "action": "Revisar pipeline de datos, posible cambio en fuente"})
    return alerts


def push_to_pushgateway(weekly_metrics, drift_df, n_critical, url):
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    for wm in weekly_metrics:
        registry = CollectorRegistry()
        metrics_map = {
            "churn_model_f1_score": wm["f1"],
            "churn_model_roc_auc": wm["roc_auc"],
            "churn_model_gini": wm["gini"],
            "churn_model_accuracy": wm["accuracy"],
            "churn_model_predicted_rate": wm["predicted_churn_rate"],
            "churn_model_actual_rate": wm["actual_churn_rate"],
            "churn_drift_features_count": float((drift_df["psi_level"] != "OK").sum()),
            "churn_drift_psi_max": float(drift_df["psi"].max()),
            "churn_model_critical_alerts": float(n_critical),
        }
        for name, value in metrics_map.items():
            Gauge(name, name, registry=registry).set(value)
        push_to_gateway(url, job="churn_monitor", registry=registry)
    print(f"Pushed {len(weekly_metrics)} weeks -> {url}")


def push_to_cloudwatch(latest, drift_df, n_critical, namespace):
    import boto3

    cw = boto3.client("cloudwatch")
    data = [
        ("F1Score", latest["f1"]),
        ("ROCAUC", latest["roc_auc"]),
        ("Gini", latest["gini"]),
        ("PredictedChurnRate", latest["predicted_churn_rate"]),
        ("ActualChurnRate", latest["actual_churn_rate"]),
        ("PSIMax", float(drift_df["psi"].max())),
        ("DriftFeaturesCount", float((drift_df["psi_level"] != "OK").sum())),
        ("CriticalAlertsCount", float(n_critical)),
    ]
    cw.put_metric_data(
        Namespace=namespace,
        MetricData=[{"MetricName": n, "Value": v} for n, v in data],
    )
    print(f"Published {len(data)} metrics -> CloudWatch {namespace}")


def run(weeks, output, data_dir, pushgateway=None, cloudwatch_namespace=None):
    np.random.seed(99)
    X_ref = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_prod = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_prod = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze("columns")
    model = joblib.load(os.path.join(data_dir, "production_model.pkl"))

    weekly_metrics = []
    X_latest = X_prod
    for week in range(1, weeks + 1):
        drift = 0.0 if week <= weeks // 2 else (week - weeks // 2) * 0.3
        X_week, m = simulate_production_week(X_prod, y_prod, model, week, drift)
        weekly_metrics.append(m)
        X_latest = X_week

    metrics_df = pd.DataFrame(weekly_metrics)
    print(metrics_df[["week", "f1", "roc_auc", "predicted_churn_rate"]].to_string(index=False))

    drift_report = detect_drift(X_ref, X_latest)
    print("\n" + drift_report[["feature", "mean_ref", "mean_actual", "psi", "psi_level"]].to_string(index=False))

    baseline = weekly_metrics[0]
    alerts = check_alerts(weekly_metrics, baseline, drift_report)
    n_critical = sum(1 for a in alerts if a["level"] == "CRITICAL")

    report = {
        "model": "churn-predictor",
        "monitoring_period_weeks": weeks,
        "baseline_metrics": {k: baseline[k] for k in ("f1", "roc_auc", "gini", "accuracy")},
        "latest_metrics": {k: weekly_metrics[-1][k] for k in ("f1", "roc_auc", "gini", "accuracy")},
        "weekly_metrics": weekly_metrics,
        "drift_report": drift_report.to_dict(orient="records"),
        "drift_features_count": int((drift_report["psi_level"] != "OK").sum()),
        "critical_alerts": n_critical,
        "alerts": alerts,
        "recommendation": "REENTRENAR" if n_critical > 0 else "MONITOREAR",
    }

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nRecommendation: {report['recommendation']} | critical alerts: {n_critical}")
    print(f"Report -> {output}")

    if pushgateway:
        try:
            push_to_pushgateway(weekly_metrics, drift_report, n_critical, pushgateway)
        except Exception as exc:
            print(f"Pushgateway skipped: {exc}")
    if cloudwatch_namespace:
        try:
            push_to_cloudwatch(weekly_metrics[-1], drift_report, n_critical, cloudwatch_namespace)
        except Exception as exc:
            print(f"CloudWatch skipped: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor churn model in production")
    parser.add_argument("--weeks", type=int, default=8)
    parser.add_argument("--output", default="reports/monitoring_report.json")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--pushgateway", default=None, help="host:port of Pushgateway")
    parser.add_argument("--cloudwatch-namespace", default=None)
    args = parser.parse_args()
    run(args.weeks, args.output, args.data_dir, args.pushgateway, args.cloudwatch_namespace)
