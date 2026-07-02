# MLOps Pipeline — Churn Prediction (Fintech)

End-to-end MLOps solution for a Telco/Fintech churn use case: reproducible data
prep, experiment tracking, model registry, a FastAPI inference service, drift
monitoring, and AWS deployment automated with GitHub Actions.

## Stack

- **Modeling:** scikit-learn (LogisticRegression, RandomForest), XGBoost
- **Tracking / Registry:** MLflow (file backend)
- **Serving:** FastAPI + Uvicorn, Docker
- **Monitoring:** KS test + PSI, Prometheus + Grafana (local), CloudWatch (AWS)
- **Infra:** Terraform (VPC, ECR, ECS Fargate, ALB, CloudWatch, IAM)
- **CI/CD:** GitHub Actions — 10 chained jobs

## Dataset

`data/telco_customer_churn_mlops.csv` — 300 customers, 9 columns. Target `churn`
is balanced (~50%). 30 missing `internet_service` values imputed as "No".

## Local setup (uv)

```bash
uv venv --python 3.11 .venv
uv pip install -r requirements.txt
source .venv/bin/activate
```

## Pipeline (sequential)

```bash
python src/data_prep.py                       # Phase 1 -> data/ artifacts + scaler
python src/train.py --model all               # Phase 2 -> MLflow runs, best_model
python src/evaluate_and_register.py           # Phase 3 -> None->Staging->Production
uvicorn app.main:app --port 8000              # Phase 4 -> REST API (/docs)
python src/monitor.py --weeks 8               # Phase 5 -> reports/monitoring_report.json
```

MLflow UI: `mlflow ui --backend-store-uri ./mlruns` → http://localhost:5000

## API

| Endpoint         | Method | Description                     |
|------------------|--------|---------------------------------|
| `/health`        | GET    | Service + model status          |
| `/model/info`    | GET    | Production model metadata        |
| `/predict`       | POST   | Individual prediction + risk    |
| `/predict/batch` | POST   | Batch prediction                |

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "gender": "Female", "senior_citizen": 0, "tenure_months": 3,
  "contract_type": "Month-to-Month", "internet_service": "Fiber",
  "monthly_charges": 90.0, "support_tickets_last_6m": 6
}'
```

## Monitoring stack (local)

```bash
cd monitoring && docker compose up -d
# Grafana http://localhost:3000 (admin/admin) — dashboard auto-provisioned
python src/monitor.py --weeks 8 --pushgateway localhost:9091
```

## Docker

```bash
docker build -t churn-predictor:latest .
docker run -p 8000:8000 churn-predictor:latest
```

## AWS deployment

Requires AWS credentials with ECR + ECS + CloudWatch + IAM permissions.

```bash
./scripts/deploy_aws.sh 'YourGrafanaPassword123'
```

The script does a two-phase Terraform apply (ECR first, push image, then the
rest) to avoid the empty-registry race. See `infra/` for the Terraform.

### CI/CD secrets (GitHub → Settings → Secrets → Actions)

| Secret | Used by |
|--------|---------|
| `AWS_ACCESS_KEY_ID` | build-image, deploy-*, terraform, monitor |
| `AWS_SECRET_ACCESS_KEY` | idem |
| `GRAFANA_ADMIN_PASSWORD` | terraform-infra |

## CI/CD jobs

`test → data-prep → train → register → smoke-test → build-image → deploy-(staging|production) → terraform-infra → monitor`

Push to `main` deploys production; push to `develop` deploys staging. The
`monitor` job publishes metrics to CloudWatch and auto-triggers retraining when
the model degrades (recommendation `REENTRENAR`).
