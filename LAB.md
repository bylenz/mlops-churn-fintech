# MLOps Pipeline End-to-End: Predicción de Churn Fintech

## Objetivos

Al completar esta guía serás capaz de:

- Convertir notebooks explorativos en scripts de producción reutilizables y ejecutables desde CI/CD
- Configurar MLflow para registrar experimentos, comparar modelos y versionar el mejor candidato
- Gestionar el ciclo de vida de un modelo en MLflow Model Registry (None → Staging → Production) con promoción automática basada en métricas
- Exponer el modelo como API REST con FastAPI, validar su contrato de entrada/salida y empaquetarlo en Docker
- Implementar un sistema de monitoreo que detecte data drift (KS Test + PSI) y degradación del modelo, con alertas por umbral y visualización en Grafana
- Desplegar la solución completa en AWS con Terraform (VPC, ECR, ECS Fargate, ALB) y automatizar todo el ciclo de vida con GitHub Actions — incluyendo auto-retrain cuando el modelo se degrada

---

## Entregables

| # | Entregable                                | Formato               | Fase |
|---|-------------------------------------------|-----------------------|------|
| 1 | Datasets procesados y scaler              | `data/*.csv`, `scaler.pkl`, `feature_names.csv` | 1 |
| 2 | 3 experimentos trackeados en MLflow       | MLflow runs en `mlruns/` | 2 |
| 3 | Metadata del mejor modelo                 | `data/best_model_meta.json` | 2 |
| 4 | Modelo registrado en Production           | MLflow Model Registry  | 3 |
| 5 | Artefactos de producción                  | `data/production_model.pkl`, `data/production_model_info.json` | 3 |
| 6 | API REST funcional con endpoints `/predict` y `/predict/batch` | FastAPI en `app/main.py` | 4 |
| 7 | Imagen Docker de la API                   | `churn-predictor:latest` | 4 |
| 8 | Reporte de monitoreo (8 semanas simuladas)| `reports/monitoring_report.json` | 5 |
| 9 | Dashboard de métricas en vivo             | Grafana en `localhost:3000` | 5 |
| 10 | Infraestructura en AWS aprovisionada     | Terraform state + recursos AWS (VPC, ECR, ECS, ALB) | 6 |
| 11 | API en producción accesible via ALB      | `http://<alb-dns>/predict` | 6 |
| 12 | Pipeline CI/CD funcional                 | GitHub Actions — 10 jobs encadenados | 6 |
| 13 | Métricas de modelo en CloudWatch         | Namespace `MLOps/ChurnPredictor` | 6 |

---

## Tabla de Responsabilidades

| Entregable                         | Responsable              |
| ---------------------------------- | ------------------------ |
| Notebooks de exploración y EDA     | Data Scientist           |
| Scripts modulares (`src/`)         | Ingeniero de MLOps       |
| Tracking con MLflow                | Data Scientist / MLOps   |
| Model Registry y promoción         | Ingeniero de MLOps       |
| API REST (FastAPI) + Docker        | Ingeniero de MLOps       |
| Sistema de monitoreo (Prometheus)  | Ingeniero de MLOps       |
| Dashboard de métricas (Grafana)    | Ingeniero de MLOps       |

---

## Contexto del Proyecto

**Caso:** Predicción de churn (abandono) para una Fintech  
**Dataset:** Telco Customer Churn (`telco_customer_churn_mlops.csv`) — 300 clientes, 9 columnas  
**Objetivo de negocio:** Identificar clientes con alta probabilidad de abandonar para actuar con retención proactiva  

### Estructura del Proyecto

```
mlops-project/
├── data/                        # Datasets y artefactos generados
├── notebooks/                   # Exploración interactiva (fases 1-5)
│   ├── 01_eda_data_preparation.ipynb
│   ├── 02_training_mlflow.ipynb
│   ├── 03_model_registry.ipynb
│   ├── 04_deployment_inference.ipynb
│   └── 05_monitoring.ipynb
├── src/                         # Scripts de producción
│   ├── data_prep.py
│   ├── train.py
│   ├── evaluate_and_register.py
│   └── monitor.py
├── app/
│   └── main.py                  # FastAPI API
├── monitoring/
│   ├── docker-compose.yml       # Prometheus + Grafana + Pushgateway
│   ├── prometheus/
│   └── grafana/
├── Dockerfile
└── requirements.txt
```

### Flujo Completo

```
Raw Data
    │
    ▼ Fase 1: EDA + Preparación
[01_eda_data_preparation] → X_train.csv, X_test.csv, scaler.pkl
    │
    ▼ Fase 2: Entrenamiento + Tracking
[02_training_mlflow] → MLflow runs (LR, RF, XGBoost) → best_model.pkl
    │
    ▼ Fase 3: Model Registry
[03_model_registry] → None → Staging → Production (MLflow Registry)
    │
    ▼ Fase 4: Deployment + Inferencia
[04_deployment_inference] → FastAPI /predict → Docker
    │
    ▼ Fase 5: Monitoreo
[05_monitoring] → KS Test / PSI + alertas → Prometheus + Grafana
```

---

## Fase 1: EDA y Preparación de Datos

**Notebook:** `notebooks/01_eda_data_preparation.ipynb`  
**Script:** `src/data_prep.py`

### Objetivo

Explorar el dataset, identificar features relevantes, aplicar preprocesamiento reproducible y guardar artefactos versionados.

### Paso 1.1 — Setup e Inspección General

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = '../data'

# Cargar dataset
df_raw = pd.read_csv(os.path.join(DATA_DIR, 'telco_customer_churn_mlops.csv'))

# Normalizar nombres de columna al estándar del pipeline
df = df_raw.rename(columns={
    'customer_id':              'customerID',
    'senior_citizen':           'SeniorCitizen',
    'tenure_months':            'tenure',
    'contract_type':            'Contract',
    'internet_service':         'InternetService',
    'monthly_charges':          'MonthlyCharges',
    'support_tickets_last_6m':  'support_tickets',
    'churn':                    'Churn',
})

print(df.shape)         # (300, 9)
print(df.dtypes)
print(df.isnull().sum())  # 30 NaN en InternetService
```

### Paso 1.2 — Análisis de la Variable Objetivo

```python
churn_counts = df['Churn'].value_counts()
churn_rate = churn_counts['Yes'] / len(df) * 100
print(f'Churn rate: {churn_rate:.1f}%')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].bar(churn_counts.index, churn_counts.values, color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Distribución de Churn')
plt.tight_layout()
plt.show()
```

### Paso 1.3 — Análisis por Feature

```python
# Features numéricas vs Churn
num_cols = ['tenure', 'MonthlyCharges', 'support_tickets']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(num_cols):
    df[col].hist(ax=axes[i], bins=20, color='steelblue', edgecolor='white')
    axes[i].set_title(f'Distribución: {col}')
plt.tight_layout()
plt.show()

# Análisis por tipo de contrato
ct = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack() * 100
ct.plot(kind='bar', color=['#2ecc71', '#e74c3c'], rot=15)
plt.title('Churn por Tipo de Contrato')
plt.show()
```

### Paso 1.4 — Preprocesamiento

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

df_proc = df.copy()
df_proc = df_proc.drop(columns=['customerID', 'tenure_bin'], errors='ignore')

# 30 NaN en InternetService: asumidos como "sin servicio"
df_proc['InternetService'] = df_proc['InternetService'].fillna('No')

# Target: Yes/No → 1/0
df_proc['Churn'] = (df_proc['Churn'] == 'Yes').astype(int)

# Encoding binario para género
df_proc['gender'] = LabelEncoder().fit_transform(df_proc['gender'])

# One-hot para categóricas restantes
cat_cols = df_proc.select_dtypes(include='object').columns.tolist()
df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=True)

# Split
X = df_proc.drop(columns=['Churn'])
y = df_proc['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling de features numéricas
scaler = StandardScaler()
num_features = ['tenure', 'MonthlyCharges', 'support_tickets']
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

print(f'Train: {X_train.shape} | Test: {X_test.shape}')
```

### Paso 1.5 — Guardar Artefactos

```python
X_train.to_csv(os.path.join(DATA_DIR, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(DATA_DIR, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(DATA_DIR, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(DATA_DIR, 'y_test.csv'), index=False)
joblib.dump(scaler, os.path.join(DATA_DIR, 'scaler.pkl'))
pd.Series(X_train.columns.tolist()).to_csv(
    os.path.join(DATA_DIR, 'feature_names.csv'), index=False, header=False
)
print('Artefactos guardados.')
```

### Ejecutar como Script

```bash
python src/data_prep.py
```

El script `src/data_prep.py` encapsula todo el preprocesamiento anterior. Produce los mismos artefactos y es invocable desde GitHub Actions o cualquier orquestador.

### Job en CI/CD (`data-prep`)

Este script es el corazón del job `data-prep` en `.github/workflows/mlops.yml`. Se ejecuta después del quality gate y pasa sus artefactos al job de entrenamiento via `actions/upload-artifact`.

```yaml
# .github/workflows/mlops.yml — job: data-prep
data-prep:
  name: Data Preparation
  runs-on: ubuntu-latest
  needs: test                        # espera que pasen los tests
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.10'
        cache: 'pip'
    - name: Instalar dependencias
      run: pip install -r requirements.txt
    - name: Preparar datos
      run: python src/data_prep.py    # ← el script de esta fase
    - name: Subir artefactos de datos
      uses: actions/upload-artifact@v4
      with:
        name: data-prep-artifacts
        path: data/                   # X_train.csv, X_test.csv, scaler.pkl, feature_names.csv
```

### Resumen Fase 1

| Paso         | Acción                                  | Resultado                         |
|--------------|---------------------------------------- |-----------------------------------|
| Carga        | Dataset `telco_customer_churn_mlops.csv`| 300 filas, 9 columnas             |
| Imputación   | 30 NaN en InternetService → `'No'`      | Asunción documentada              |
| Encoding     | Binary (gender) + One-hot (Contract…)  | Features numéricas                |
| Split        | 80/20 estratificado                     | Train=240 | Test=60               |
| Scaling      | StandardScaler en 3 features numéricas  | `scaler.pkl` guardado             |

---

## Fase 2: Entrenamiento y Tracking con MLflow

**Notebook:** `notebooks/02_training_mlflow.ipynb`  
**Script:** `src/train.py`

### Objetivo

Entrenar tres modelos (Logistic Regression, Random Forest, XGBoost), registrar parámetros y métricas en MLflow, comparar y seleccionar el mejor por F1.

### Paso 2.1 — Configurar MLflow

```python
import mlflow
import mlflow.sklearn
import os

MLFLOW_TRACKING_URI = '../mlruns'
EXPERIMENT_NAME = 'churn-prediction-fintech'

mlflow.set_tracking_uri(f'file://{os.path.abspath(MLFLOW_TRACKING_URI)}')
mlflow.set_experiment(EXPERIMENT_NAME)

exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print(f'Experiment ID: {exp.experiment_id}')
print(f'Tracking URI: {mlflow.get_tracking_uri()}')
```

### Paso 2.2 — Cargar Datos

```python
import pandas as pd

X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
X_test  = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze()

print(f'Train: {X_train.shape} | Test: {X_test.shape}')
print(f'Churn rate train: {y_train.mean():.3f}')
```

### Paso 2.3 — Función de Entrenamiento con Tracking

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

def train_and_log(model, model_name, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tags({'model_type': model_name, 'dataset': 'telco_churn'})
        mlflow.log_params(params)
        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size', len(X_test))

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy':  round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall':    round(recall_score(y_test, y_pred, zero_division=0), 4),
            'f1':        round(f1_score(y_test, y_pred, zero_division=0), 4),
            'roc_auc':   round(roc_auc_score(y_test, y_prob), 4),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, 'model', input_example=X_train.head(3))

        return run.info.run_id, metrics, model
```

### Paso 2.4 — Experimento 1: Logistic Regression (Baseline)

```python
from sklearn.linear_model import LogisticRegression

lr_params = {'C': 1.0, 'max_iter': 500, 'class_weight': 'balanced', 'solver': 'lbfgs', 'random_state': 42}
lr_model = LogisticRegression(**lr_params)
lr_run_id, lr_metrics, lr_fitted = train_and_log(
    lr_model, 'LogisticRegression', lr_params, X_train, X_test, y_train, y_test
)
print(f'LR → F1={lr_metrics["f1"]} | ROC-AUC={lr_metrics["roc_auc"]}')
```

### Paso 2.5 — Experimento 2: Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5,
             'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}
rf_model = RandomForestClassifier(**rf_params)
rf_run_id, rf_metrics, rf_fitted = train_and_log(
    rf_model, 'RandomForest', rf_params, X_train, X_test, y_train, y_test
)
print(f'RF  → F1={rf_metrics["f1"]} | ROC-AUC={rf_metrics["roc_auc"]}')
```

### Paso 2.6 — Experimento 3: XGBoost

```python
from xgboost import XGBClassifier

# Peso de clase positiva para manejar desbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f'scale_pos_weight: {scale_pos_weight:.2f}')

xgb_params = {
    'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'scale_pos_weight': round(scale_pos_weight, 2),
    'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0,
}
xgb_model = XGBClassifier(**xgb_params)
xgb_run_id, xgb_metrics, xgb_fitted = train_and_log(
    xgb_model, 'XGBoost', xgb_params, X_train, X_test, y_train, y_test
)
print(f'XGB → F1={xgb_metrics["f1"]} | ROC-AUC={xgb_metrics["roc_auc"]}')
```

### Paso 2.7 — Comparación y Selección

```python
import joblib
import json

results = pd.DataFrame({
    'Logistic Regression': lr_metrics,
    'Random Forest':       rf_metrics,
    'XGBoost':             xgb_metrics,
}).T.round(4)

print(results[['accuracy', 'f1', 'roc_auc']])

# Selección por F1 (más relevante en dataset desbalanceado)
best_name = results['f1'].idxmax()
model_map = {
    'Logistic Regression': (lr_fitted, lr_run_id),
    'Random Forest':       (rf_fitted, rf_run_id),
    'XGBoost':             (xgb_fitted, xgb_run_id),
}
best_model, best_run_id = model_map[best_name]
best_metrics = results.loc[best_name].to_dict()

print(f'\nMejor modelo: {best_name}')
print(f'  F1={best_metrics["f1"]} | ROC-AUC={best_metrics["roc_auc"]}')

# Guardar modelo y metadata
joblib.dump(best_model, os.path.join(DATA_DIR, 'best_model.pkl'))
with open(os.path.join(DATA_DIR, 'best_model_meta.json'), 'w') as f:
    json.dump({'best_model': best_name, 'best_run_id': best_run_id, 'metrics': best_metrics}, f, indent=2)
```

### Paso 2.8 — Ver Resultados en MLflow UI

```bash
cd notebooks
mlflow ui --backend-store-uri ../mlruns
# Abrir http://localhost:5000
```

### Ejecutar como Script

```bash
# Entrenar los 3 modelos y seleccionar el mejor
python src/train.py --model all --experiment churn-prediction-fintech

# Entrenar solo XGBoost
python src/train.py --model xgboost
```

### Job en CI/CD (`train`)

Descarga los artefactos de datos del job anterior, ejecuta el entrenamiento y sube MLflow runs + modelo al siguiente job. El tipo de modelo puede elegirse via `workflow_dispatch` input.

```yaml
# .github/workflows/mlops.yml — job: train
train:
  name: Modeling
  runs-on: ubuntu-latest
  needs: data-prep
  timeout-minutes: 20
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.10'
        cache: 'pip'
    - name: Instalar dependencias
      run: pip install -r requirements.txt
    - name: Descargar artefactos de datos
      uses: actions/download-artifact@v4
      with:
        name: data-prep-artifacts
        path: data/
    - name: Entrenar modelo
      run: |
        python src/train.py \
          --model ${{ github.event.inputs.model_type || 'all' }} \
          --experiment ${{ env.EXPERIMENT_NAME }}
    - name: Subir artefactos de MLflow
      uses: actions/upload-artifact@v4
      with:
        name: train-artifacts
        path: |
          mlruns/             # runs de MLflow con métricas y parámetros
          data/               # best_model.pkl, best_model_meta.json
```

> El input `model_type` se expone via `workflow_dispatch` para poder disparar el pipeline manualmente con un modelo específico (`logistic`, `random_forest`, `xgboost` o `all`).

### Resumen Fase 2

| Modelo               | Accuracy | F1     | ROC-AUC |
|----------------------|----------|--------|---------|
| Logistic Regression  | ~0.78    | ~0.65  | ~0.82   |
| Random Forest        | ~0.80    | ~0.67  | ~0.84   |
| XGBoost              | ~0.81    | ~0.69  | ~0.86   |

> **Criterio de selección:** F1-score — equilibra precision y recall en datasets desbalanceados (26% churn).

---

## Fase 3: Model Registry y Versionamiento

**Notebook:** `notebooks/03_model_registry.ipynb`  
**Script:** `src/evaluate_and_register.py`

### Objetivo

Registrar el mejor modelo en MLflow Model Registry, gestionar el ciclo de vida de versiones (None → Staging → Production) con validación automática.

### Paso 3.1 — Conectar Cliente MLflow

```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import json, os

REGISTERED_MODEL_NAME = 'churn-predictor'
PRODUCTION_F1_THRESHOLD = 0.50

mlflow.set_tracking_uri(f'file://{os.path.abspath("../mlruns")}')
client = MlflowClient()
print(f'Conectado: {mlflow.get_tracking_uri()}')
```

### Paso 3.2 — Recuperar Mejor Run

```python
with open(os.path.join(DATA_DIR, 'best_model_meta.json')) as f:
    meta = json.load(f)

best_run_id = meta['best_run_id']
best_model_name = meta['best_model']

print(f'Mejor modelo: {best_model_name}')
print(f'Run ID: {best_run_id}')
print(f'Métricas: {meta["metrics"]}')
```

### Paso 3.3 — Listar Todos los Runs del Experimento

```python
exp = mlflow.get_experiment_by_name('churn-prediction-fintech')

all_runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=['metrics.roc_auc DESC'],
)

display_cols = ['run_id', 'tags.model_type', 'metrics.accuracy', 'metrics.f1', 'metrics.roc_auc']
available = [c for c in display_cols if c in all_runs.columns]
print(all_runs[available].to_string(index=False))
```

### Paso 3.4 — Registrar Modelo en Registry

```python
model_uri = f'runs:/{best_run_id}/model'
print(f'Registrando: {model_uri}')

mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
print(f'Nombre:  {mv.name}')
print(f'Versión: {mv.version}')
print(f'Estado:  {mv.current_stage}')
```

### Paso 3.5 — Agregar Descripción y Tags

```python
client.update_registered_model(
    name=REGISTERED_MODEL_NAME,
    description='Modelo de predicción de churn para clientes Fintech. Dataset Telco Customer Churn.'
)

client.update_model_version(
    name=REGISTERED_MODEL_NAME,
    version=mv.version,
    description=f'Entrenado con {best_model_name}. F1={meta["metrics"]["f1"]:.4f} | ROC-AUC={meta["metrics"]["roc_auc"]:.4f}'
)

client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, 'dataset', 'telco_churn')
client.set_model_version_tag(REGISTERED_MODEL_NAME, mv.version, 'model_type', best_model_name)
```

### Paso 3.6 — Ciclo de Vida: None → Staging → Production

```python
# 1. Mover a Staging
client.transition_model_version_stage(
    name=REGISTERED_MODEL_NAME, version=mv.version,
    stage='Staging', archive_existing_versions=False,
)
print(f'Versión {mv.version} → Staging')

# 2. Validar en Staging
X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze()

staging_model = mlflow.sklearn.load_model(f'models:/{REGISTERED_MODEL_NAME}/Staging')
y_pred_staging = staging_model.predict(X_test)
staging_f1 = f1_score(y_test, y_pred_staging, zero_division=0)
print(f'Staging F1: {staging_f1:.4f} | Threshold: {PRODUCTION_F1_THRESHOLD}')

# 3. Promover a Production si pasa validación
if staging_f1 >= PRODUCTION_F1_THRESHOLD:
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME, version=mv.version,
        stage='Production', archive_existing_versions=True,  # archiva versiones previas
    )
    print(f'Versión {mv.version} → Production')
else:
    print(f'Validación fallida. F1={staging_f1:.4f} < {PRODUCTION_F1_THRESHOLD}')
```

### Paso 3.7 — Exportar para Deployment

```python
import joblib

prod_model = mlflow.sklearn.load_model(f'models:/{REGISTERED_MODEL_NAME}/Production')
joblib.dump(prod_model, os.path.join(DATA_DIR, 'production_model.pkl'))

prod_info = {
    'model_name': REGISTERED_MODEL_NAME,
    'version': str(mv.version),
    'stage': 'Production',
    'run_id': best_run_id,
    'model_type': best_model_name,
    'model_uri': f'models:/{REGISTERED_MODEL_NAME}/Production',
}
with open(os.path.join(DATA_DIR, 'production_model_info.json'), 'w') as f:
    json.dump(prod_info, f, indent=2)

print('production_model.pkl y production_model_info.json guardados.')
```

### Ejecutar como Script

```bash
python src/evaluate_and_register.py
```

### Job en CI/CD (`register`)

Toma los artefactos de MLflow del job `train`, registra el mejor modelo en el Registry, lo valida en Staging y lo promueve a Production. El job publica un resumen de métricas en el Step Summary de GitHub Actions.

```yaml
# .github/workflows/mlops.yml — job: register
register:
  name: Evaluation
  runs-on: ubuntu-latest
  needs: train
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.10'
        cache: 'pip'
    - name: Instalar dependencias
      run: pip install -r requirements.txt
    - name: Descargar artefactos de datos
      uses: actions/download-artifact@v4
      with:
        name: data-prep-artifacts
        path: data/
    - name: Descargar artefactos de MLflow
      uses: actions/download-artifact@v4
      with:
        name: train-artifacts
        path: .
    - name: Evaluar y registrar modelo
      run: python src/evaluate_and_register.py  # ← None → Staging → Production
    - name: Subir modelo de producción
      uses: actions/upload-artifact@v4
      with:
        name: production-artifacts
        retention-days: 30
        path: data/   # production_model.pkl, production_model_info.json
    - name: Resumen de métricas
      run: |
        if [ -f data/best_model_meta.json ]; then
          echo "## Métricas del Modelo" >> $GITHUB_STEP_SUMMARY
          echo '```json' >> $GITHUB_STEP_SUMMARY
          cat data/best_model_meta.json >> $GITHUB_STEP_SUMMARY
          echo '```' >> $GITHUB_STEP_SUMMARY
        fi
```

### Flujo del Registry

```
Training Run
    │
    ▼
register_model() → Versión N (None)
    │
    ▼
transition → Staging
    │
    ▼
Validación automática (F1 >= 0.50)
    │ OK          │ FAIL
    ▼             ▼
Production     SystemExit(1)
    │
    ▼
production_model.pkl → API / Batch
```

---

## Fase 4: Deployment e Inferencia

**Notebook:** `notebooks/04_deployment_inference.ipynb`  
**Código:** `app/main.py` + `Dockerfile`

### Objetivo

Exponer el modelo como API REST con FastAPI: predicción individual, predicción batch, y empaquetado con Docker.

### Paso 4.1 — Estructura de la API (`app/main.py`)

La API tiene tres grupos de endpoints:

```
GET  /          → info del servicio
GET  /health    → health check
GET  /model/info → metadata del modelo en producción
POST /predict   → predicción individual (CustomerFeatures → PredictionResponse)
POST /predict/batch → predicción múltiple
```

**Schema de entrada (`CustomerFeatures`):**

```python
class CustomerFeatures(BaseModel):
    gender: str                     # "Male" | "Female"
    senior_citizen: int             # 0 | 1
    tenure_months: int              # meses como cliente
    contract_type: str              # "Month-to-Month" | "One Year" | "Two Year"
    internet_service: Optional[str] # "Fiber" | "DSL" | None
    monthly_charges: float          # cargo mensual
    support_tickets_last_6m: int    # tickets de soporte últimos 6 meses
```

**Schema de salida (`PredictionResponse`):**

```python
class PredictionResponse(BaseModel):
    customer_id: Optional[str]
    churn_prediction: int       # 0 | 1
    churn_label: str            # "Churn" | "No Churn"
    churn_probability: float    # 0.0 – 1.0
    risk_level: str             # "BAJO" | "MEDIO" | "ALTO"
    recommendation: str         # acción recomendada
    model_version: str
```

### Paso 4.2 — Preprocesamiento en la API

El preprocessing en la API replica exactamente el pipeline de entrenamiento:

```python
def preprocess_customer(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([{
        'gender':          1 if data['gender'] == 'Male' else 0,
        'SeniorCitizen':   data['senior_citizen'],
        'tenure':          data['tenure_months'],
        'MonthlyCharges':  data['monthly_charges'],
        'support_tickets': data['support_tickets_last_6m'],
        'Contract':        data['contract_type'],
        'InternetService': data['internet_service'] or 'No',
    }])

    # One-hot igual que en entrenamiento
    df = pd.get_dummies(df, columns=['Contract', 'InternetService'], drop_first=True)

    # Alinear con feature set de entrenamiento
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0
    df = df[feature_names]

    # Scaling
    df[['tenure', 'MonthlyCharges', 'support_tickets']] = scaler.transform(
        df[['tenure', 'MonthlyCharges', 'support_tickets']]
    )
    return df
```

### Paso 4.3 — Probar Predicción Individual

```python
# Sin API: predicción directa
model = joblib.load(os.path.join(DATA_DIR, 'production_model.pkl'))
scaler = joblib.load(os.path.join(DATA_DIR, 'scaler.pkl'))
feature_names = pd.read_csv(os.path.join(DATA_DIR, 'feature_names.csv'), header=None)[0].tolist()

# Cliente de ALTO RIESGO
high_risk = {
    'gender': 'Female', 'senior_citizen': 0, 'tenure_months': 3,
    'contract_type': 'Month-to-Month', 'internet_service': 'Fiber optic',
    'monthly_charges': 90.0, 'support_tickets_last_6m': 6,
}

X = preprocess_customer(high_risk)
prob = float(model.predict_proba(X)[0][1])
pred = int(model.predict(X)[0])
print(f'Predicción: {"Churn" if pred else "No Churn"} | Probabilidad: {prob:.4f}')

# Cliente de BAJO RIESGO
low_risk = {
    'gender': 'Male', 'senior_citizen': 0, 'tenure_months': 60,
    'contract_type': 'Two year', 'internet_service': 'DSL',
    'monthly_charges': 45.0, 'support_tickets_last_6m': 0,
}
X2 = preprocess_customer(low_risk)
prob2 = float(model.predict_proba(X2)[0][1])
print(f'Predicción: {"Churn" if model.predict(X2)[0] else "No Churn"} | Probabilidad: {prob2:.4f}')
```

### Paso 4.4 — Inferencia Batch

```python
X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze()

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

def get_risk_level(prob):
    return 'ALTO' if prob >= 0.6 else ('MEDIO' if prob >= 0.3 else 'BAJO')

batch_results = pd.DataFrame({
    'customer_id':       [f'CUST-{i:05d}' for i in range(len(X_test))],
    'churn_prediction':  y_pred,
    'churn_label':       ['Churn' if p == 1 else 'No Churn' for p in y_pred],
    'churn_probability': y_prob.round(4),
    'risk_level':        [get_risk_level(p) for p in y_prob],
})

print(f'Total: {len(batch_results)} clientes')
print(f'Churn predicho: {batch_results["churn_prediction"].sum()}')
print(f'Tasa predicha: {batch_results["churn_prediction"].mean():.2%}')
print(batch_results['risk_level'].value_counts())

batch_results.to_csv(os.path.join(DATA_DIR, 'batch_predictions.csv'), index=False)
```

### Paso 4.5 — Levantar la API Localmente

```bash
# Desde la raíz del proyecto
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Paso 4.6 — Probar Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Predicción individual
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "senior_citizen": 0,
    "tenure_months": 3,
    "contract_type": "Month-to-Month",
    "internet_service": "Fiber optic",
    "monthly_charges": 90.0,
    "support_tickets_last_6m": 6
  }'

# Documentación interactiva
# http://localhost:8000/docs
```

### Paso 4.7 — Dockerizar la API

**`Dockerfile`:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/production_model.pkl       ./data/production_model.pkl
COPY data/production_model_info.json ./data/production_model_info.json
COPY data/scaler.pkl                 ./data/scaler.pkl
COPY data/feature_names.csv          ./data/feature_names.csv

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build
docker build -t churn-predictor:latest .

# Run
docker run -p 8000:8000 churn-predictor:latest

# Verificar
curl http://localhost:8000/health
```

### Jobs en CI/CD (`smoke-test` + `build-image`)

Dos jobs encadenados: el primero verifica que la API arranca correctamente con el modelo (sin AWS), el segundo construye la imagen Docker y la sube a ECR con el SHA del commit como tag.

```yaml
# .github/workflows/mlops.yml — job: smoke-test
smoke-test:
  name: Smoke Test
  runs-on: ubuntu-latest
  needs: register
  if: github.ref == 'refs/heads/main' || github.ref == 'refs/heads/develop'
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.10'
        cache: 'pip'
    - name: Instalar dependencias
      run: pip install -r requirements.txt
    - name: Descargar modelo de producción
      uses: actions/download-artifact@v4
      with:
        name: production-artifacts
        path: data/
    - name: Verificar que la API arranca con el modelo
      run: |
        cd app
        timeout 15 uvicorn main:app --host 0.0.0.0 --port 8000 &
        sleep 8
        curl -f http://localhost:8000/health    || exit 1
        curl -f http://localhost:8000/model/info || exit 1
        echo "Smoke test OK"

# .github/workflows/mlops.yml — job: build-image
build-image:
  name: Build Image → ECR
  runs-on: ubuntu-latest
  needs: smoke-test
  outputs:
    image: ${{ steps.push.outputs.image }}   # imagen con SHA, usada por deploy jobs
  steps:
    - uses: actions/checkout@v4
    - name: Descargar modelo de producción
      uses: actions/download-artifact@v4
      with:
        name: production-artifacts
        path: data/
    - name: Descargar feature_names y scaler
      uses: actions/download-artifact@v4
      with:
        name: data-prep-artifacts
        path: data/
    - name: Verificar artefactos antes del build
      run: |
        for f in data/production_model.pkl data/production_model_info.json \
                 data/scaler.pkl data/feature_names.csv; do
          test -f "$f" && echo "OK  $f" || { echo "MISSING  $f"; exit 1; }
        done
    - name: Configurar credenciales AWS
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}       # ← secret de repo
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }} # ← secret de repo
        aws-region: ${{ env.AWS_REGION }}
    - name: Login a Amazon ECR
      id: ecr-login
      uses: aws-actions/amazon-ecr-login@v2
    - name: Build, tag y push imagen
      id: push
      env:
        REGISTRY: ${{ steps.ecr-login.outputs.registry }}
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -t $REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        echo "image=$REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT
```

### Resumen Fase 4

| Endpoint            | Método | Descripción                            |
|---------------------|--------|----------------------------------------|
| `/health`           | GET    | Estado del servicio y modelo cargado   |
| `/model/info`       | GET    | Versión, tipo y run_id del modelo      |
| `/predict`          | POST   | Predicción individual con recomendación|
| `/predict/batch`    | POST   | Predicción para lista de clientes      |

---

## Fase 5: Monitoreo del Modelo en Producción

**Notebook:** `notebooks/05_monitoring.ipynb`  
**Script:** `src/monitor.py`

### Objetivo

Detectar data drift (distribución de entrada), monitorear degradación del modelo en el tiempo (model drift), emitir alertas y visualizar en Grafana.

### Paso 5.1 — Cargar Datos de Referencia y Producción

```python
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Referencia = distribución de entrenamiento
X_ref  = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
y_ref  = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).squeeze()

# Producción = test set como proxy (en prod: logs de inferencia real)
X_prod = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
y_prod = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze()

model = joblib.load(os.path.join(DATA_DIR, 'production_model.pkl'))
print(f'Referencia: {X_ref.shape} | Producción: {X_prod.shape}')
```

### Paso 5.2 — Simular Producción con Drift Gradual

```python
np.random.seed(99)

def simulate_production_week(X_base, y_base, model, week, drift_factor=0.0):
    """Simula drift gradual introduciendo ruido en features numéricas."""
    X_week = X_base.copy()
    if drift_factor > 0:
        for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            if col in X_week.columns:
                X_week[col] += np.random.normal(drift_factor, 0.1, len(X_week))

    y_pred = model.predict(X_week)
    y_prob = model.predict_proba(X_week)[:, 1]

    # Ruido en labels (simula cambio de comportamiento de mercado)
    label_noise = np.random.random(len(X_base)) < (0.05 * week * drift_factor)
    y_actual = y_base.values.copy()
    y_actual[label_noise] = 1 - y_actual[label_noise]

    return X_week, {
        'week': week,
        'accuracy':             accuracy_score(y_actual, y_pred),
        'f1':                   f1_score(y_actual, y_pred, zero_division=0),
        'roc_auc':              roc_auc_score(y_actual, y_prob),
        'gini':                 2 * roc_auc_score(y_actual, y_prob) - 1,
        'predicted_churn_rate': float(y_pred.mean()),
        'actual_churn_rate':    float(y_actual.mean()),
    }

# 8 semanas: primeras 4 sin drift, últimas 4 con drift creciente
weekly_metrics = []
X_latest = X_prod
for week in range(1, 9):
    drift = 0.0 if week <= 4 else (week - 4) * 0.3
    X_week, metrics = simulate_production_week(X_prod, y_prod, model, week, drift)
    weekly_metrics.append(metrics)
    X_latest = X_week

metrics_df = pd.DataFrame(weekly_metrics)
print(metrics_df[['week', 'f1', 'roc_auc', 'predicted_churn_rate']].to_string(index=False))
```

### Paso 5.3 — Visualizar Métricas en el Tiempo

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
weeks = metrics_df['week']

# Métricas de clasificación
for metric, color in [('f1', 'steelblue'), ('roc_auc', 'darkorange'), ('accuracy', 'green')]:
    axes[0, 0].plot(weeks, metrics_df[metric], marker='o', label=metric.upper(), color=color)
axes[0, 0].axvline(4.5, color='red', linestyle='--', alpha=0.7, label='Drift inicio')
axes[0, 0].axhline(0.65, color='orange', linestyle=':', alpha=0.7, label='Threshold F1')
axes[0, 0].set_title('Métricas de Clasificación por Semana')
axes[0, 0].legend()

# Churn rate
axes[0, 1].plot(weeks, metrics_df['actual_churn_rate'], marker='s', color='red', label='Real')
axes[0, 1].plot(weeks, metrics_df['predicted_churn_rate'], marker='o', color='blue', label='Predicha')
axes[0, 1].set_title('Tasa de Churn: Real vs Predicha')
axes[0, 1].legend()

plt.tight_layout()
plt.show()
```

### Paso 5.4 — Detección de Data Drift (KS Test + PSI)

```python
def calculate_psi(expected, actual, bins=10):
    """PSI < 0.10: sin cambio | 0.10-0.25: moderado | > 0.25: significativo."""
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
            'feature':    col,
            'mean_ref':   round(float(ref_vals.mean()), 4),
            'mean_actual':round(float(cur_vals.mean()), 4),
            'ks_pvalue':  round(float(p_value), 4),
            'psi':        round(psi, 4),
            'psi_level':  'CRITICO' if psi > 0.25 else ('MODERADO' if psi > 0.10 else 'OK'),
        })
    return pd.DataFrame(rows).sort_values('psi', ascending=False)

drift_report = detect_drift(X_ref, X_latest)
print(drift_report[['feature', 'mean_ref', 'mean_actual', 'psi', 'psi_level']].to_string(index=False))
print(f"\nFeatures con drift: {(drift_report['psi_level'] != 'OK').sum()}")
```

### Paso 5.5 — Sistema de Alertas

```python
THRESHOLDS = {
    'f1_min':             0.60,
    'roc_auc_min':        0.65,
    'churn_rate_delta':   0.10,
    'psi_critical':       0.25,
}

def check_alerts(weekly_metrics, baseline, drift_df):
    alerts = []
    for m in weekly_metrics:
        if m['f1'] < THRESHOLDS['f1_min']:
            alerts.append({
                'week': m['week'], 'level': 'CRITICAL',
                'alert': f'F1 degradado: {m["f1"]:.3f} < {THRESHOLDS["f1_min"]}',
                'action': 'Reentrenar modelo urgente',
            })
        if m['roc_auc'] < THRESHOLDS['roc_auc_min']:
            alerts.append({
                'week': m['week'], 'level': 'WARNING',
                'alert': f'ROC-AUC bajo: {m["roc_auc"]:.3f} < {THRESHOLDS["roc_auc_min"]}',
                'action': 'Revisar features y datos de entrada',
            })
        delta = abs(m['actual_churn_rate'] - baseline['actual_churn_rate'])
        if delta > THRESHOLDS['churn_rate_delta']:
            alerts.append({
                'week': m['week'], 'level': 'WARNING',
                'alert': f'Tasa churn cambió: delta={delta:.3f}',
                'action': 'Analizar cambio de comportamiento de clientes',
            })
    n_psi_critical = int((drift_df['psi_level'] == 'CRITICO').sum())
    if n_psi_critical > 0:
        alerts.append({
            'week': weekly_metrics[-1]['week'], 'level': 'CRITICAL',
            'alert': f'PSI crítico en {n_psi_critical} feature(s)',
            'action': 'Revisar pipeline de datos, posible cambio en fuente',
        })
    return alerts

baseline = weekly_metrics[0]
alerts = check_alerts(weekly_metrics, baseline, drift_report)

for alert in alerts:
    icon = '🔴' if alert['level'] == 'CRITICAL' else '🟡'
    print(f'{icon} Semana {alert["week"]}: {alert["alert"]}')
    print(f'   → {alert["action"]}')
```

### Paso 5.6 — Reporte JSON

```python
import json

report = {
    'model': 'churn-predictor',
    'monitoring_period_weeks': 8,
    'baseline_metrics': {k: round(baseline[k], 4) for k in ('f1', 'roc_auc', 'gini', 'accuracy')},
    'latest_metrics':   {k: round(weekly_metrics[-1][k], 4) for k in ('f1', 'roc_auc', 'gini', 'accuracy')},
    'drift_report':     drift_report.to_dict(orient='records'),
    'drift_features_count': int((drift_report['psi_level'] != 'OK').sum()),
    'critical_alerts':  sum(1 for a in alerts if a['level'] == 'CRITICAL'),
    'alerts':           alerts,
    'recommendation':   'REENTRENAR' if any(a['level'] == 'CRITICAL' for a in alerts) else 'MONITOREAR',
}

os.makedirs('../reports', exist_ok=True)
with open('../reports/monitoring_report.json', 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f'Recomendación: {report["recommendation"]}')
print(f'Alertas críticas: {report["critical_alerts"]}')
```

### Paso 5.7 — Stack de Monitoreo Local (Prometheus + Grafana)

```bash
# Levantar el stack
cd monitoring
docker compose up -d

# Puertos:
# Pushgateway → http://localhost:9091
# Prometheus  → http://localhost:9090
# Grafana     → http://localhost:3000 (admin/admin)
```

**Configuración `monitoring/docker-compose.yml`:**

```yaml
name: churn-monitoring-local

services:
  pushgateway:
    image: prom/pushgateway:v1.9.0
    ports:
      - "9091:9091"

  prometheus:
    image: prom/prometheus:v2.51.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    depends_on:
      - pushgateway

  grafana:
    image: grafana/grafana-oss:11.3.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
    depends_on:
      - prometheus
```

### Paso 5.8 — Push de Métricas al Pushgateway

```python
import time
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

PUSHGATEWAY_URL = 'localhost:9091'

def push_week_metrics(week_metrics, drift_df, n_critical_alerts):
    registry = CollectorRegistry()

    metrics_map = {
        'churn_model_f1_score':          week_metrics['f1'],
        'churn_model_roc_auc':           week_metrics['roc_auc'],
        'churn_model_gini':              week_metrics['gini'],
        'churn_model_accuracy':          week_metrics['accuracy'],
        'churn_model_predicted_rate':    week_metrics['predicted_churn_rate'],
        'churn_model_actual_rate':       week_metrics['actual_churn_rate'],
        'churn_drift_features_count':    float((drift_df['psi_level'] != 'OK').sum()),
        'churn_drift_psi_max':           float(drift_df['psi'].max()),
        'churn_model_critical_alerts':   float(n_critical_alerts),
    }

    for name, value in metrics_map.items():
        g = Gauge(name, name, registry=registry)
        g.set(value)

    push_to_gateway(PUSHGATEWAY_URL, job='churn_monitor', registry=registry)
    print(f'Semana {week_metrics["week"]} enviada → Pushgateway')

# Simular envío semana a semana
for week_m in weekly_metrics:
    push_week_metrics(week_m, drift_report, report['critical_alerts'])
    time.sleep(12)  # > scrape_interval de Prometheus (10s)
```

### Ejecutar como Script

```bash
# Script completo con reporte JSON
python src/monitor.py --weeks 8 --output reports/monitoring_report.json

# Con envío a CloudWatch (requiere credenciales AWS configuradas)
python src/monitor.py --cloudwatch-namespace ChurnPredictor/Production
```

### Job en CI/CD (`monitor`)

Se ejecuta después del deploy a producción y también por schedule (cada lunes 06:00 UTC). Si detecta degradación, dispara automáticamente el pipeline de reentrenamiento y abre un issue de alerta.

```yaml
# .github/workflows/mlops.yml — job: monitor
monitor:
  name: Monitoring
  runs-on: ubuntu-latest
  needs: [deploy-production, terraform-infra]
  if: always() && (github.ref == 'refs/heads/main' || github.event_name == 'schedule')
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.10'
        cache: 'pip'
    - name: Instalar dependencias
      run: pip install -r requirements.txt
    - name: Configurar credenciales AWS
      uses: aws-actions/configure-aws-credentials@v4
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
    - name: Descargar modelo de producción
      uses: actions/download-artifact@v4
      with:
        name: production-artifacts
        path: data/
    - name: Ejecutar monitoreo (PSI + métricas + alertas)
      run: |
        python src/monitor.py \
          --output reports/monitoring_report.json \
          --cloudwatch-namespace ${{ env.CLOUDWATCH_NAMESPACE }}
    - name: Subir reporte de monitoreo
      uses: actions/upload-artifact@v4
      with:
        name: monitoring-report
        retention-days: 90
        path: reports/
    - name: Verificar criticidad → trigger retraining
      id: check
      run: |
        REC=$(jq -r '.recommendation' reports/monitoring_report.json)
        echo "recommendation=$REC" >> $GITHUB_OUTPUT
        if [ "$REC" == "REENTRENAR" ]; then
          echo "::warning::Modelo degradado, retraining requerido"
        fi
    - name: Disparar pipeline de reentrenamiento (auto-retrain)
      if: steps.check.outputs.recommendation == 'REENTRENAR'
      env:
        GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      run: gh workflow run mlops.yml --ref main -f model_type=all
    - name: Abrir issue de alerta
      if: steps.check.outputs.recommendation == 'REENTRENAR'
      env:
        GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      run: |
        gh issue create \
          --title "Model Drift detectado - $(date -u +%F)" \
          --body-file reports/monitoring_report.json \
          --label "model-drift,critical"
```

> El schedule `cron: '0 6 * * 1'` ejecuta este job cada lunes a las 06:00 UTC independientemente de si hubo push.

### Resumen Fase 5

| Check               | Herramienta        | Umbral / Criterio                      |
|---------------------|--------------------|----------------------------------------|
| Data Drift          | KS Test + PSI      | PSI > 0.10 warning, > 0.25 crítico     |
| F1-Score            | sklearn             | < 0.60 → alerta CRITICAL              |
| ROC-AUC             | sklearn             | < 0.65 → alerta WARNING               |
| Churn Rate Delta    | Comparación baseline| > 10% delta → alerta WARNING          |
| Dashboard           | Grafana             | <http://localhost:3000>                  |

### Cuándo Reentrenar

- F1 cae por debajo de 0.60 por 2 semanas consecutivas
- ROC-AUC < 0.65 sostenido
- PSI crítico (> 0.25) en más de 2 features numéricas
- Tasa de churn real diverge > 10% de la baseline

---

## Configuración del Pipeline CI/CD

### Variables de entorno del Workflow

Definidas a nivel de workflow en `.github/workflows/mlops.yml`. Se aplican a todos los jobs:

```yaml
env:
  MLFLOW_TRACKING_URI: ./mlruns            # tracking local dentro del runner
  EXPERIMENT_NAME: churn-prediction-fintech
  AWS_REGION: us-east-1
  ECR_REPOSITORY: churn-predictor-prod     # nombre del repositorio en ECR
  CLOUDWATCH_NAMESPACE: MLOps/ChurnPredictor
```

### Triggers del Workflow

```yaml
on:
  push:
    branches: [main, develop]
    paths:                       # solo dispara si cambian estos archivos
      - 'src/**'
      - 'data/**'
      - 'app/**'
      - 'infra/**'
      - 'Dockerfile'
      - 'requirements.txt'
      - '.github/workflows/mlops.yml'
  schedule:
    - cron: '0 6 * * 1'         # monitoreo semanal — lunes 06:00 UTC
  workflow_dispatch:
    inputs:
      model_type:
        description: 'Tipo de modelo a entrenar'
        required: false
        default: 'logistic'
        type: choice
        options: [logistic, random_forest, xgboost]
```

### Secrets del Repositorio

Configurar en **GitHub → Settings → Secrets and variables → Actions → Repository secrets**:

| Secret                   | Descripción                                                     | Requerido en job   |
|--------------------------|-----------------------------------------------------------------|--------------------|
| `AWS_ACCESS_KEY_ID`      | Access key del usuario IAM con permisos ECR + ECS + CloudWatch  | `build-image`, `deploy-*`, `monitor`, `terraform-infra` |
| `AWS_SECRET_ACCESS_KEY`  | Secret key correspondiente al `AWS_ACCESS_KEY_ID`               | ídem               |
| `GRAFANA_ADMIN_PASSWORD` | Password de admin para Grafana (mínimo 8 chars)                 | `terraform-infra`  |

> `GITHUB_TOKEN` es automático — GitHub lo provee en cada ejecución, no requiere configuración manual.

**Pasos para agregar un secret:**

1. Ir a `Settings → Secrets and variables → Actions`
2. Clic en **New repository secret**
3. Ingresar el nombre exacto (ej. `AWS_ACCESS_KEY_ID`) y el valor
4. Clic en **Add secret**

Los secrets `AWS_ACCESS_KEY_ID` y `AWS_SECRET_ACCESS_KEY` deben pertenecer a un usuario IAM con la política mínima descrita en el Paso 6.5.

---

## Fase 6: Despliegue en AWS

**Infraestructura:** `infra/` (Terraform)  
**CI/CD:** `.github/workflows/mlops.yml` (GitHub Actions)

### Arquitectura en AWS

```
Internet
    │
    ▼
Application Load Balancer (ALB)
    ├── :80   → ECS Fargate: churn-predictor API
    └── :3000 → ECS Fargate: Grafana (EFS para persistencia)
                    │
                    └── Datasource: CloudWatch
                              ▲
                              │ src/monitor.py --cloudwatch-namespace
                         GitHub Actions (cron lunes 06:00 UTC)
```

**Recursos creados con Terraform:**

| Recurso                  | Descripción                                               |
|--------------------------|-----------------------------------------------------------|
| VPC + 2 subnets públicas | Red aislada en 2 AZs, CIDR `10.0.0.0/16`                |
| ECR Repository           | Registro de imágenes Docker (`churn-predictor-prod`)     |
| ECS Cluster (Fargate)    | Cluster con Container Insights habilitado                 |
| ECS Task: API            | 512 CPU / 1024 MB — imagen del commit SHA                |
| ECS Task: Grafana        | 256 CPU / 512 MB — EFS para `/var/lib/grafana`           |
| ALB + Target Groups      | Health check en `/health` y `/api/health`                 |
| CloudWatch Log Group     | Logs de contenedores — retención 7 días                  |
| IAM Roles                | Execution role (ECR + logs) + Task role (CloudWatch read) |

### Paso 6.1 — Pre-requisitos

```bash
# AWS CLI configurado
aws configure
# o variables de entorno:
# AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION=us-east-1

# Terraform >= 1.6
terraform -version

# Docker (para build local)
docker -v
```

### Paso 6.2 — Aprovisionamiento de Infraestructura con Terraform

```bash
cd infra

# Inicializar providers
terraform init

# Ver plan de recursos a crear
terraform plan \
  -var="grafana_admin_password=TuPasswordSeguro123"

# Aplicar (crea ~15 recursos en AWS)
terraform apply \
  -var="grafana_admin_password=TuPasswordSeguro123" \
  -auto-approve

# Outputs importantes
terraform output ecr_repository_url   # → URL del ECR
terraform output api_url              # → http://<alb-dns>/
terraform output grafana_url          # → http://<alb-dns>:3000
```

> El estado de Terraform se guarda localmente en `infra/terraform.tfstate`. Para equipos, descomentar el backend S3 en `infra/main.tf`.

### Paso 6.3 — Build y Push Manual a ECR

```bash
# Variables
ECR_URL=$(cd infra && terraform output -raw ecr_repository_url)
IMAGE_TAG=$(git rev-parse --short HEAD)

# Login
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin $ECR_URL

# Verificar artefactos (el Dockerfile los copia)
ls data/production_model.pkl data/production_model_info.json \
      data/scaler.pkl data/feature_names.csv

# Build y push
docker build -t $ECR_URL:$IMAGE_TAG .
docker push $ECR_URL:$IMAGE_TAG

echo "Imagen: $ECR_URL:$IMAGE_TAG"
```

### Paso 6.4 — Deploy Manual a ECS

```bash
ECS_CLUSTER="churn-predictor-prod"
ECS_SERVICE="churn-predictor-prod-service"
ECR_URL=$(cd infra && terraform output -raw ecr_repository_url)
IMAGE_TAG=$(git rev-parse --short HEAD)
NEW_IMAGE="$ECR_URL:$IMAGE_TAG"

# 1. Obtener task definition family actual
TASK_FAMILY=$(aws ecs describe-services \
  --cluster $ECS_CLUSTER --services $ECS_SERVICE \
  --query 'services[0].taskDefinition' --output text | sed 's/:[0-9]*$//')

# 2. Registrar nueva task definition con la imagen actualizada
NEW_TASK_DEF=$(aws ecs describe-task-definition \
  --task-definition "$TASK_FAMILY" \
  --query 'taskDefinition' \
  | jq --arg IMAGE "$NEW_IMAGE" \
    'del(.taskDefinitionArn,.revision,.status,.requiresAttributes,.compatibilities,.registeredAt,.registeredBy)
     | .containerDefinitions[0].image = $IMAGE')

NEW_ARN=$(aws ecs register-task-definition \
  --cli-input-json "$NEW_TASK_DEF" \
  --query 'taskDefinition.taskDefinitionArn' --output text)

# 3. Actualizar servicio (rolling deploy sin downtime)
aws ecs update-service \
  --cluster $ECS_CLUSTER \
  --service $ECS_SERVICE \
  --task-definition "$NEW_ARN"

# 4. Esperar estabilización
aws ecs wait services-stable \
  --cluster $ECS_CLUSTER \
  --services $ECS_SERVICE

echo "Deploy OK → $NEW_IMAGE"
```

### Paso 6.5 — CI/CD con GitHub Actions

El workflow `.github/workflows/mlops.yml` automatiza las 10 fases del pipeline. Se dispara en push a `main`/`develop`, schedule semanal (monitoreo) y `workflow_dispatch`.

**Jobs y orden de ejecución:**

```
push → main / develop
         │
         ▼
    [1] test           Quality Gate: pytest + coverage
         │
         ▼
    [2] data-prep      python src/data_prep.py → artifact
         │
         ▼
    [3] train          python src/train.py --model all → artifact
         │
         ▼
    [4] register       python src/evaluate_and_register.py → artifact
         │
         ▼
    [5] smoke-test     uvicorn + curl /health (sin AWS)
         │
         ▼
    [6] build-image    docker build + push ECR (SHA tag)
         │
    ┌────┴────┐
    ▼         ▼
[7] deploy    [7] deploy
  staging       production
(develop)      (main)
                 │
                 ▼
           [9] terraform-infra   terraform apply
                 │
                 ▼
          [10] monitor           PSI + CloudWatch + auto-retrain
```

**Permisos IAM mínimos para el usuario de CI:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecs:DescribeServices",
        "ecs:DescribeTaskDefinition",
        "ecs:RegisterTaskDefinition",
        "ecs:UpdateService",
        "ecs:DescribeClusters"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": ["iam:PassRole"],
      "Resource": "arn:aws:iam::*:role/churn-predictor-*"
    },
    {
      "Effect": "Allow",
      "Action": ["cloudwatch:PutMetricData"],
      "Resource": "*"
    }
  ]
}
```

### Paso 6.6 — Monitoreo Automático con CloudWatch

El job `monitor` del workflow envía métricas al namespace `MLOps/ChurnPredictor` en CloudWatch. Grafana (en ECS) las lee via datasource CloudWatch usando el task role (sin access keys hardcodeadas).

```bash
# Ejecutar monitoreo manualmente con envío a CloudWatch
python src/monitor.py \
  --weeks 8 \
  --output reports/monitoring_report.json \
  --cloudwatch-namespace MLOps/ChurnPredictor
```

**Métricas publicadas en CloudWatch:**

| Métrica                    | Descripción                            |
|----------------------------|----------------------------------------|
| `F1Score`                  | F1 del modelo en producción            |
| `ROCAUC`                   | ROC-AUC actual                         |
| `Gini`                     | Gini = 2×ROC-AUC − 1                  |
| `PredictedChurnRate`       | Tasa de churn predicha                 |
| `ActualChurnRate`          | Tasa de churn real                     |
| `PSIMax`                   | PSI máximo entre features              |
| `DriftFeaturesCount`       | Nº de features con drift moderado/crítico |
| `CriticalAlertsCount`      | Nº de alertas CRITICAL activas         |

### Paso 6.7 — Auto-Retrain

Si `recommendation == "REENTRENAR"` en el reporte de monitoreo, el job `monitor` dispara automáticamente el pipeline de reentrenamiento y abre un issue de alerta:

```yaml
- name: Disparar pipeline de reentrenamiento
  if: steps.check.outputs.recommendation == 'REENTRENAR'
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: gh workflow run mlops.yml --ref main -f model_type=all

- name: Abrir issue de alerta
  if: steps.check.outputs.recommendation == 'REENTRENAR'
  env:
    GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
  run: |
    gh issue create \
      --title "Model Drift detectado - $(date -u +%F)" \
      --body-file reports/monitoring_report.json \
      --label "model-drift,critical"
```

### Paso 6.8 — Verificar Deployment

```bash
# URL del ALB
API_URL=$(cd infra && terraform output -raw api_url)

# Health check
curl $API_URL/health

# Info del modelo desplegado
curl $API_URL/model/info

# Predicción en producción
curl -X POST $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "senior_citizen": 0,
    "tenure_months": 3,
    "contract_type": "Month-to-Month",
    "internet_service": "Fiber optic",
    "monthly_charges": 90.0,
    "support_tickets_last_6m": 6
  }'

# Logs del contenedor
aws logs tail /ecs/churn-predictor-prod --follow
```

### Paso 6.9 — Destruir Infraestructura

```bash
cd infra

# Ver qué se destruirá
terraform plan -destroy -var="grafana_admin_password=x"

# Destruir todos los recursos (IRREVERSIBLE)
terraform destroy \
  -var="grafana_admin_password=x" \
  -auto-approve
```

> Destruir elimina VPC, ECR, ECS cluster, ALB, EFS y todos los recursos asociados. Las imágenes en ECR se pierden.

### Resumen Fase 6

| Componente         | Servicio AWS          | Costo estimado (us-east-1)        |
|--------------------|-----------------------|-----------------------------------|
| API container      | ECS Fargate (0.5 vCPU / 1 GB) | ~$15/mes               |
| Grafana container  | ECS Fargate (0.25 vCPU / 0.5 GB) | ~$8/mes             |
| Grafana storage    | EFS                   | ~$0.30/GB/mes                     |
| Load Balancer      | ALB                   | ~$18/mes + $0.008/LCU-hora        |
| Container registry | ECR                   | $0.10/GB/mes (primeros 10 free)   |
| Logs               | CloudWatch Logs       | $0.50/GB ingestado                |
| Métricas           | CloudWatch Metrics    | $0.30/métrica/mes (>10 free)      |

---

## Pipeline Completo: Ejecución Secuencial

```bash
# 1. Preparar datos
python src/data_prep.py

# 2. Entrenar modelos y seleccionar el mejor
python src/train.py --model all --experiment churn-prediction-fintech

# 3. Registrar y promover a Production
python src/evaluate_and_register.py

# 4. Levantar API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 5. Monitoreo (en otra terminal)
python src/monitor.py --weeks 8 --output reports/monitoring_report.json

# Opcional: stack de visualización
cd monitoring && docker compose up -d
```

## Dependencias

```bash
pip install -r requirements.txt
```

Librerías clave del proyecto:

| Librería               | Versión   | Uso                                    |
|------------------------|-----------|----------------------------------------|
| `mlflow`               | 2.10.2    | Tracking, Registry                     |
| `scikit-learn`         | 1.4.0     | Modelos, métricas, preprocesamiento    |
| `xgboost`              | 2.0.3     | Modelo XGBoost                         |
| `fastapi` + `uvicorn`  | 0.109.2   | API REST                               |
| `pydantic`             | 2.6.1     | Validación de schemas de entrada       |
| `prometheus-client`    | 0.20.0    | Push de métricas al stack de monitoreo |
| `scipy`                | —         | KS test para detección de drift        |
| `boto3`                | 1.34.51   | CloudWatch (opcional, para AWS)        |
