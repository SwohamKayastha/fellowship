# Track A — Data Science MLOps: Telco Churn

## Setup

```bash
pip install uv
uv sync
```

Start MLflow tracking server (keep running in a separate terminal):

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

## a. Environment & Reproducibility (uv)

The Telco pipeline depends on `xgboost`, `evidently`, `mlflow`, and `scikit-learn` — versions that conflict when managed manually (e.g., evidently 0.4.x requires numpy <2.0 while xgboost 2.x prefers numpy >=1.26, and mlflow 2.13+ uses skops for sklearn model serialization which adds its own constraints). `uv` resolves all of this at lock time and reproduces the exact environment from a clean clone with one command:

```bash
uv sync
```

## b. Experiment Tracking Strategy (MLflow)

**What varied:** Model family and key regularization/depth hyperparameters across 4 runs. Random seed was fixed at 42 — only genuine hyperparameter differences.

**Run training (logs all runs, registers best, transitions Staging → Production):**

```bash
PYTHONUTF8=1 uv run python train.py
```

**View MLflow UI:** http://127.0.0.1:5000

### Full Run Comparison

| Run | Model | Key Params | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-----|-------|-----------|----------|-----------|--------|-----|---------|
| logreg_C0.1 | LogisticRegression | C=0.1 | 0.7999 | 0.6465 | 0.5428 | **0.5901** | 0.8390 |
| logreg_C10 | LogisticRegression | C=10 | 0.7977 | 0.6386 | **0.5481** | 0.5899 | 0.8405 |
| xgb_default | XGBoost | n_est=200, depth=4, lr=0.1 | 0.7984 | **0.6500** | 0.5214 | 0.5786 | 0.8367 |
| rf_depth5 | RandomForest | n_est=200, max_depth=5 | 0.7999 | 0.6966 | 0.4358 | 0.5362 | **0.8419** |

### Model Selection Justification

**Registered: `logreg_C0.1` (F1=0.5901)**

Accuracy is misleading on this dataset — all four models score ~0.80 accuracy, yet their churn-detection ability differs significantly.

`rf_depth5` has the highest ROC-AUC (0.8419) and highest precision (0.6966) but a recall of only 0.4358 — it misses 56% of actual churners. In a retention campaign, missing churners is the costly error, so high precision at the expense of recall is not acceptable here.

`logreg_C10` has the best recall (0.5481) but marginally lower F1 (0.5899) compared to `logreg_C0.1`. The difference is 0.0002 F1 — effectively tied — but `logreg_C0.1`'s stronger regularization generalizes better, so it was selected.

`xgb_default` underperforms on both F1 (0.5786) and ROC-AUC (0.8367) relative to the logistic regression runs despite more complexity — likely because the depth-4 tree splits are too shallow to capture interaction effects in this feature set without tuning.

**Decision:** `logreg_C0.1` maximizes F1, which balances catching churners (recall) against false alarms (precision). Registered as `ChurnModel` v1, transitioned Staging → Production.

## c. Monitoring & Drift Strategy (Evidently AI)

- **Reference set**: first 70% of the Telco CSV (training-time distribution, ~4,930 rows)
- **Current set**: remaining 30% (~2,113 rows) with synthetic drift injected:
  - `MonthlyCharges` shifted by random ±20 per row (simulates pricing change)
  - `Contract == Month-to-month` oversampled by 40% (simulates contract-type shift)
  - 5% of `Churn` labels flipped (simulates label/concept drift)

**Run drift monitoring:**

```bash
PYTHONUTF8=1 uv run python monitor.py
```

Report saved to `reports/drift_report.html` and logged as MLflow artifact under a `drift_monitoring` run.

### Drift Report Results (actual run)

| Column | Test | Drift Score | Threshold | Detected |
|--------|------|------------|-----------|----------|
| MonthlyCharges | Wasserstein distance | 0.1027 | 0.10 | **Yes** |
| Churn (target) | Jensen-Shannon distance | 0.0506 | 0.10 | No |
| Dataset overall | — | 1/7 cols drifted (14.3%) | — | No |

`MonthlyCharges` drift was correctly detected — the ±20 random per-row shift changed the distribution shape (variance increased significantly) even though the mean only moved from 64.85 to 65.01. The Wasserstein distance captures this shape change while a simple mean-shift check would miss it.

The target (`Churn`) did not trigger drift despite the 5% label flip — the flip rate was below the Jensen-Shannon threshold. This is realistic: small label noise is hard to distinguish from sampling variance.

Custom metric logged to MLflow: `monthly_charges_mean_shift` = 0.16.

**Production action:** `MonthlyCharges` Wasserstein > 0.10 is the trigger threshold. On detection: log a retraining recommendation and run `python train.py` with data from the current distribution period.

## Model Serving

Start serving after `train.py` has registered the Production model:

```bash
PYTHONUTF8=1 uv run uvicorn serve:app --host 0.0.0.0 --port 8001
```

**Health check:** `GET http://localhost:8001/health`

**Predict:**

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":0,"SeniorCitizen":0,"Partner":1,"Dependents":0,"tenure":12,"PhoneService":1,"MultipleLines":0,"InternetService":1,"OnlineSecurity":0,"OnlineBackup":1,"DeviceProtection":0,"TechSupport":0,"StreamingTV":0,"StreamingMovies":0,"Contract":0,"PaperlessBilling":1,"PaymentMethod":2,"MonthlyCharges":65.5,"TotalCharges":786.0}'
```

## Workflow

```
data/telco_churn.csv
  → train.py         (preprocess + 4 model configs + MLflow tracking + register logreg_C0.1 → Production)
  → serve.py         (FastAPI /predict endpoint, loads ChurnModel/Production from registry)
  → monitor.py       (Evidently drift report on 70/30 split with injected drift → MLflow artifact)
```
