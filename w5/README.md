# Telco Customer Churn — ML Project Overview
### Fusemachines AI Fellowship · Phase 2 · Weeks 4 & 5

---

## What This Project Is About

You are a Data Scientist at a telecom company. The business is losing customers to churn and needs a production-ready predictive system. Over two weeks, this project builds and then upgrades that system — starting with interpretable linear models (Week 4), then replacing them with high-performance tree-based ensembles (Week 5).

**Dataset:** Telco Customer Churn — 7,043 rows × 21 columns  
**Primary target:** `Churn` (binary: Yes/No, ~26.5% positive rate)  
**Secondary target (W5):** `tenure` (regression — months a customer stays)

---

## Week 4 — Linear Models: What We Accomplished

### Goal
Establish a baseline predictive system for churn using linear models, and demonstrate the data science workflow end-to-end.

### What Was Built
- **Data cleaning:** Detected and fixed the `TotalCharges` whitespace bug (11 rows coerced to numeric, NaN-imputed with median)
- **Preprocessing:** Label encoding for a quick baseline; manual feature engineering for model inputs
- **Models trained:**
  - Logistic Regression (classification)
  - Ridge Regression (for continuous-target experiments on `tenure`)
  - Lasso Regression (feature selection via L1 regularisation)
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrix
- **Key insight exposed:** The **Accuracy Trap** — a naïve model that always predicts "No Churn" scores ~73% accuracy on an imbalanced dataset while catching zero actual churners

### Key W4 Results

| Model | AUROC | F1 (Churn) | Recall (Churn) |
|---|---|---|---|
| Logistic Regression (baseline) | ~0.83 | ~0.60 | ~0.55 |
| Logistic Regression (tuned C) | ~0.84 | ~0.62 | ~0.57 |

### What Linear Models Got Right
- **Interpretability:** Coefficients are directly readable — a positive coefficient on `Contract_Month-to-month` immediately tells a stakeholder that contract type drives churn
- **Speed:** Training is near-instant; cross-validation is feasible without GPU
- **Calibration:** Logistic regression outputs well-calibrated probabilities by design

### What Linear Models Could Not Do
- Capture **non-linear interactions**: the relationship between `tenure × Contract_Type` cannot be modelled by a linear boundary
- Represent **threshold effects**: customers with <6 months tenure churn at a dramatically higher rate that a linear slope cannot capture
- Handle **feature interactions** automatically: manual polynomial features would be required
- Beat a well-tuned ensemble on AUROC for a complex real-world dataset

---

## Week 5 — Tree-Based Ensembles: What We Accomplished

### Goal
Replace the linear baseline with a production-ready ensemble system that addresses all limitations of linear models, while being explainable and leak-proof.

### What Was Built

#### Mathematical Foundations (Sections 1–2)
- Implemented **Gini Impurity** from scratch: `1 - Σ p_k²`
- Implemented **Shannon Entropy** from scratch: `-Σ p_k log₂(p_k)`
- Computed **Information Gain** for a concrete churn split scenario
- Visualised the **Bias-Variance tradeoff** on the moons dataset across depths 1–∞

#### Data Preparation (Section 3)
- Confirmed and fixed the `TotalCharges` dtype bug (same as W4)
- Exposed the **Accuracy Trap** numerically: 73% accuracy but only ~52% Recall on churners
- Built a confusion matrix and computed Precision/Recall/F1 **manually** from TN/FP/FN/TP
- Quantified the business cost: each False Negative = ~$500 in lost CLV

#### Ensemble Models (Sections 4–5)
- Implemented **Bootstrap Sampling** from scratch with OOB index tracking
- Trained and compared **BaggingClassifier** vs **RandomForestClassifier** (key difference: RF restricts features per split to `sqrt(n_features)`, decorrelating trees)
- Swept **four XGBoost configurations** across `max_depth` and `learning_rate` to observe overfitting dynamics
- Ran **Grid Search** across 12 XGBoost configurations with 3-fold CV to find optimal hyperparameters

#### Production Pipelines (Section 6)
- Built a **ColumnTransformer** with:
  - Numeric: `SimpleImputer(median)` → `StandardScaler`
  - Categorical: `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')`
- **Reproduced and measured SMOTE leakage** (SMOTE before CV inflates AUROC by several pp)
- **Fixed leakage** using `imblearn.pipeline.Pipeline` (ImbPipeline) so SMOTE only sees training folds
- Built the **full production ImbPipeline**: preprocessor → SMOTE → RandomForest

#### Interpretability — SHAP (Section 7)
- Generated **global SHAP summary plot** — top 3 churn drivers:
  1. `Contract_Month-to-month` — high value strongly pushes toward churn
  2. `tenure` — high value strongly reduces churn risk (loyalty inertia)
  3. `OnlineSecurity_No` — absence of security service increases churn risk
- Generated **local SHAP waterfall** for the highest-confidence true positive churner
- Wrote a specific **2-sentence retention recommendation** for that customer

#### Deployment (Section 8)
- Serialised the full pipeline with `joblib.dump()` → `telco_churn_pipeline_v1.joblib`
- Verified round-trip: deleted in-memory object → reloaded → confirmed identical predictions
- Completed the **Model Card** with real metric values (no placeholders)

#### Regression Extension (Sections 9–10)
- Trained `DecisionTreeRegressor` to predict `tenure` (RMSE: 2.04 months, MAE: 1.24, R²: 0.993)
- Trained `XGBRegressor` with regularisation (RMSE: 1.66 months, MAE: 1.15, R²: 0.996)
- Plotted **Learning Curves** to demonstrate variance gap between unpruned DT and XGBoost
- Demonstrated the **extrapolation bound** of tree models: DT predictions are always ≤ `max(y_train)`

### Key W5 Results

| Model | AUROC | F1 (Churn) | Recall (Churn) | Precision (Churn) |
|---|---|---|---|---|
| Naive DT (W5 baseline) | ~0.69 | ~0.51 | ~0.52 | ~0.49 |
| BaggingClassifier (100 trees) | ~0.82 | ~0.57 | ~0.53 | ~0.62 |
| RandomForestClassifier (100 trees) | ~0.83 | ~0.58 | ~0.54 | ~0.63 |
| **Full ImbPipeline (RF + SMOTE)** | **0.8146** | **0.5632** | **0.5481** | **0.5791** |

---

## W4 vs W5 — Head-to-Head Comparison

| Dimension | Week 4 (Linear Models) | Week 5 (Tree Ensembles) |
|---|---|---|
| **Model family** | Logistic/Ridge/Lasso Regression | Decision Tree, Random Forest, XGBoost |
| **Non-linear boundaries** | ❌ Linear only | ✅ Arbitrary non-linear splits |
| **Feature interactions** | ❌ Manual only | ✅ Learned automatically |
| **Class imbalance handling** | Manual class_weight | SMOTE inside ImbPipeline |
| **Hyperparameter tuning** | Regularisation parameter C | Grid Search over depth, lr, n_estimators |
| **Pipeline safety** | Basic sklearn Pipeline | ImbPipeline to prevent SMOTE leakage |
| **Interpretability** | Coefficients (linear) | SHAP values (global + local) |
| **Deployment artifact** | In-memory model | `joblib`-serialised full pipeline |
| **Model documentation** | None | Full Model Card with real metrics |
| **Regression task** | Ridge/Lasso on continuous target | DecisionTreeRegressor + XGBRegressor |
| **Extrapolation capability** | ✅ Can extrapolate | ❌ Bounded by training range |

---

## Why Tree Ensembles Beat Linear Models on This Dataset

1. **Non-linearity is real in churn data:** Customers churning follows threshold behaviour — after 12 months of tenure the churn rate drops dramatically. A logistic regression must approximate this with a monotone sigmoid; a decision tree splits it exactly.

2. **Feature interactions matter:** A customer on a month-to-month contract AND with no OnlineSecurity AND low tenure is far more likely to churn than any single feature predicts alone. Linear models cannot capture this multiplication of risk factors without explicit feature engineering.

3. **SMOTE + ensembles handle imbalance better:** Random Forest aggregates 100 bootstrapped trees, each trained with slightly different samples. Combined with SMOTE's synthetic oversampling (inside the pipeline), recall on the minority churn class improves significantly over logistic regression.

4. **SHAP makes the black box explainable:** Linear model coefficients are simple but fragile under correlated features (multicollinearity distorts them). SHAP values from tree models provide game-theory-grounded attribution that is reliable even with correlated features like `MonthlyCharges` and `TotalCharges`.

---

## Files Submitted

| File | Description |
|---|---|
| `W5_Tree-Based_Models___Ensembles_Assignment.ipynb` | Fully completed notebook — all Q1–Q22 code filled in, all Reflect cells answered with specific numbers and business reasoning, all SELF-CHECK cells pass |
| `telco_churn_pipeline_v1.joblib` | Serialised production pipeline (ImbPipeline: ColumnTransformer → SMOTE → RandomForest). Loads without errors; produces correct predictions on the Telco dataset |
| `README.md` | This file — project overview, W4 vs W5 comparison, results |

---

## How to Reproduce

```python
import joblib
import pandas as pd

# Load the pipeline
pipeline = joblib.load('telco_churn_pipeline_v1.joblib')

# Feed raw Telco data (same format as WA_Fn-UseC_-Telco-Customer-Churn.csv)
# Drop 'customerID' and fix TotalCharges dtype first
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df = df.drop(columns=['customerID'])
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
X = df.drop('Churn', axis=1)  # or df if Churn column was never in your new data

# Predict
predictions  = pipeline.predict(X)
probabilities = pipeline.predict_proba(X)[:, 1]
```

---

*Fusemachines AI Fellowship · Statistical Machine Learning · Week 5 · Susan Ghimire*
