# Week 6 Overview
## Probabilistic Models & Bayesian Inference

## What This Project Is About

Suppose the business scenario you are operating in for this entire assignment:
You are a Data Scientist at a telecom company. The business is losing customers to churn and needs a production-ready predictive system with honest uncertainty quantification. Over three weeks, this project builds that system in stages — interpretable linear models (Week 4), high-performance tree ensembles (Week 5), then probabilistic models that replace every point estimate with a full posterior distribution (Week 6).

**Dataset:** Telco Customer Churn — 7,043 rows × 21 columns  
**Primary target:** `Churn` (binary: Yes/No, ~26.5% positive rate)  
**Secondary target (W5):** `tenure` (regression — months a customer stays)  
**Additional dataset (W6):** Mauna Loa CO₂ monthly means 1958–2001 (statsmodels built-in)

---

## Week 4 — Linear Models

### Goal
Establish a baseline predictive system for churn using linear models.

### What Was Built
- **Data cleaning:** Detected and fixed the `TotalCharges` whitespace bug (11 rows → NaN → median-imputed)
- **Models trained:** Logistic Regression (classification), Ridge Regression, Lasso Regression (feature selection)
- **Key insight:** The **Accuracy Trap** — a naïve all-"No Churn" model scores ~73% accuracy while catching zero churners
- **Evaluation:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, confusion matrix

### Key W4 Results

| Model | AUROC | F1 (Churn) | Recall (Churn) |
|---|---|---|---|
| Logistic Regression (baseline) | ~0.83 | ~0.60 | ~0.55 |
| Logistic Regression (tuned C) | ~0.84 | ~0.62 | ~0.57 |

### What Linear Models Got Right
- **Interpretability:** Coefficients are directly readable
- **Speed:** Training is near-instant; cross-validation is feasible without GPU
- **Calibration:** Logistic regression outputs well-calibrated probabilities by design

### Limitations That Motivated Week 5
- Cannot capture non-linear interactions (e.g., tenure × Contract_Type)
- Cannot represent threshold effects (customers < 6 months tenure churn at dramatically higher rate)
- Cannot handle feature interactions automatically

---

## Week 5 — Tree-Based Ensembles

### Goal
Replace the linear baseline with a production-ready ensemble system that addresses all limitations of linear models.

### What Was Built
- **Mathematical foundations:** Gini Impurity, Shannon Entropy, Information Gain from scratch
- **Ensemble models:** BaggingClassifier vs RandomForestClassifier, XGBoost with Grid Search
- **Production pipeline:** ImbPipeline (ColumnTransformer → SMOTE → RandomForest), leak-proof
- **SHAP interpretability:** Global summary plot + local waterfall for highest-confidence churner
- **Top 3 churn drivers (SHAP):** `Contract_Month-to-month`, `tenure`, `OnlineSecurity_No`
- **Deployment artifact:** `telco_churn_pipeline_v1.joblib`
- **Regression extension:** DecisionTreeRegressor + XGBRegressor on `tenure` (R² = 0.996)

### Key W5 Results

| Model | AUROC | F1 (Churn) | Recall (Churn) |
|---|---|---|---|
| Naive DT | ~0.69 | ~0.51 | ~0.52 |
| BaggingClassifier (100 trees) | ~0.82 | ~0.57 | ~0.53 |
| RandomForestClassifier (100 trees) | ~0.83 | ~0.58 | ~0.54 |
| **Full ImbPipeline (RF + SMOTE)** | **0.8146** | **0.5632** | **0.5481** |

### What Tree Ensembles Added Over Linear Models
- Non-linear boundaries (threshold effects captured exactly by splits)
- Automatic feature interactions (no manual polynomial engineering)
- SMOTE inside ImbPipeline for leak-proof class-imbalance handling
- SHAP values: game-theory-grounded attribution reliable under correlated features

---

## Week 6 — Probabilistic Models & Bayesian Inference

### Business Motivation
The VP of Retention raised two concerns about the Week 5 XGBoost model: (1) "When you say 73% churn probability — what is the margin of error on that number?" and (2) "We just launched a new contract tier with 40 customers — how far should I trust the model on this segment?" Week 6 replaces every point estimate with a posterior distribution to answer both directly.

### What Was Built

#### Part 1 — The Estimation Trinity: MLE, MAP, Full Bayes
- Extracted Group A (Month-to-month, n=3,875), Group B (Two-year, n=1,695), Group A_small (n=40)
- Computed MLE, MAP under Beta(2,8) prior, and full Beta posteriors for each group
- **Key finding:** For Group A_small (n=40), prior pull |MAP − MLE| = **0.0417** — the prior shifts the estimate 4.2 pp. For Group A large (n=3,875), pull = **0.0006** — negligible.
- Monte Carlo P(θ_A > θ_B) = **1.0000** — VP can be told with 100% probability that Month-to-month churns higher than Two-year, without a p-value.

#### Part 2 — Sequential Bayesian Updating & Dirichlet-Multinomial
- Implemented `update_posterior()` for Beta-Binomial sequential learning
- Posterior evolves from Beta(2,8) to tightly concentrated around the true churn rate (~0.265) as n grows from 1 to 500
- **Bayesian decision boundary:** P(θ > 0.25) first exceeds 90% at **n = 17** — compared to frequentist z-test requiring **n = 6,304** for 80% power
- Dirichlet-Multinomial for 3-category contract type; Laplace smoothing for unseen "Biannual" category

#### Part 3 — Multivariate Gaussians
- Fitted 2D Gaussian on (tenure, MonthlyCharges): μ = [32.37, 64.76], ρ = **+0.248** (positive correlation)
- Conditional distribution P(MonthlyCharges | tenure = 24): mean **~$61/month**, std **~$30/month**, 95% interval [$2, $120]
- 3D Gaussian on (tenure, MonthlyCharges, TotalCharges): condition number κ = **>> 100**, revealing near-collinearity
- Marginalisation proof: extracting the 2×2 upper-left block of Σ_3D recovers Σ_2D exactly (max diff < 1e-6)

#### Part 4 — Probabilistic Graphical Models
- **Bayesian Network** (pgmpy): DAG with 5 nodes, MLE-fitted CPTs via VariableElimination
  - Forward: P(Churn=1 | Contract=Month-to-month) = **0.3542** (empirical: 0.4271; discrepancy from discretisation + DAG structure)
  - Backward: P(Contract=Month-to-month | Churn=1) = **0.8017** — "explaining away" in action
- **Markov Random Field** (undirected): pairwise DiscreteFactor potentials from empirical joint frequencies; BeliefPropagation marginal

#### Part 5 — Gaussian Process Regression on Mauna Loa CO₂
- Composite kernel: `DotProduct` (linear trend) + `RBF * ExpSineSquared(period=1yr)` (annual cycle) + `WhiteKernel` (noise)
- **Test RMSE: 4.329 ppm** on 1994–2001 holdout (89 months, ~7 years beyond training)
- Gap experiment: uncertainty inflates **~2–4× inside the 1973–1975 gap** vs outside
- Extrapolation: 95% credible band exceeds 5 ppm at **~10 years** beyond training — the model confidence boundary
- **Structural contrast with W5 trees:** GP uncertainty grows continuously beyond training range; DecisionTreeRegressor predicts a flat line bounded at max(y_train) with no uncertainty signal

#### Part 6 — MCMC Bayesian Logistic Regression (PyMC / NUTS)
- Features: tenure + MonthlyCharges (scaled) + Contract_Month-to-month + InternetService_DSL + SeniorCitizen
- Priors: Normal(0,5) for intercept, Normal(0,2) for β coefficients (weakly informative)
- Sampling: 4 chains × 2,000 draws (1,000 tune), target_accept=0.90, cores=1
- **Convergence: R̂ ≤ 1.001, bulk-ESS ≥ 2,000 for all parameters** — all chains converged cleanly
- **β_Contract_Month-to-month:** posterior mean = **1.2932**, 94% HDI = **[1.0845, 1.4920]**
- **Frequentist MLE:** 1.2927 — nearly identical, because n ≈ 5,640 dominates the prior
- **Prior sensitivity check:** Normal(0, 2) vs Normal(0, 0.5) posteriors differ by < 0.05 — data-dominated
- **Deployment artifact:** `telco_bayes_lr_v1.pkl` — PyMC InferenceData, 4 chains, verified round-trip

### Key W6 Results

| Component | Key Output |
|---|---|
| Group A_small churn rate (MLE) | 0.375 (94% HDI: [0.22, 0.47]) |
| P(θ_A > θ_B) via Monte Carlo | 1.0000 (100% probability) |
| Bayesian decision boundary | n = 17 (vs frequentist n = 6,304) |
| Conditional P(MC \| tenure=24) | $61 ± $30 /month |
| Condition number κ(Σ_3D) | >> 100 (near-collinear features) |
| BN: P(Churn=1 \| Contract=M2M) | 0.3542 |
| BN: P(Contract=M2M \| Churn=1) | 0.8017 |
| GP Test RMSE (Mauna Loa) | 4.329 ppm |
| MCMC R̂ (all parameters) | ≤ 1.001 ✅ |
| MCMC bulk-ESS (all parameters) | ≥ 2,000 ✅ |
| β_Contract_M2M 94% HDI | [1.08, 1.49] |

---

## W4 vs W5 vs W6 — Progressive Comparison

| Dimension | Week 4 (Linear) | Week 5 (Tree Ensembles) | Week 6 (Probabilistic) |
|---|---|---|---|
| **Core question** | What is the predicted class? | What is the predicted class (better)? | What is the distribution over predictions? |
| **Output type** | Point estimate (probability) | Point estimate + SHAP attributions | Full posterior distribution + HDI |
| **Uncertainty** | None reported | None reported | 94% HDI reported explicitly |
| **Small-data regime** | Unreliable | Unreliable | Regularised via prior (Bayesian shrinkage) |
| **Feature dependencies** | Linear only | Non-linear, learned | Gaussian (MVN), graphical (BN/MRF) |
| **Sequential learning** | Not applicable | Not applicable | Beta-Binomial online update |
| **Temporal structure** | Not modelled | Not modelled | Gaussian Process with composite kernel |
| **Interpretability** | Coefficients | SHAP values | Posterior distributions + causal BN |
| **Deployment artifact** | In-memory model | `telco_churn_pipeline_v1.joblib` | `telco_bayes_lr_v1.pkl` (InferenceData) |
| **VP answer quality** | "73% churn probability" | "73% churn probability (SHAP explained)" | "73% churn probability — 94% HDI [68%, 78%]" |

---

## Why Probabilistic Models Complete the System

1. **Point estimates are incomplete answers.** The XGBoost model saying "73% churn probability" gives the peak of a distribution it never shows. The Bayesian logistic regression shows the VP the entire posterior — including that for the 40-customer segment, the uncertainty band spans [22%, 47%].

2. **Sequential updating enables early decisions.** The Beta-Binomial framework crossed the 90% decision boundary at n=17, compared to n=6,304 for a frequentist z-test. For new product segments, this is the difference between acting in weeks vs waiting for years.

3. **Gaussian Processes model continuous uncertainty.** Where the W5 DecisionTreeRegressor predicts a flat line bounded at max(y_train) for extrapolation, the GP provides a trumpet-shaped credible band that widens honestly as we move beyond the training horizon — with a quantifiable confidence boundary at +10 years.

4. **Bayesian Networks enable causal reasoning.** The BN answers questions the logistic regression cannot: given that we observe a customer has churned, which contract type was most likely? (80% Month-to-month). This backward inference is impossible from a discriminative model.

---

*Fusemachines AI Fellowship · Statistical Machine Learning*
