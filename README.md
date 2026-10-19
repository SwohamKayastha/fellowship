# Fusemachines AI Fellowship 2026 — Work Log (W1–W18)

Everything built across the fellowship, one folder per week. Arc: data wrangling, then backend and text-to-SQL agents, classical ML, computer vision, NLP and transformers, a production RAG assistant, agentic loops, MLOps, guardrails, and a final research proposal.

## Overview

| Week | Folder | Topic | Deliverable |
|---|---|---|---|
| 1 | `w1_data_wrangling/` | Data wrangling and EDA | Heart-attack-risk notebooks, reflection answers |
| 2 | `w2/` | Backend and data (Postgres, FastAPI) | Docker Postgres, CRUD API, dashboard-count API |
| 3 | `w3/` | LLM text-to-SQL, agentic AI | SQL benchmark, prompt-chain pipeline, LangGraph agent |
| 4 | `w4/` | Linear models | Telco churn: LogReg, Ridge, Lasso |
| 5 | `w5/` | Tree models and ensembles | RF/XGBoost, SMOTE pipeline, SHAP |
| 6 | `w6/` | Probabilistic and Bayesian models | Beta-Binomial, GP regression, PyMC MCMC |
| 7 | `w7/` | Clustering | Customer segmentation notebook |
| 8 | `w8/` | Depth-map geometry features, MCQs | Screenshots of deliverable and quiz answers |
| 9–10 | — | No artifacts in the repo | — |
| 11 | `w11/` | Computer vision with deep learning | ResNet, Faster R-CNN, DeepLab, VAE, ViT, CLIP, ONNX |
| 12 | `w12/` | NER for customer support | CRF on CoNLL-2003 and WNUT-17 |
| 13 | `w13/` | LSTM text classification | PyTorch LSTM and reflection |
| 14 | `w14/` | Support-ticket routing | DistilBERT vs Qwen2.5-0.5B + LoRA |
| 15 | `w15/` | Production RAG assistant | FastAPI, ChromaDB, Streamlit, Docker, tool calling |
| 16 | `w16/` | Agentic self-check loop | `/chat/agent`, verifier sub-LLM, context compaction, eval harness |
| 17 | `w17/` | MLOps | Track A: churn with MLflow, Evidently, serving. Track B: RAG with prompt-version tracking |
| 18 | `w18/` | Guardrails | Input and output guardrail pipeline, 47-case eval |
| Final | `project/` | Research project | FactCAR proposal and scope docs, AstroCAR future-project scope |

Per-week READMEs exist in `w3`, `w5`, `w6`, `w14`, `w15`, `w16`, `w17/task_a`, `w17/task_b` and `w18`. Open those for setup details.

---

## Phase 1 — Foundations and Backend (W1–W3)

### W1 — Data wrangling (`w1_data_wrangling/`)
- Heart-attack-risk dataset: 3 files, 8,763 patients each, zero missing values.
- Split `Blood Pressure` ("120/80") into numeric `Systolic BP` and `Diastolic BP` (r = 0.95).
- Bar charts for categoricals, distributions, Pearson and Spearman correlations, anomaly checks.
- Notebooks: `completed/…COMPLETE.ipynb` and `…REFINED.ipynb`. Summary in `COMPLETION_SUMMARY.md` and `FINAL_REFLECTION_ANSWERS.md`.
- Helper scripts used to fill the notebook (`fill_*.py`, `generate_completed_notebook.py`).

### W2 — Postgres and FastAPI (`w2/`)
- `task1/`: Postgres via `docker-compose.yml`, seeded with the `classicmodels` schema (`seed.sql`).
- `task2/`: FastAPI + SQLAlchemy + Pydantic CRUD for customers (`GET/POST/PUT/DELETE`), plus nested `/{id}/orders` and `/{id}/payments`. File logging.
- `task3/`: adds a dashboard router with `/…/count` endpoints for customers, orders, products, employees, offices, payments, order details and product lines, plus `/overall_counts`.
- Assignment PDFs are in `task/` and `Completed-assignment/`. Zipped copies are in `w2.rar` and `compressed-task2/`.

### W3 — Text-to-SQL and agents (`w3/`)
All on the `classicmodels` Postgres schema.
- `Task1/`, `Task2/`: SQL benchmark queries with explanations, evaluation framework (LaTeX), and query-decomposition list.
- `SQL_AGENT/`: FastAPI baseline. Prompt chain `pipeline.py` → `llm.py` → `validator.py` (SELECT-only safety) → `database.py`, plus `benchmark.py` run over the question CSV.
- `Task3/`: same pipeline with a Streamlit UI, `evaluate.py`, Docker setup and logging.
- `Task4/`: agentic version on LangGraph. Agents are planner, generator, validator, executor and summarizer. It has FastAPI and Streamlit front ends.
- `Agent/`, `Main_Task/`: earlier iterations, including `query_decomposition.py`.

---

## Phase 2 — Statistical ML (W4–W8)

Shared scenario: Telco Customer Churn (7,043 × 21, about 26.5% churn).

### W4 — Linear models (`w4/`)
- Fixed the `TotalCharges` whitespace bug (11 rows, median-imputed).
- Demonstrated the accuracy trap: an always-"No" model scores about 73% accuracy and catches zero churners.
- Logistic Regression (AUROC ~0.83–0.84, F1 ~0.60–0.62), Ridge, Lasso.

### W5 — Tree ensembles (`w5/`)
- Gini, entropy and information gain implemented from scratch. Bias-variance demo.
- Bagging vs Random Forest vs tuned XGBoost.
- Leak-proof `ImbPipeline` (ColumnTransformer → SMOTE → RF): AUROC 0.8146, F1 0.5632, recall 0.5481.
- SHAP: top drivers are `Contract_Month-to-month`, `tenure`, `OnlineSecurity_No`.
- Artifact `telco_churn_pipeline_v1.joblib`. Regression extension on `tenure` (R² = 0.996).

### W6 — Probabilistic models (`w6/`)
- MLE vs MAP vs full Beta posteriors per segment. Sequential Beta-Binomial updating.
- Gaussian Process regression on Mauna Loa CO₂.
- PyMC (NUTS) Bayesian logistic regression. β_Contract_Month-to-month = 1.29, 94% HDI [1.08, 1.49].
- `W6_Reflection.md`: how a Bayesian answer changes the decision for a 40-customer segment. `quiz/` holds quiz screenshots.

### W7 — Clustering (`w7/task/`)
Customer segmentation notebook. It covers cleaning and feature engineering into a customer matrix, K-Means, hierarchical clustering and DBSCAN, then cluster validation and comparison. Last come a business narrative with cluster profiles and executive summary, and a failure log.

### W8 (`w8/`)
Screenshots only: a depth-map geometric-features catalogue (edge sharpness, depth variance, radial symmetry, planar fit, curvature, aspect ratio, gradient direction, shape consistency) and four MCQ answers.

---

## Phase 3 — Deep Learning and NLP (W11–W14)

### W11 — Computer vision (`w11/`)
`W11_CV_Assignment_Notebook_LOCAL.ipynb` has 20 questions across five modules, with self-checks passing.
- CNN output-size formula, ResNet-50 transfer learning on CIFAR-10 (75% val accuracy in the run), GradCAM.
- IoU, Faster R-CNN inference, NMS.
- DeepLabv3+, pixel accuracy and mIoU, U-Net diagram (`diagram.png`).
- VAE training, reconstruction and latent interpolation.
- ViT patch embedding, CLIP zero-shot (92% on 200 test images), ONNX export and onnxruntime check, deployment memo.

### W12 — NER for customer support (`w12/`)
- CRF with hand-built contextual features on CoNLL-2003. Entity-level scores include B-PER F1 0.84, B-LOC 0.84, B-ORG 0.75.
- Error analysis, business-impact and deployment discussion.
- Extension to the harder WNUT-17 emerging-entities set. Scores drop sharply, with rare classes near zero recall.
- Notebook: `Assignment3_NER_CustomerSupport.ipynb`. Drafts are in `unused/`.

### W13 — LSTM text classification (`w13/`)
PyTorch LSTM with vocabulary building, `padding_idx`, and `hidden[-1]` as the sequence summary. Three experiments went from underfitting to overfitting; the best run reached about 48% val accuracy at epoch 10. Reflection and inference are in `LSTMs_for_Text_Classification_final.ipynb`. Earlier drafts are alongside it.

### W14 — Support-ticket routing (`w14/`, `README.md`)
ShopAssist AI routes messages to 11 agents using the Bitext dataset.
- Approach 1: DistilBERT `[CLS]` classifier. Macro-F1 99.87%, 4.64 ms, 789 MB GPU.
- Approach 2: Qwen2.5-0.5B-Instruct + LoRA generating the agent name. Macro-F1 99.94%, 198 ms, 6.3 GB GPU.
- Recommendation: ship DistilBERT. About 42× faster for a 0.07-point gap. Plots are in `public/`, weights in `model_weights/`.
- `code_support_routing.ipynb` and `bipad_car_architecture.svg` at the repo root are related scratch work.

---

## Phase 4 — LLM Applications and MLOps (W15–W18)

### W15 — AI assistant (`w15/`)
- FastAPI backend, Streamlit UI, Docker Compose.
- RAG: Sentence-Transformers (all-MiniLM-L6-v2) with ChromaDB (HNSW cosine), auto-ingest on startup.
- LLM client: OpenAI gpt-4o-mini → Gemini 2.0 Flash → local Qwen3-4B-AWQ (vLLM). Falls back to local models when no API key is set.
- Tool calling (`calculator`, `get_datetime`), `/chat/json` structured output, response cache, rate limiting (slowapi), retry with backoff (tenacity), cost tracking and `eval_with_budget.py`.

### W16 — Agentic self-check (`w16/`)
- `POST /chat/agent`: loop of at most 5 iterations. `rag_search` → draft → `verify_answer`, an isolated sub-LLM call with only the answer and sources.
- Context engineering: tool results are cleared after a passing verification (`_compact_messages`).
- `eval/run_eval.py` measures completion rate, tool-call correctness, trajectory length and tokens, with a failure taxonomy (hard / soft / cascading-soft) and optional failure injection.
- Agent-instruction files for several coding tools are included (`AGENTS.md`, `.cursor`, `.windsurf`, `.clinerules`, `.github`).

### W17 — MLOps (`w17/`)
- **Track A (churn)**: `uv`-locked environment. MLflow tracked four runs, and `logreg_C0.1` (F1 0.5901) was registered and promoted Staging → Production. Evidently drift report detected `MonthlyCharges` drift (Wasserstein 0.1027 against a 0.10 threshold). Scripts: `train.py`, `monitor.py`, `serve.py` (health and predict endpoints).
- **Track B (RAG assistant)**: `uv` environment. Three system-prompt versions (`prompts/prompt_v1–v3.txt`) compared in MLflow on completion, verified rate, iterations and tokens. Evidently LLM-as-judge regression tests and the cost/budget tooling are included.

### W18 — Guardrails (`w18/`)
- Support bot for a fictional company (Acme Cloud), answering only from a policy document. Pure standard library, offline mock LLM by default, optional OpenAI.
- Input guards: PII masking, prompt-injection blocking, sensitive-topic routing.
- Output guards: JSON schema, sensitive info and secret leaks, per-sentence groundedness.
- `evaluate.py` runs 47 labeled cases: 42/47 passed. Known misses are negation in groundedness and paraphrased injections. See `results.md`, `REPORT.md` and `tests/cases.py`.
- Run: `python chat.py "What is the refund window?"` or `python evaluate.py`.

---

## Final Project (`project/`)

- **FactCAR**: multi-agent fact-checking (Decomposer → Retriever → Verifier → Aggregator) on AVeriTeC inside a Control-Agency-Runtime harness. It is a measurement study of harness × LoRA interaction with 2-way ANOVA. Contents: scope and feasibility doc, LaTeX proposal and architecture SVGs.
- **AstroCAR**: future-project scope, demo and proposal.
- `agent.md`, `CLAUDE.md`, `diagram.py`: project helper files.

---

## Running things

- Notebooks (W1, W4–W7, W11–W14): open in Jupyter. Telco CSV is in `w4/`, and shared datasets (e.g. CIFAR-10) are in `data/`.
- Services (W2, W3, W15–W17): copy `.env.example` to `.env`, then use `docker compose up` or `uvicorn` per the week README. LLM calls need `OPENAI_API_KEY` / `GOOGLE_API_KEY` or a local server.
- Local LLM launch commands (llama.cpp `llama-server.exe` with Qwen/Gemma GGUFs) live in [`llama-server-commands.md`](llama-server-commands.md). This was the previous content of this file.

## Notes
- No `w9` or `w10` work is in the repo.
- Large artifacts (`.pkl`, weights) are tracked with Git LFS (`.gitattributes`).
