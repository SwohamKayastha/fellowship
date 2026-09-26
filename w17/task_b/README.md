# Track B — Agentic AI MLOps: RAG Assistant

Built on top of the W15 RAG assistant and W16 agentic self-check feature.

## Setup

```bash
pip install uv
uv sync
cp .env.example .env
# Fill in GOOGLE_API_KEY in .env
```

Start the assistant:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Start MLflow tracking server (separate terminal):

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

## a. Environment & Reproducibility (uv)

W16 used unpinned `requirements.txt` — `chromadb`, `sentence-transformers`, and `evidently[llm]` had conflicting numpy/protobuf versions. `uv` resolves these at lock time. From a clean clone:

```bash
uv sync
```

Also pinned `sentence-transformers>=2.2.0,<3.0.0` and `pyarrow>=15.0` to avoid downstream `datasets` package incompatibilities with Python 3.12.

## b. Experiment Tracking Strategy (MLflow)

**What was varied:** System prompt versions — `prompt_v1.txt` (baseline) → `prompt_v2.txt` (out-of-scope handling) → `prompt_v3.txt` (hard limits + workflow labeling).

**What was measured:** `task_completion_rate`, `verified_rate`, `avg_iterations`, `avg_tokens_per_query`, and regression test `pct_tests_passed`.

### Experiment Results

All 3 prompt versions ran against 6 test queries:

| Version | Completion | Verified | Avg Iter | Avg Tokens | Status |
|---------|-----------|----------|----------|------------|--------|
| v1 | 6/6 (100%) | 0/6 (0%) | 1.0 | 0 | Baseline |
| v2 | 6/6 (100%) | 0/6 (0%) | 1.0 | 0 | Variant 1 |
| v3 | 6/6 (100%) | 0/6 (0%) | 1.0 | 0 | Variant 2 |

**MLflow run logs:** http://127.0.0.1:5000/#/experiments/2

### Model Testing

**gemini-3.5-flash-lite** supports function calling via OpenAI-compatible endpoint:
- ✅ Tested: finish_reason="tool_calls" 
- ✅ Tool calls work: rag_search called successfully
- ✅ Token tracking: 18 completion + 344 prompt tokens returned

(Earlier testing with gemini-2.0-flash had Settings caching issue — server kept old model value. Fixed by using gemini-3.5-flash-lite in .env.)

**Run experiment:**

```bash
uv run python eval/run_prompt_experiment.py --base-url http://localhost:8000
```

## c. Monitoring & Regression Testing (Evidently AI)

**Reference set**: 5 golden query-answer pairs from the best-performing prompt version.

**Regression test suite**: Compares new prompt responses against reference using Evidently's `Test Suite` with LLM-as-judge (`BinaryClassificationPromptTemplate`).

**Run regression test:**

```bash
uv run python eval/regression_test.py --new-version v2
uv run python eval/regression_test.py --new-version v3
```

Reports saved to `reports/regression_<version>.html` and `pct_tests_passed` logged to MLflow for cross-version comparison.

## Code Structure

```
task_b/
├── app/              ← RAG assistant + agent loop + tool definitions
│   ├── agent/       ← Modified loop.py with trace capture
│   ├── llm/         ← LLM client, tool execution
│   ├── rag/         ← Retrieval, embeddings, ingestion
│   └── main.py      ← FastAPI endpoints
├── prompts/         ← prompt_v1.txt, v2.txt, v3.txt (versioned system prompts)
├── eval/            ← Experiment runner + regression test suite
├── data/            ← sample.txt (knowledge base)
└── pyproject.toml   ← Dependencies (uv managed)
```

## Agent Endpoint

`POST /chat/agent` accepts `prompt_version` field (`"v1"`, `"v2"`, `"v3"`):

```bash
curl -X POST http://localhost:8000/chat/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this assistant about?", "prompt_version": "v3"}'
```

Response includes full trace (step-by-step tool calls and reasoning) if tool-calling is enabled.

## Artifacts

- **pyproject.toml + uv.lock** — reproducible environment
- **MLflow runs** — experiment tracking across all 3 versions
- **Evidently reports** — regression test suite for each version
- **Prompt files** — explicit versioning in `prompts/`
- **Trace capture** — `app/agent/loop.py` records `{step, tool, args, result, reasoning}` per iteration
