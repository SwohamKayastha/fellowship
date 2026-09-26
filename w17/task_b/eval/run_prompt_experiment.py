"""
MLflow experiment runner for prompt versioning.

For each prompt version (v1, v2, v3):
  - Runs all test queries through the agent
  - Logs params, metrics, and trace artifacts to MLflow
  - Saves representative traces (success + failure cases)

Usage:
  python eval/run_prompt_experiment.py [--base-url http://localhost:8000] [--versions v1 v2 v3]
"""

import argparse
import json
import pathlib
import time

import mlflow
import requests

MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "agentic_prompt_experiment"
REPORTS_DIR = pathlib.Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

TEST_QUERIES = [
    {"id": "TC01", "query": "What is this assistant about?", "complexity": "simple"},
    {"id": "TC02", "query": "Summarize the main topics in the knowledge base.", "complexity": "simple"},
    {"id": "TC03", "query": "What is the capital of France?", "complexity": "out-of-kb"},
    {"id": "TC04", "query": "Compare the advantages and disadvantages of RAG versus fine-tuning for LLMs.", "complexity": "complex"},
    {"id": "TC05", "query": "What specific numbers or statistics are mentioned in the documents?", "complexity": "specific"},
    {"id": "TC06", "query": "xzqwerty nonsense query that matches nothing", "complexity": "no-match"},
]

PROMPT_VERSIONS = ["v1", "v2", "v3"]


def call_agent(base_url: str, query: str, prompt_version: str, timeout: int = 60) -> dict:
    resp = requests.post(
        f"{base_url}/chat/agent",
        json={"query": query, "temperature": 0.1, "top_p": 0.9, "prompt_version": prompt_version},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def run_version(base_url: str, version: str) -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    prompt_path = pathlib.Path(__file__).parent.parent / "prompts" / f"prompt_{version}.txt"
    prompt_text = prompt_path.read_text() if prompt_path.exists() else ""

    with mlflow.start_run(run_name=f"prompt_{version}"):
        # Log config params
        mlflow.log_param("prompt_version", version)
        mlflow.log_param("temperature", 0.1)
        mlflow.log_param("top_p", 0.9)
        mlflow.log_param("max_iterations", 5)
        mlflow.log_text(prompt_text, f"prompts/prompt_{version}.txt")

        results = []
        traces_success = []
        traces_failure = []

        for tc in TEST_QUERIES:
            t0 = time.time()
            try:
                raw = call_agent(base_url, tc["query"], version)
                completed = bool(raw.get("response", "").strip())
            except Exception as exc:
                raw = {}
                completed = False
                print(f"  [{tc['id']}] ERROR: {exc}")

            latency_ms = int((time.time() - t0) * 1000)
            verified = raw.get("verified", False)
            iterations = raw.get("iterations", 0)
            tokens = raw.get("tokens", {})
            trace = raw.get("trace", [])

            result = {
                "id": tc["id"],
                "query": tc["query"],
                "complexity": tc["complexity"],
                "completed": completed,
                "verified": verified,
                "iterations": iterations,
                "tokens_total": tokens.get("total", 0),
                "latency_ms": latency_ms,
                "response_snippet": raw.get("response", "")[:150],
                "termination_reason": raw.get("termination_reason", "unknown"),
            }
            results.append(result)
            print(f"  [{tc['id']}] verified={verified} iter={iterations} tokens={tokens.get('total', 0)}")

            # Rate limit: 5 req/min = 1 req per 12 sec
            if tc != TEST_QUERIES[-1]:  # don't sleep after last query
                time.sleep(13)

            # Collect traces: up to 2 success + 2 failure
            trace_record = {"query": tc["query"], "trace": trace, "verified": verified}
            if verified and len(traces_success) < 2:
                traces_success.append(trace_record)
            elif not verified and len(traces_failure) < 2:
                traces_failure.append(trace_record)

        # Aggregate metrics
        total = len(results)
        completed_count = sum(1 for r in results if r["completed"])
        verified_count = sum(1 for r in results if r["verified"])
        avg_iter = sum(r["iterations"] for r in results) / total
        avg_tokens = sum(r["tokens_total"] for r in results) / total
        avg_latency = sum(r["latency_ms"] for r in results) / total

        mlflow.log_metric("task_completion_rate", completed_count / total)
        mlflow.log_metric("verified_rate", verified_count / total)
        mlflow.log_metric("avg_iterations", avg_iter)
        mlflow.log_metric("avg_tokens_per_query", avg_tokens)
        mlflow.log_metric("avg_latency_ms", avg_latency)

        # Save trace artifacts
        for i, tr in enumerate(traces_success):
            path = REPORTS_DIR / f"{version}_trace_success_{i+1}.json"
            path.write_text(json.dumps(tr, indent=2))
            mlflow.log_artifact(str(path), artifact_path="traces")

        for i, tr in enumerate(traces_failure):
            path = REPORTS_DIR / f"{version}_trace_failure_{i+1}.json"
            path.write_text(json.dumps(tr, indent=2))
            mlflow.log_artifact(str(path), artifact_path="traces")

        # Save full results
        results_path = REPORTS_DIR / f"{version}_results.json"
        results_path.write_text(json.dumps(results, indent=2))
        mlflow.log_artifact(str(results_path), artifact_path="results")

        print(f"\n[{version}] completion={completed_count}/{total} verified={verified_count}/{total} avg_iter={avg_iter:.1f} avg_tokens={avg_tokens:.0f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--versions", nargs="*", default=PROMPT_VERSIONS)
    args = parser.parse_args()

    for version in args.versions:
        print(f"\n=== Running prompt {version} ===")
        run_version(args.base_url, version)


if __name__ == "__main__":
    main()
