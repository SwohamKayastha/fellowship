"""
Evidently AI LLM-as-judge regression test suite.

Compares new prompt version responses against golden reference answers.
Logs pct_tests_passed to MLflow for cross-version comparison.

Usage:
  python eval/regression_test.py --base-url http://localhost:8000 --new-version v2 --mlflow-run-id <run_id>
"""

import argparse
import pathlib

import mlflow
import pandas as pd
import requests
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import TextEvals
from evidently.descriptors import LLMEval, SemanticSimilarity

MLFLOW_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "agentic_prompt_experiment"
REPORTS_DIR = pathlib.Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Golden reference set — representative queries + approved answers from v1 best run
GOLDEN_SET = [
    {
        "query": "What is this assistant about?",
        "reference": (
            "This assistant is a RAG-based AI system that retrieves information from a knowledge base "
            "and verifies its answers before responding. It uses tools like rag_search and verify_answer "
            "to ensure grounded, accurate responses."
        ),
    },
    {
        "query": "Summarize the main topics in the knowledge base.",
        "reference": (
            "The knowledge base covers topics related to AI assistants, retrieval-augmented generation (RAG), "
            "vector embeddings, and LLM-based question answering systems."
        ),
    },
    {
        "query": "What is the capital of France?",
        "reference": (
            "This question is outside the scope of my knowledge base. I cannot find relevant information "
            "to answer this accurately."
        ),
    },
    {
        "query": "Compare the advantages and disadvantages of RAG versus fine-tuning for LLMs.",
        "reference": (
            "RAG allows dynamic retrieval of up-to-date information without retraining, making it flexible "
            "and cost-effective. Fine-tuning bakes knowledge into model weights for faster inference but "
            "requires expensive retraining when knowledge changes."
        ),
    },
    {
        "query": "What are the key features of this AI system?",
        "reference": (
            "Key features include: RAG-based retrieval using vector embeddings, a self-verification loop "
            "using a secondary LLM call, tool-use capabilities (calculator, rag_search, verify_answer), "
            "and a FastAPI backend with a Gradio UI."
        ),
    },
]


def call_agent(base_url: str, query: str, prompt_version: str, timeout: int = 60) -> str:
    resp = requests.post(
        f"{base_url}/chat/agent",
        json={"query": query, "temperature": 0.1, "top_p": 0.9, "prompt_version": prompt_version},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def run_regression(base_url: str, new_version: str, mlflow_run_id: str | None = None) -> float:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"Collecting responses for prompt {new_version}...")
    rows = []
    for item in GOLDEN_SET:
        try:
            response = call_agent(base_url, item["query"], new_version)
        except Exception as exc:
            response = f"ERROR: {exc}"
        rows.append({
            "question": item["query"],
            "reference": item["reference"],
            "response": response,
        })
        print(f"  [{item['query'][:50]}...] done")

    df = pd.DataFrame(rows)

    # Evidently report with LLM-as-judge semantic similarity check
    report = Report(metrics=[
        TextEvals(
            column_name="response",
            descriptors=[
                SemanticSimilarity(with_column="reference", display_name="semantic_similarity"),
            ],
        ),
    ])

    # Reference dataset uses golden answers as "response"; current uses new version responses
    reference_df = df[["question", "reference"]].rename(columns={"reference": "response"})
    current_df = df[["question", "response"]]

    report.run(reference_data=reference_df, current_data=current_df)

    report_path = REPORTS_DIR / f"regression_{new_version}.html"
    report.save_html(str(report_path))
    print(f"Regression report saved: {report_path}")

    # Simple pass/fail: similarity > 0.5 = pass
    result_dict = report.as_dict()
    similarity_scores = []
    for metric in result_dict.get("metrics", []):
        result = metric.get("result", {})
        current_val = result.get("current", {})
        if isinstance(current_val, dict):
            mean_val = current_val.get("mean")
            if mean_val is not None:
                similarity_scores.append(float(mean_val))

    pct_passed = sum(1 for s in similarity_scores if s > 0.5) / len(similarity_scores) if similarity_scores else 0.0
    print(f"\nSemantic similarity scores: {similarity_scores}")
    print(f"pct_tests_passed (similarity > 0.5): {pct_passed:.2%}")

    # Log to MLflow — either append to existing run or create new
    run_ctx = mlflow.start_run(run_id=mlflow_run_id) if mlflow_run_id else mlflow.start_run(run_name=f"regression_{new_version}")
    with run_ctx:
        mlflow.log_metric("pct_tests_passed", pct_passed)
        mlflow.log_metric("num_test_cases", len(GOLDEN_SET))
        mlflow.log_artifact(str(report_path), artifact_path="regression_reports")

    return pct_passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--new-version", default="v2", help="Prompt version to test")
    parser.add_argument("--mlflow-run-id", default=None, help="Append metrics to existing MLflow run")
    args = parser.parse_args()

    pct = run_regression(args.base_url, args.new_version, args.mlflow_run_id)

    if pct < 0.6:
        print(f"\nWARNING: Only {pct:.0%} tests passed — prompt {args.new_version} should NOT be promoted.")
    else:
        print(f"\nOK: {pct:.0%} tests passed — prompt {args.new_version} looks good.")


if __name__ == "__main__":
    main()
