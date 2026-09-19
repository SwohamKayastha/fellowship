#!/usr/bin/env python3
"""
Evaluation harness for the W16 agentic self-check feature.

Measures:
  - Task completion rate
  - Tool-call correctness (rag_search + verify_answer both called)
  - Trajectory length (iterations per query)
  - Token usage per query
  - Failure log with taxonomy: hard / soft / cascading_soft

Usage:
  python eval/run_eval.py [--base-url http://localhost:8000] [--inject-failure]
  python eval/run_eval.py --output eval/results.md
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    # (id, query, expected_tools_subset, expect_verified, description)
    {
        "id": "TC01",
        "query": "What is this assistant about?",
        "expected_tools": ["rag_search", "verify_answer"],
        "expect_verified": True,
        "complexity": "simple",
        "description": "Basic KB query — expect 1 rag_search + 1 verify, pass on first try",
    },
    {
        "id": "TC02",
        "query": "Summarize the main topics in the knowledge base.",
        "expected_tools": ["rag_search", "verify_answer"],
        "expect_verified": True,
        "complexity": "simple",
        "description": "Broad KB query",
    },
    {
        "id": "TC03",
        "query": "What is the capital of France?",
        "expected_tools": ["rag_search"],
        "expect_verified": False,
        "complexity": "out-of-kb",
        "description": "Out-of-KB factual — model should acknowledge limitation, not hallucinate",
    },
    {
        "id": "TC04",
        "query": "Compare the advantages and disadvantages of RAG versus fine-tuning for LLMs.",
        "expected_tools": ["rag_search", "verify_answer"],
        "expect_verified": True,
        "complexity": "complex",
        "description": "Comparison query — may require multiple searches",
    },
    {
        "id": "TC05",
        "query": "What specific numbers or statistics are mentioned in the documents?",
        "expected_tools": ["rag_search", "verify_answer"],
        "expect_verified": True,
        "complexity": "specific",
        "description": "High-specificity query — verify likely fails once, triggers re-search",
    },
    {
        "id": "TC06",
        "query": "xzqwerty nonsense query that matches nothing",
        "expected_tools": ["rag_search"],
        "expect_verified": False,
        "complexity": "no-match",
        "description": "Guaranteed KB miss — agent should not fabricate",
    },
    {
        "id": "TC07",
        "query": "What are the key features of this AI system?",
        "expected_tools": ["rag_search", "verify_answer"],
        "expect_verified": True,
        "complexity": "simple",
        "description": "Feature query against KB content",
    },
    {
        "id": "TC08",
        "query": "Explain in detail how vector embeddings work and give 3 concrete examples from the documents.",
        "expected_tools": ["rag_search", "verify_answer"],
        "expect_verified": True,
        "complexity": "complex",
        "description": "Multi-part query requiring rich retrieval — likely 2+ searches",
    },
    {
        "id": "TC09",
        "query": "What is 2 + 2?",
        "expected_tools": ["calculator"],
        "expect_verified": False,
        "complexity": "simple",
        "description": "Math query — should use calculator, not rag_search",
    },
    {
        "id": "TC10",
        "query": "Give me a recommendation based on the documents: should I use this system for production?",
        "expected_tools": ["rag_search", "verify_answer"],
        "expect_verified": True,
        "complexity": "complex",
        "description": "Opinion/recommendation — model must ground in sources, not speculate",
    },
]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

FAILURE_TAXONOMY = {
    "hard": "Agent returned error / empty response / crashed",
    "soft": "Agent completed but answer unverified or hallucinated",
    "cascading_soft": "Verification failed, re-search also failed, final answer still wrong",
}


@dataclass
class EvalResult:
    id: str
    query: str
    complexity: str
    completed: bool
    tool_calls_correct: bool
    trajectory_length: int
    verified: bool
    tokens_input: int
    tokens_output: int
    tokens_total: int
    response_snippet: str
    failure_type: Optional[str] = None
    failure_detail: str = ""
    latency_ms: int = 0
    tools_used: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def call_agent(base_url: str, query: str, timeout: int = 60) -> dict:
    resp = requests.post(
        f"{base_url}/chat/agent",
        json={"query": query, "temperature": 0.1, "top_p": 0.9},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def classify_failure(result: EvalResult) -> Optional[str]:
    if not result.completed:
        return "hard"
    tools_used = set(result.tools_used)
    if "rag_search" in tools_used and not result.verified:
        # Check if there were multiple rag_search calls (re-search attempted but still failed)
        rag_calls = sum(1 for t in result.tools_used if t == "rag_search")
        if rag_calls > 1:
            return "cascading_soft"
        return "soft"
    return None


def evaluate_case(base_url: str, case: dict, inject_failure: bool = False) -> EvalResult:
    t0 = time.time()
    completed = False
    raw = {}
    error_detail = ""

    try:
        raw = call_agent(base_url, case["query"])
        completed = bool(raw.get("response", "").strip())
    except requests.exceptions.Timeout:
        error_detail = "Request timed out"
    except requests.exceptions.HTTPError as exc:
        error_detail = f"HTTP {exc.response.status_code}: {exc.response.text[:100]}"
    except Exception as exc:
        error_detail = str(exc)

    latency_ms = int((time.time() - t0) * 1000)

    tools_used = [tc["tool"] for tc in raw.get("tool_calls", [])]
    expected = set(case["expected_tools"])
    tools_correct = expected.issubset(set(tools_used)) if completed else False

    tokens = raw.get("tokens", {})
    response_text = raw.get("response", "")

    result = EvalResult(
        id=case["id"],
        query=case["query"],
        complexity=case["complexity"],
        completed=completed,
        tool_calls_correct=tools_correct,
        trajectory_length=raw.get("iterations", 0),
        verified=raw.get("verified", False),
        tokens_input=tokens.get("input", 0),
        tokens_output=tokens.get("output", 0),
        tokens_total=tokens.get("total", 0),
        response_snippet=response_text[:120].replace("\n", " "),
        failure_detail=error_detail,
        latency_ms=latency_ms,
        tools_used=tools_used,
    )
    result.failure_type = classify_failure(result)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_markdown(results: list[EvalResult], inject_failure: bool) -> str:
    lines = []
    lines.append(f"# W16 Agentic Eval Results\n")
    lines.append(f"Generated: {datetime.now().isoformat()}  ")
    lines.append(f"Failure injection: {'ON' if inject_failure else 'OFF'}  \n")

    # Summary stats
    total = len(results)
    completed = sum(1 for r in results if r.completed)
    tools_correct = sum(1 for r in results if r.tool_calls_correct)
    verified_count = sum(1 for r in results if r.verified)
    avg_traj = sum(r.trajectory_length for r in results) / total if total else 0
    total_tokens = sum(r.tokens_total for r in results)
    failures = [r for r in results if r.failure_type]

    lines.append("## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Task completion rate | {completed}/{total} ({100*completed//total}%) |")
    lines.append(f"| Tool-call correctness | {tools_correct}/{total} ({100*tools_correct//total}%) |")
    lines.append(f"| Verified answers | {verified_count}/{total} |")
    lines.append(f"| Avg trajectory length | {avg_traj:.1f} iterations |")
    lines.append(f"| Total tokens (all queries) | {total_tokens:,} |")
    lines.append(f"| Failures | {len(failures)} |")
    lines.append("")

    # Per-query table
    lines.append("## Per-Query Results\n")
    lines.append("| ID | Complexity | Done | Tools OK | Verified | Iter | Tokens | Failure | Latency |")
    lines.append("|----|-----------|------|----------|----------|------|--------|---------|---------|")
    for r in results:
        done = "✓" if r.completed else "✗"
        tok = "✓" if r.tool_calls_correct else "✗"
        ver = "✓" if r.verified else "✗"
        fail = r.failure_type or "—"
        lines.append(
            f"| {r.id} | {r.complexity} | {done} | {tok} | {ver} | "
            f"{r.trajectory_length} | {r.tokens_total:,} | {fail} | {r.latency_ms}ms |"
        )
    lines.append("")

    # Failure log
    if failures:
        lines.append("## Failure Log\n")
        for r in failures:
            lines.append(f"### {r.id} — {r.failure_type}")
            lines.append(f"**Query:** {r.query}  ")
            lines.append(f"**Tools used:** {r.tools_used}  ")
            lines.append(f"**Detail:** {r.failure_detail or 'See response snippet'}  ")
            lines.append(f"**Response:** {r.response_snippet}  ")
            lines.append(f"**Taxonomy:** {FAILURE_TAXONOMY.get(r.failure_type, '')}  \n")

    return "\n".join(lines)


def print_table(results: list[EvalResult]) -> None:
    header = f"{'ID':<6} {'Complexity':<12} {'Done':<5} {'Tools':<6} {'Verified':<9} {'Iter':<5} {'Tokens':<8} {'Failure':<16} {'ms'}"
    print(header)
    print("-" * len(header))
    for r in results:
        done = "Y" if r.completed else "N"
        tok = "Y" if r.tool_calls_correct else "N"
        ver = "Y" if r.verified else "N"
        fail = r.failure_type or "-"
        print(f"{r.id:<6} {r.complexity:<12} {done:<5} {tok:<6} {ver:<9} {r.trajectory_length:<5} {r.tokens_total:<8,} {fail:<16} {r.latency_ms}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="W16 agentic eval harness")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--inject-failure", action="store_true",
                        help="Signal server-side failure injection (documents manually)")
    parser.add_argument("--output", default="eval/results.md")
    parser.add_argument("--cases", nargs="*", help="Run specific case IDs e.g. TC01 TC03")
    args = parser.parse_args()

    cases = TEST_CASES
    if args.cases:
        cases = [c for c in TEST_CASES if c["id"] in args.cases]

    print(f"Running {len(cases)} test cases against {args.base_url}/chat/agent\n")

    results = []
    for case in cases:
        print(f"  [{case['id']}] {case['description']}...")
        result = evaluate_case(args.base_url, case, args.inject_failure)
        results.append(result)
        status = "✓" if result.completed else "✗"
        print(f"       {status} iter={result.trajectory_length} tokens={result.tokens_total} failure={result.failure_type or '-'}")

    print("\n" + "=" * 80)
    print_table(results)
    print("=" * 80)

    md = format_markdown(results, args.inject_failure)
    with open(args.output, "w") as f:
        f.write(md)
    print(f"\nResults written to {args.output}")

    # Exit non-zero if any hard failures
    hard_failures = sum(1 for r in results if r.failure_type == "hard")
    sys.exit(1 if hard_failures > 0 else 0)


if __name__ == "__main__":
    main()
