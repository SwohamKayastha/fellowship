#!/usr/bin/env python3
"""
Evaluation runner with cost tracking and budget limits.

Usage:
  python eval_with_budget.py --max-tokens-budget 100000 --base-url http://localhost:8000
"""

import argparse
import sys
import subprocess

from app.llm.cost_tracker import reset_tracker, get_tracker


def main():
    parser = argparse.ArgumentParser(
        description="Run prompt experiments with token budget tracking"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for agent endpoint",
    )
    parser.add_argument(
        "--versions",
        nargs="*",
        default=["v1", "v2", "v3"],
        help="Prompt versions to test (default: v1 v2 v3)",
    )
    parser.add_argument(
        "--max-tokens-budget",
        type=int,
        default=0,
        help="Max tokens allowed (0=unlimited). Stop early if exceeded.",
    )

    args = parser.parse_args()

    # Initialize cost tracker with budget
    reset_tracker(budget_tokens=args.max_tokens_budget)
    tracker = get_tracker()

    print(f"Starting evaluation with budget: {args.max_tokens_budget} tokens")
    print(f"Target: {args.base_url}\n")

    # Run the experiments
    cmd = [
        "python",
        "eval/run_prompt_experiment.py",
        "--base-url",
        args.base_url,
        "--versions",
        *args.versions,
    ]

    try:
        result = subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")

    # Print summary
    print("\n" + "=" * 60)
    print(tracker.summary())
    print("=" * 60)

    if tracker.is_over_budget():
        print("❌ BUDGET EXCEEDED - partial results saved")
        sys.exit(1)
    else:
        print("✓ Completed within budget")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
