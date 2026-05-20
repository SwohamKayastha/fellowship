import argparse
import csv
import json
import os
import time

from dotenv import load_dotenv

from pipeline import run

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)


def load_questions(path: str) -> list[str]:
    questions: list[str] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            return questions

        if "question" in header:
            idx = header.index("question")
            for row in reader:
                if row and len(row) > idx:
                    questions.append(row[idx])
        else:
            for row in reader:
                if row:
                    questions.append(row[0])

    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark questions.")
    parser.add_argument(
        "--questions",
        default="benchmark_questions.csv",
        help="CSV file with questions (column: question)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("logs", "benchmark_results.json"),
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per question",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    questions = load_questions(args.questions)
    if not questions:
        raise SystemExit("No questions found in the CSV file.")

    results = []
    t0 = time.time()
    for q in questions:
        results.append(run(q, max_retries=args.max_retries))

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)

    total = len(results)
    ok = sum(1 for r in results if r.get("status") == "success")
    duration = round(time.time() - t0, 2)
    print(f"Saved {total} results to {args.out}")
    print(f"Success rate: {ok}/{total} ({ok/total:.2%})")
    print(f"Total runtime: {duration}s")


if __name__ == "__main__":
    main()
