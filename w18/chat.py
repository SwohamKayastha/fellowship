"""CLI: `python chat.py "question"` for one shot, or no args for an interactive loop."""
import sys

from guardrails.pipeline import run


def show(q: str) -> None:
    r = run(q)
    for t in r.trace:
        print(f"  [{t}]")
    print(f"{r.status}: {r.response}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
    else:
        while (q := input("> ").strip()) not in ("", "quit", "exit"):
            show(q)
