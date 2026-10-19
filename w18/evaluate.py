"""Run every labeled case, write results.md / results.json, print metrics. Exit code 0 always (known gaps are expected)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from guardrails.input_guards import check_injection, check_pii, classify_topic  # noqa: E402
from guardrails.llm import mock_llm  # noqa: E402
from guardrails.output_guards import check_groundedness, check_sensitive, validate_schema  # noqa: E402
from guardrails.pipeline import ALLOW_DOMAINS, run  # noqa: E402
from guardrails.reference import REFERENCE  # noqa: E402
from tests.cases import CASES, SCRIPTED_LLMS  # noqa: E402


def evaluate_case(c: dict) -> tuple[bool, str]:
    """Return (flagged, detail) for a guardrail case."""
    g, text = c["guardrail"], c["input"]
    if g == "pii":
        r = check_pii(text)
        return r.flagged, ",".join(sorted({f.kind for f in r.findings}))
    if g == "injection":
        r = check_injection(text)
        return r.flagged, ",".join(r.matches)
    if g == "topic":
        r = classify_topic(text)
        return r.category is not None, r.category or ""
    if g == "schema":
        r = validate_schema(text)
        return not r.valid, "; ".join(r.errors)
    if g == "grounded":
        r = check_groundedness(text, {"ref": c["reference"]})
        return r.flagged, f"unsupported={len(r.unsupported)}"
    if g == "sensitive":
        r = check_sensitive(text, ALLOW_DOMAINS)
        return r.flagged, ",".join(r.pii_kinds + r.secret_kinds)
    raise ValueError(g)


def evaluate_e2e(c: dict) -> tuple[bool, str]:
    seen = []

    def spy(q, ref, attempt=0):
        seen.append(q)
        base = mock_llm if c["llm"] == "mock" else SCRIPTED_LLMS[c["llm"]]
        return base(q, ref, attempt)

    r = run(c["input"], llm=spy)
    ok = r.status == c["status"]
    if "llm_must_not_see" in c:
        ok = ok and all(c["llm_must_not_see"] not in q for q in seen)
    if "response_contains" in c:
        ok = ok and c["response_contains"] in r.response
    return ok, f"status={r.status}"


def main() -> None:
    rows, counts = [], {}
    for c in CASES:
        if c["guardrail"] == "e2e":
            ok, actual = evaluate_e2e(c)
            expected = f"status={c['status']}"
        else:
            flagged, detail = evaluate_case(c)
            ok = flagged == c["flag"] and (not c["detail"] or not c["flag"] or c["detail"] in detail)
            expected = f"flag={c['flag']}" + (f" ({c['detail']})" if c["detail"] else "")
            actual = f"flag={flagged}" + (f" ({detail})" if detail else "")
            m = counts.setdefault(c["guardrail"], dict(tp=0, fp=0, fn=0, tn=0))
            m[("t" if flagged == c["flag"] else "f") + ("p" if flagged else "n")] += 1
        rows.append({"id": c["id"], "guardrail": c["guardrail"], "input": c["input"][:90],
                     "reference": c.get("reference", ""), "expected": expected, "actual": actual,
                     "pass": ok, "known_gap": c.get("known_gap", "")})

    def pr(m):
        p = m["tp"] / (m["tp"] + m["fp"]) if m["tp"] + m["fp"] else float("nan")
        r = m["tp"] / (m["tp"] + m["fn"]) if m["tp"] + m["fn"] else float("nan")
        return p, r

    total = dict(tp=0, fp=0, fn=0, tn=0)
    for m in counts.values():
        for k in total:
            total[k] += m[k]
    lines = ["# Evaluation results", "", f"{sum(r['pass'] for r in rows)}/{len(rows)} cases passed.", "",
             "## Metrics (flag = guardrail fires; ground truth = case label)", "",
             "| guardrail | TP | FP | FN | TN | precision | recall |", "|---|---|---|---|---|---|---|"]
    for name, m in [*counts.items(), ("overall", total)]:
        p, r = pr(m)
        lines.append(f"| {name} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} | {p:.2f} | {r:.2f} |")
    lines += ["", "## Cases", "", "| id | input | reference | expected | actual | result | note |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        esc = lambda s: s.replace("|", "\\|").replace("\n", " ")  # noqa: E731
        lines.append(f"| {r['id']} | {esc(r['input'])} | {esc(r['reference'][:40])} | {esc(r['expected'])} | "
                     f"{esc(r['actual'][:80])} | {'PASS' if r['pass'] else 'FAIL'} | {r['known_gap']} |")
    (ROOT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "results.json").write_text(json.dumps({"rows": rows, "metrics": counts}, indent=2), encoding="utf-8")

    print("\n".join(lines[:4 + len(counts) + 5]))
    for r in rows:
        if not r["pass"]:
            print(f"FAIL {r['id']}: expected {r['expected']}, got {r['actual']}  [{r['known_gap'] or 'UNEXPECTED'}]")


if __name__ == "__main__":
    main()
