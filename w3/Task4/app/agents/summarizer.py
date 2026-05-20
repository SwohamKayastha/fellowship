import json
import re

from agents.llm import call_llm
from prompts import SUMMARIZER_SYSTEM, summarizer_user_prompt


def _guess_subject(question: str) -> str:
    if not question:
        return ""
    cleaned = question.strip().lower().rstrip("? .")
    patterns = [
        r"number of\s+(.+)",
        r"count of\s+(.+)",
        r"how many\s+(.+)",
        r"total\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, cleaned)
        if match:
            subject = match.group(1).strip()
            return subject
    return ""


def _fallback_summary(question: str, result: list[dict]) -> str:
    if not result:
        return "No rows matched."
    if len(result) == 1:
        row = result[0]
        if len(row) == 1:
            key = next(iter(row.keys()))
            value = row[key]
            subject = _guess_subject(question)
            key_lower = str(key).lower()
            if subject and isinstance(value, (int, float)):
                if any(token in key_lower for token in ["count", "total", "sum", "avg", "average", "min", "max"]):
                    return f"There are {value} {subject}."
                return f"The {subject} is {value}."
            return f"The answer is {value}."
        parts = [f"{k}={v}" for k, v in row.items()]
        return "One matching record: " + ", ".join(parts) + "."
    return f"Found {len(result)} matching records."


def summarize_results(state: dict) -> dict:
    question = state.get("user_query", "")
    sql = state.get("generated_sql", "")
    results = state.get("execution_results") or []
    error = state.get("errors")

    if error:
        summary = f"Query failed: {error}"
        return {"summary": summary, "final_answer": summary}

    row_count = len(results)
    if row_count <= 50:
        payload = results
        note = "Full result provided."
    else:
        payload = {"head": results[:5], "tail": results[-5:]}
        note = "Result truncated: head/tail provided."

    user = summarizer_user_prompt(
        question,
        sql,
        json.dumps(payload, default=str),
        note,
    )
    summary = call_llm(SUMMARIZER_SYSTEM, user, max_tokens=200).strip()
    if not summary:
        summary = _fallback_summary(question, results)

    return {"summary": summary, "final_answer": summary}
