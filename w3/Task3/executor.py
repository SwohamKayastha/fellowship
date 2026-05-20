import json
import os
import time
from datetime import datetime, timezone

from database import run_query
from sql_generator import decompose_question, fix_sql, generate_sql
from validator import validate_sql

LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "query_logs.json")


def _preview_result(result: list[dict] | None) -> dict:
    if not result:
        return {"row_count": 0, "rows": []}
    if len(result) <= 50:
        return {"row_count": len(result), "rows": result}
    return {"row_count": len(result), "rows": {"head": result[:5], "tail": result[-5:]}}


def _log_execution(record: dict) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    entries = []
    if os.path.exists(LOG_PATH):
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as handle:
                entries = json.load(handle)
            if not isinstance(entries, list):
                entries = []
        except (json.JSONDecodeError, OSError):
            entries = []

    entries.append(record)
    with open(LOG_PATH, "w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2, default=str)


def run_pipeline(question: str) -> dict:
    if not question or not question.strip():
        response = {
            "question": question or "",
            "sql": "",
            "result": [],
            "status": "failed",
            "retry_count": 0,
            "error": "Question cannot be empty",
        }
        _log_execution(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "question": response["question"],
                "sql": response["sql"],
                "status": response["status"],
                "retry_count": response["retry_count"],
                "error": response["error"],
                "result_preview": {"row_count": 0, "rows": []},
            }
        )
        return response

    t0 = time.time()
    decomposition = decompose_question(question)
    sql = generate_sql(question, decomposition)

    result = None
    error = None
    retry_count = 0

    ok, reason = validate_sql(sql)
    if not ok:
        error = f"Blocked: {reason}"
    else:
        try:
            result = run_query(sql)
        except Exception as exc:
            error = str(exc)

    if result is None and error:
        retry_count = 1
        sql_retry = fix_sql(sql, error)
        ok_retry, reason_retry = validate_sql(sql_retry)
        if ok_retry:
            try:
                result = run_query(sql_retry)
                error = None
                sql = sql_retry
            except Exception as exc:
                error = str(exc)
                sql = sql_retry
        else:
            error = f"Blocked: {reason_retry}"
            sql = sql_retry

    status = "success" if result is not None else "failed"
    duration_ms = round((time.time() - t0) * 1000, 1)

    response = {
        "question": question,
        "sql": sql,
        "result": result or [],
        "status": status,
        "retry_count": retry_count,
        "error": error,
        "duration_ms": duration_ms,
    }

    _log_execution(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "sql": sql,
            "status": status,
            "retry_count": retry_count,
            "error": error,
            "duration_ms": duration_ms,
            "result_preview": _preview_result(result),
        }
    )

    return response
