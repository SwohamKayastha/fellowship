import json
import os
import re
from typing import Any

import requests
from dotenv import load_dotenv

from prompts import templates

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "local")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")


def _call_llm(system: str, user: str, max_tokens: int = 600) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    response = requests.post(LLM_BASE_URL, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    payload = response.json()

    content = None
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if choices and isinstance(choices, list):
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
            if content is None:
                content = first.get("text")

    if content is None:
        return ""

    return str(content).strip()


def _extract_sql(raw: str) -> str:
    if not raw:
        return ""

    fenced = re.search(r"```(?:sql)?\s*([\s\S]+?)```", raw, re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()

    select_match = re.search(r"(SELECT[\s\S]+?);?\s*$", raw, re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip() + ";"

    return raw.strip()


def _pick(mapping: dict, key: str, default: Any) -> Any:
    for k, value in mapping.items():
        if str(k).lower() == key.lower():
            return value
    return default


def decompose_question(question: str) -> dict:
    raw = _call_llm(
        templates.DECOMPOSE_SYSTEM,
        templates.decomposition_user_prompt(question),
        max_tokens=400,
    )
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            parsed = {}
    except json.JSONDecodeError:
        parsed = {}

    intent = _pick(parsed, "Intent", cleaned if cleaned else "")
    tables = _pick(parsed, "Tables", [])
    columns = _pick(parsed, "Columns", [])
    filters = _pick(parsed, "Filters", [])
    joins = _pick(parsed, "Joins", [])

    return {
        "Intent": intent,
        "Tables": tables if isinstance(tables, list) else [tables],
        "Columns": columns if isinstance(columns, list) else [columns],
        "Filters": filters if isinstance(filters, list) else [filters],
        "Joins": joins if isinstance(joins, list) else [joins],
    }


def generate_sql(question: str, decomposition: dict) -> str:
    user = templates.generation_user_prompt(
        question, json.dumps(decomposition, indent=2)
    )
    raw = _call_llm(templates.GENERATE_SYSTEM, user, max_tokens=800)
    return _extract_sql(raw)


def fix_sql(sql: str, error: str) -> str:
    user = templates.fix_user_prompt(sql, error)
    raw = _call_llm(templates.FIX_SYSTEM, user, max_tokens=600)
    return _extract_sql(raw)
