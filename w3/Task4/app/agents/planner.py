import json

from agents.llm import call_llm
from prompts import planner_user_prompt, PLANNER_SYSTEM


def _pick(mapping: dict, key: str, default):
    for k, value in mapping.items():
        if str(k).lower() == key.lower():
            return value
    return default


def _to_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def plan_query(state: dict) -> dict:
    question = state.get("user_query", "")
    raw = call_llm(PLANNER_SYSTEM, planner_user_prompt(question), max_tokens=500)
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        plan = json.loads(cleaned)
        if not isinstance(plan, dict):
            plan = {}
    except json.JSONDecodeError:
        plan = {}

    normalized = {
        "intent": _pick(plan, "intent", "unknown"),
        "tables": _to_list(_pick(plan, "tables", [])),
        "columns": _to_list(_pick(plan, "columns", [])),
        "filters": _to_list(_pick(plan, "filters", [])),
        "joins": _to_list(_pick(plan, "joins", [])),
    }

    return {"plan": normalized, "decomposition": normalized}
