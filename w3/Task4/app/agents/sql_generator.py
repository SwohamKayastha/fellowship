import json
import re

from agents.llm import call_llm
from prompts import GENERATOR_SYSTEM, generator_user_prompt


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


def generate_sql(state: dict) -> dict:
    question = state.get("user_query", "")
    plan = state.get("plan", {})
    plan_json = json.dumps(plan, indent=2, default=str)
    user = generator_user_prompt(question, plan_json)
    raw = call_llm(GENERATOR_SYSTEM, user, max_tokens=800)
    return {"generated_sql": _extract_sql(raw)}
