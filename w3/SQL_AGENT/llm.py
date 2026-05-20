import json
import logging
import os
import re

import requests
from dotenv import load_dotenv

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

LLAMA_URL = os.getenv("LLAMA_URL", "http://localhost:8080/v1/chat/completions")
log = logging.getLogger(__name__)

SCHEMA = """
Database: classicmodels (PostgreSQL)
CRITICAL: ALL column names are camelCase and MUST be double-quoted.
  Example: SELECT "productName", "buyPrice" FROM products;

Tables:
  productlines("productLine" PK, "textDescription", "htmlDescription", "image")
  products("productCode" PK, "productName", "productLine" FK, "productVendor",
           "quantityInStock", "buyPrice", "MSRP")
  offices("officeCode" PK, "city", "country", "phone", "addressLine1", "addressLine2",
          "state", "postalCode", "territory")
  employees("employeeNumber" PK, "firstName", "lastName", "officeCode" FK,
            "reportsTo" FK->self, "jobTitle", "email", "extension")
  customers("customerNumber" PK, "customerName", "city", "country",
            "salesRepEmployeeNumber" FK->employees, "creditLimit")
  payments("customerNumber" FK, "checkNumber", "paymentDate", "amount")
  orders("orderNumber" PK, "orderDate", "status", "customerNumber" FK,
         "requiredDate", "shippedDate", "comments")
  orderdetails("orderNumber" FK, "productCode" FK, "quantityOrdered", "priceEach")
"""


def _quote_column(col: str) -> str:
    col = col.strip()
    if not col:
        return col

    if "." in col:
        table, column = col.split(".", 1)
        return f"{table}.\"{column}\""

    return f"\"{col}\""


def _build_sql_from_decomp(decomp: dict) -> str:
    tables = decomp.get("tables") or []
    columns = decomp.get("columns") or []
    filters = decomp.get("filters") or []
    joins = decomp.get("joins") or []
    aggregations = decomp.get("aggregations") or []

    if not tables:
        return ""

    select_parts: list[str] = []
    for col in columns:
        select_parts.append(_quote_column(col))
    for agg in aggregations:
        select_parts.append(str(agg).strip())

    select_clause = ", ".join(select_parts) if select_parts else "*"
    base_table = tables[0]
    sql_parts = [f"SELECT {select_clause}", f"FROM {base_table}"]

    for join in joins:
        join_text = str(join)
        if "=" not in join_text or "." not in join_text:
            continue
        left, right = [side.strip() for side in join_text.split("=", 1)]
        left_table = left.split(".", 1)[0].strip()
        right_table = right.split(".", 1)[0].strip()
        join_table = right_table if right_table != base_table else left_table
        left_expr = _quote_column(left)
        right_expr = _quote_column(right)
        sql_parts.append(f"JOIN {join_table} ON {left_expr} = {right_expr}")

    if filters:
        sql_parts.append("WHERE " + " AND ".join(filters))

    return " ".join(sql_parts).strip() + ";"


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


def _call(system: str, user: str, max_tokens: int = 600) -> str:
    response = requests.post(
        LLAMA_URL,
        json={
            "model": "local",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
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
        log.warning("LLM response missing content: %s", payload)
        return ""

    return str(content).strip()


def decompose(question: str) -> dict:
    system = (
        "You are a SQL analyst. Break the question into components.\n"
        "Respond ONLY with valid JSON. No markdown, no explanation.\n"
        "Format: {\"intent\":\"...\",\"tables\":[],\"columns\":[],\"filters\":[],\"joins\":[]}"
    )
    raw = _call(system, f"Schema:\n{SCHEMA}\n\nQuestion: {question}")
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"intent": raw, "tables": [], "columns": [], "filters": [], "joins": []}


def generate_sql(question: str, decomp: dict) -> str:
    system = (
        "You are a PostgreSQL expert. Write only a SELECT query.\n"
        "Rules:\n"
        "1. ALL column names MUST be double-quoted: \"columnName\"\n"
        "2. Only SELECT -- never INSERT/UPDATE/DELETE/DROP\n"
        "3. End with a semicolon\n"
        "4. Return ONLY the SQL -- no explanation, no markdown\n\n"
        f"Schema:\n{SCHEMA}"
    )
    user = f"Question: {question}\n\nDecomposition:\n{json.dumps(decomp, indent=2)}"
    raw = _call(system, user)

    fenced = re.search(r"```(?:sql)?\s*([\s\S]+?)```", raw, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    select_match = re.search(r"(SELECT[\s\S]+?);?\s*$", raw, re.IGNORECASE)
    if select_match:
        return select_match.group(1).strip() + ";"

    fallback = _build_sql_from_decomp(decomp)
    if fallback:
        return fallback

    return raw.strip()


def fix_sql(sql: str, error: str) -> str:
    system = (
        "You are a PostgreSQL debugger.\n"
        "Fix the failing query. Return ONLY the corrected SQL.\n"
        "Rules: SELECT only, all column names double-quoted.\n"
        f"Schema:\n{SCHEMA}"
    )
    user = f"Failing SQL:\n{sql}\n\nError:\n{error}"
    raw = _call(system, user)
    fenced = re.search(r"```(?:sql)?\s*([\s\S]+?)```", raw, re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return raw.strip()


def summarise(question: str, sql: str, result: list[dict]) -> str:
    system = (
        "You are a data analyst. Write a natural language answer in 1-2 sentences. "
        "Sound like: 'There are 42 shipped orders from customers in USA.' "
        "Never say 'Retrieved N rows' or 'Top result'. "
        "If the result is a single aggregate (COUNT/SUM/AVG/etc.), state it directly. "
        "Use ONLY the provided result data. If there are no rows, say so."
    )

    row_count = len(result)
    columns = list(result[0].keys()) if result else []

    if row_count <= 50:
        data_payload = result
        data_note = "Full result provided."
    else:
        data_payload = {
            "head": result[:5],
            "tail": result[-5:],
        }
        data_note = "Result truncated: only head/tail provided."

    user = (
        f"Question: {question}\n"
        f"SQL: {sql}\n"
        f"Rows: {row_count}\n"
        f"Columns: {json.dumps(columns)}\n"
        f"Note: {data_note}\n"
        f"Result: {json.dumps(data_payload, default=str)}"
    )

    summary = _call(system, user, max_tokens=200).strip()
    if not summary:
        return _fallback_summary(question, result)
    return summary
