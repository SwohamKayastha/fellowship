import re

BLOCKED = [
    r"\bDROP\b",
    r"\bDELETE\b",
    r"\bINSERT\b",
    r"\bUPDATE\b",
    r"\bTRUNCATE\b",
    r"\bALTER\b",
]


def validate_sql(sql: str) -> tuple[bool, str]:
    upper = (sql or "").upper().strip()
    if not upper.startswith("SELECT"):
        return False, "Must start with SELECT"

    for pattern in BLOCKED:
        match = re.search(pattern, upper)
        if match:
            return False, f"Blocked keyword: {match.group()}"

    return True, ""
