from sqlalchemy import text

from db import get_engine


def run_query(sql: str) -> list[dict]:
    sql_clean = (sql or "").strip()
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql_clean))
        rows = result.fetchall()
        return [dict(row._mapping) for row in rows]
