import os
from contextlib import contextmanager

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)


def _get_env(primary: str, fallback: str, default: str) -> str:
    return os.getenv(primary) or os.getenv(fallback, default)


@contextmanager
def get_conn():
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        conn = psycopg2.connect(dsn, connect_timeout=10)
    else:
        conn = psycopg2.connect(
            host=_get_env("DB_HOST", "POSTGRES_HOST", "localhost"),
            port=int(_get_env("DB_PORT", "POSTGRES_PORT", "5432")),
            dbname=_get_env("DB_NAME", "POSTGRES_DB", "classicmodels"),
            user=_get_env("DB_USER", "POSTGRES_USER", "postgres"),
            password=_get_env("DB_PASSWORD", "POSTGRES_PASSWORD", "postgres"),
            connect_timeout=10,
        )

    try:
        yield conn
    finally:
        conn.close()


def run_query(sql: str) -> list[dict]:
    sql_clean = (sql or "").strip()
    if not sql_clean.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql_clean)
            return [dict(row) for row in cur.fetchall()]
