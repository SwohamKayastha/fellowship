from tools.db_tools import run_query


def execute_query(state: dict) -> dict:
    sql = state.get("generated_sql", "")
    try:
        results = run_query(sql)
        return {"execution_results": results, "errors": None}
    except Exception as exc:
        return {"execution_results": None, "errors": str(exc)}
