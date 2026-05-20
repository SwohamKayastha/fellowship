import time
from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.executor import execute_query
from agents.planner import plan_query
from agents.sql_generator import generate_sql
from agents.summarizer import summarize_results
from agents.validator import validate_sql


class AgentState(TypedDict, total=False):
    user_query: str
    plan: dict
    decomposition: dict
    generated_sql: str
    is_valid_sql: bool
    execution_results: list[dict] | None
    final_answer: str
    summary: str
    errors: str | None
    attempts: int


def planner_node(state: AgentState) -> dict:
    return plan_query(state)


def generator_node(state: AgentState) -> dict:
    return generate_sql(state)


def validator_node(state: AgentState) -> dict:
    ok, error = validate_sql(state.get("generated_sql", ""))
    attempts = state.get("attempts", 0)
    if not ok:
        attempts += 1
    return {
        "is_valid_sql": ok,
        "errors": error if not ok else None,
        "attempts": attempts,
    }


def executor_node(state: AgentState) -> dict:
    return execute_query(state)


def summarizer_node(state: AgentState) -> dict:
    return summarize_results(state)


def route_after_validation(state: AgentState) -> str:
    if state.get("is_valid_sql"):
        return "executor"
    if state.get("attempts", 0) >= 2:
        return "summarizer"
    return "sql_generator"


def build_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("sql_generator", generator_node)
    graph.add_node("validator", validator_node)
    graph.add_node("executor", executor_node)
    graph.add_node("summarizer", summarizer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "sql_generator")
    graph.add_edge("sql_generator", "validator")
    graph.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "executor": "executor",
            "sql_generator": "sql_generator",
            "summarizer": "summarizer",
        },
    )
    graph.add_edge("executor", "summarizer")
    graph.add_edge("summarizer", END)

    return graph.compile()


workflow = build_workflow()


def run_workflow(user_query: str) -> dict:
    t0 = time.time()
    state: AgentState = {
        "user_query": user_query,
        "attempts": 0,
        "errors": None,
    }
    final_state = workflow.invoke(state)
    duration_ms = round((time.time() - t0) * 1000, 1)

    decomposition = final_state.get("decomposition") or final_state.get("plan") or {}
    result = final_state.get("execution_results")
    error = final_state.get("errors")
    summary = final_state.get("summary") or final_state.get("final_answer") or ""
    status = "success" if result is not None and not error else "failed"
    retry_count = max(0, int(final_state.get("attempts", 0)))

    return {
        "question": user_query,
        "decomposition": decomposition,
        "sql": final_state.get("generated_sql", ""),
        "result": result or [],
        "summary": summary,
        "status": status,
        "retry_count": retry_count,
        "duration_ms": duration_ms,
        "error": error,
    }
