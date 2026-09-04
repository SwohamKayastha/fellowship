import math
from datetime import datetime
from typing import Any

# OpenAI function-calling format — supported by vLLM + Qwen3
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate a mathematical expression. "
                "Supports arithmetic, trigonometry, logarithms, and constants from Python's math module."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Valid Python math expression, e.g. '2**10' or 'math.sqrt(144)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Return the current date and time in ISO 8601 format.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

_MATH_NS: dict[str, Any] = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_MATH_NS["math"] = math


def execute_tool(name: str, tool_input: dict[str, Any]) -> str:
    if name == "calculator":
        try:
            result = eval(tool_input["expression"], {"__builtins__": {}}, _MATH_NS)
            return str(result)
        except Exception as exc:
            return f"Error evaluating expression: {exc}"

    if name == "get_datetime":
        return datetime.now().isoformat()

    return f"Unknown tool: {name}"
