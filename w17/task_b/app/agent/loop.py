"""
Agentic loop with MLflow trace capture.

Flow per request:
  1. Model calls rag_search(query) to retrieve context.
  2. Model drafts an answer and calls verify_answer(answer, sources).
  3. verify_answer makes an isolated sub-LLM call to check grounding.
  4. If verdict == pass: compact message history → model gives final answer.
  5. If verdict == fail: model re-searches or answers with caveat.
  6. Hard stop at MAX_ITERATIONS=5.

Each step is recorded as {step, tool, args, result, reasoning} for MLflow artifact logging.
"""

import json
import logging
import pathlib
import re
from typing import Any

from openai import OpenAI

from app.llm.tools import AGENT_TOOLS, execute_tool
from app.rag.retriever import retrieve

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5

VERIFY_SYSTEM_PROMPT = """You are a fact-checker. Given an answer and source documents, determine if the answer is supported by the sources.

Rules:
- verdict "pass": answer is substantially grounded in the sources OR correctly acknowledges uncertainty
- verdict "fail": answer makes specific claims that are absent from or contradicted by the sources

Respond ONLY with valid JSON, no markdown fences:
{"verdict": "pass" or "fail", "reason": "<one concise sentence>"}"""


def _load_prompt(version: str = "v1") -> str:
    prompt_file = pathlib.Path(__file__).parent.parent.parent / "prompts" / f"prompt_{version}.txt"
    if prompt_file.exists():
        return prompt_file.read_text()
    # Fallback inline
    return (
        "You are a careful AI assistant that verifies answers before responding.\n"
        "Call rag_search, then verify_answer. Never fabricate information."
    )


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _make_client() -> tuple[OpenAI, str]:
    from app.config import settings
    from openai import OpenAI as _OAI
    try:
        client = _OAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=settings.google_api_key,
        )
        return client, settings.gemini_model
    except Exception:
        client = _OAI(base_url=settings.llama_base_url, api_key=settings.llama_api_key)
        return client, settings.llama_model


def _execute_rag_search(query: str) -> tuple[str, list[dict]]:
    results = retrieve(query)
    if not results:
        return "No relevant documents found in the knowledge base.", []
    parts = []
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        source_label = meta.get("source", "unknown") if meta else "unknown"
        parts.append(f"[Source {i} — {source_label}]\n{r['content']}")
    return "\n\n".join(parts), results


def _execute_verify_answer(answer: str, sources: str, client: OpenAI, model: str) -> tuple[dict, int, int]:
    verify_prompt = f"Sources:\n{sources}\n\nCandidate answer:\n{answer}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
                {"role": "user", "content": verify_prompt},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        raw = _strip_think(resp.choices[0].message.content or "")
        raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
        in_tok = resp.usage.prompt_tokens if resp.usage else 0
        out_tok = resp.usage.completion_tokens if resp.usage else 0
        try:
            return json.loads(raw), in_tok, out_tok
        except json.JSONDecodeError:
            verdict = "pass" if "pass" in raw.lower() else "fail"
            return {"verdict": verdict, "reason": raw}, in_tok, out_tok
    except Exception as exc:
        logger.error("verify_answer sub-call failed: %s", exc)
        return {"verdict": "fail", "reason": f"Verification error: {exc}"}, 0, 0


def _compact_messages(user_query: str, verified_sources: str, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
        {
            "role": "assistant",
            "content": (
                f"I have retrieved and verified the following information:\n\n{verified_sources}\n\n"
                "The sources support my answer. Providing final response now."
            ),
        },
    ]


def run_agent_loop(
    user_query: str,
    temperature: float = 0.1,
    top_p: float = 0.9,
    prompt_version: str = "v1",
) -> dict[str, Any]:
    from app.config import settings

    system_prompt = _load_prompt(prompt_version)
    client, model = _make_client()

    messages: list[Any] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    iterations = 0
    tool_log: list[dict] = []
    trace: list[dict] = []
    verified = False
    all_sources: list[dict] = []
    last_retrieved_sources = ""
    total_input_tokens = 0
    total_output_tokens = 0
    termination_reason = "max_iterations"

    while iterations < MAX_ITERATIONS:
        iterations += 1
        logger.info("Agent iteration %d/%d", iterations, MAX_ITERATIONS)

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=AGENT_TOOLS,
                tool_choice="auto",
                max_tokens=settings.llama_max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as exc:
            logger.error("LLM call failed at iteration %d: %s", iterations, exc)
            termination_reason = f"llm_error: {exc}"
            break

        if resp.usage:
            total_input_tokens += resp.usage.prompt_tokens
            total_output_tokens += resp.usage.completion_tokens

        choice = resp.choices[0]

        if choice.finish_reason != "tool_calls":
            final_text = _strip_think(choice.message.content or "")
            termination_reason = "success" if verified else "no_tool_call"
            trace.append({
                "step": iterations,
                "tool": None,
                "args": None,
                "result": final_text[:300],
                "reasoning": "Model delivered final answer",
            })
            return {
                "response": final_text,
                "tool_calls": tool_log,
                "model_used": model,
                "iterations": iterations,
                "verified": verified,
                "sources": all_sources,
                "trace": trace,
                "termination_reason": termination_reason,
                "tokens": {
                    "input": total_input_tokens,
                    "output": total_output_tokens,
                    "total": total_input_tokens + total_output_tokens,
                },
            }

        assistant_msg = choice.message
        messages.append(assistant_msg)
        tool_results = []

        for tool_call in assistant_msg.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            step_record: dict[str, Any] = {
                "step": iterations,
                "tool": name,
                "args": args,
                "result": None,
                "reasoning": f"Model called {name}",
            }

            if name == "rag_search":
                result_str, raw_results = _execute_rag_search(args.get("query", ""))
                all_sources.extend(raw_results)
                last_retrieved_sources = result_str
                tool_result_content = result_str
                step_record["result"] = result_str[:300]

            elif name == "verify_answer":
                verdict_data, in_tok, out_tok = _execute_verify_answer(
                    args.get("answer", ""),
                    args.get("sources", last_retrieved_sources),
                    client,
                    model,
                )
                total_input_tokens += in_tok
                total_output_tokens += out_tok
                tool_result_content = json.dumps(verdict_data)
                step_record["result"] = verdict_data
                step_record["reasoning"] = (
                    f"Verification verdict: {verdict_data.get('verdict')} — {verdict_data.get('reason')}"
                )

                if verdict_data.get("verdict") == "pass":
                    verified = True
                    tool_log.append({"tool": name, "input": args, "result": tool_result_content})
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result_content,
                    })
                    trace.append(step_record)
                    messages = _compact_messages(user_query, last_retrieved_sources, system_prompt)
                    break

            else:
                tool_result_content = execute_tool(name, args)
                step_record["result"] = str(tool_result_content)[:300]

            trace.append(step_record)
            tool_log.append({"tool": name, "input": args, "result": tool_result_content})
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result_content,
            })

        if not verified:
            messages.extend(tool_results)

    last_content = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
            last_content = msg["content"]
            break
        elif hasattr(msg, "role") and msg.role == "assistant" and msg.content:
            last_content = _strip_think(msg.content)
            break

    return {
        "response": last_content or "Could not generate a verified answer within the iteration limit.",
        "tool_calls": tool_log,
        "model_used": model,
        "iterations": iterations,
        "verified": verified,
        "sources": all_sources,
        "trace": trace,
        "termination_reason": termination_reason,
        "tokens": {
            "input": total_input_tokens,
            "output": total_output_tokens,
            "total": total_input_tokens + total_output_tokens,
        },
    }
