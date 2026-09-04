"""
LLM client with:
  - Primary: Modal/vLLM → Qwen3-8B-AWQ (OpenAI-compatible)
  - Fallback: Google Gemini 2.0 Flash (OpenAI-compatible endpoint)
  - Agentic tool-use loop
  - Exponential-backoff retry via tenacity
  - In-memory response cache with TTL
"""

import json
import hashlib
import logging
import time
from typing import Any

from openai import OpenAI, RateLimitError, APIConnectionError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.llm.tools import TOOLS, execute_tool

logger = logging.getLogger(__name__)

_RETRYABLE = (RateLimitError, APIConnectionError)

SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base and tools.
- When context from the knowledge base is provided, use it to answer accurately.
- Use tools when needed (calculations, current date/time).
- For structured JSON output requests, respond ONLY with valid JSON — no markdown fences."""

# Simple in-memory cache: cache_key → {result, timestamp}
_cache: dict[str, dict] = {}


def _cache_key(messages: list, model: str) -> str:
    payload = json.dumps({"messages": messages, "model": model}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _get_cached(key: str, ttl: int) -> dict | None:
    entry = _cache.get(key)
    if entry and time.time() - entry["ts"] < ttl:
        return entry["data"]
    if key in _cache:
        del _cache[key]
    return None


def _set_cached(key: str, data: dict) -> None:
    _cache[key] = {"data": data, "ts": time.time()}


def _primary_client() -> OpenAI:
    from app.config import settings
    return OpenAI(base_url=settings.llama_base_url, api_key=settings.llama_api_key)


def _fallback_client() -> OpenAI:
    from app.config import settings
    # Gemini exposes an OpenAI-compatible REST endpoint
    return OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=settings.google_api_key,
    )


def _build_messages(messages: list[dict], context: str) -> list[dict]:
    result = [{"role": "system", "content": SYSTEM_PROMPT}]
    result.extend(messages)
    if context:
        last = result[-1]
        result[-1] = {
            **last,
            "content": (
                f"[Context from knowledge base]\n{context}\n\n"
                f"[User question]\n{last['content']}"
            ),
        }
    return result


def _run_tool_loop(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    top_p: float,
    with_tools: bool = True,
) -> tuple[str, list[dict]]:
    """Run chat completion with agentic tool loop. Returns (text, tool_calls_log)."""

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        stop=stop_after_attempt(3),
    )
    def _call(msgs: list) -> Any:
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if with_tools:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"
        return client.chat.completions.create(**kwargs)

    current = messages.copy()
    response = _call(current)
    tool_log: list[dict] = []

    while with_tools and response.choices[0].finish_reason == "tool_calls":
        assistant_msg = response.choices[0].message
        current.append(assistant_msg)  # SDK accepts message objects directly

        tool_results: list[dict] = []
        for tc in assistant_msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = execute_tool(tc.function.name, args)
            tool_log.append({"tool": tc.function.name, "input": args, "result": result})
            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        current.extend(tool_results)

        try:
            response = _call(current)
        except Exception as exc:
            logger.error(f"Tool continuation failed: {exc}")
            break

    text = response.choices[0].message.content or ""
    return text, tool_log


def chat(
    messages: list[dict],
    context: str = "",
    temperature: float = -1.0,
    top_p: float = 0.9,
    use_cache: bool = True,
) -> dict[str, Any]:
    from app.config import settings

    temp = settings.llama_temp if temperature < 0 else temperature
    augmented = _build_messages(messages, context)

    # Cache lookup
    ck = _cache_key(augmented, settings.llama_model)
    if use_cache:
        cached = _get_cached(ck, settings.cache_ttl_seconds)
        if cached:
            logger.info("Response cache hit.")
            return {**cached, "cached": True}

    # --- Primary: Modal/vLLM Qwen3 ---
    try:
        text, tool_log = _run_tool_loop(
            _primary_client(),
            settings.llama_model,
            augmented,
            settings.llama_max_tokens,
            temp,
            top_p,
            with_tools=settings.enable_tools,
        )
        model_used = settings.llama_model
        logger.info(f"Primary responded: {model_used}")
    except Exception as primary_err:
        logger.warning(f"Primary model failed ({primary_err}). Trying Gemini fallback.")

        if not settings.google_api_key:
            raise RuntimeError(
                f"Primary failed and GOOGLE_API_KEY not set. Primary error: {primary_err}"
            ) from primary_err

        try:
            # Gemini via OpenAI-compat — tool calling works but skip for reliability
            text, tool_log = _run_tool_loop(
                _fallback_client(),
                settings.gemini_model,
                augmented,
                settings.llama_max_tokens,
                temp,
                top_p,
                with_tools=False,
            )
            model_used = settings.gemini_model
            logger.info(f"Fallback responded: {model_used}")
        except Exception as fallback_err:
            raise RuntimeError(
                f"All models failed. Primary: {primary_err}. Fallback: {fallback_err}"
            ) from fallback_err

    result: dict[str, Any] = {
        "response": text,
        "tool_calls": tool_log,
        "model_used": model_used,
        "cached": False,
    }

    if use_cache:
        _set_cached(ck, result)

    return result
