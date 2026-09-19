"""LLM client with Gemini primary and Qwen fallback."""

import hashlib
import json
import logging
import time
from typing import Any

from openai import APIConnectionError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.llm.tools import TOOLS, execute_tool

logger = logging.getLogger(__name__)
_RETRYABLE = (RateLimitError, APIConnectionError)

SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base and tools.
- When context from the knowledge base is provided, use it to answer accurately.
- Use tools when needed (calculations, current date/time).
- For structured JSON output requests, respond ONLY with valid JSON — no markdown fences."""

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


def _qwen_client() -> OpenAI:
    from app.config import settings
    return OpenAI(base_url=settings.llama_base_url, api_key=settings.llama_api_key)


def _gemini_client() -> OpenAI:
    from app.config import settings
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
            "content": f"[Context from knowledge base]\n{context}\n\n[User question]\n{last['content']}",
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
    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        stop=stop_after_attempt(3),
    )
    def _call(msgs: list) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if with_tools:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"
        return client.chat.completions.create(**kwargs)

    current = messages.copy()
    response = _call(current)
    tool_log: list[dict] = []

    while with_tools and response.choices[0].finish_reason == "tool_calls":
        assistant_msg = response.choices[0].message
        current.append(assistant_msg)
        tool_results: list[dict] = []
        for tool_call in assistant_msg.tool_calls:
            args = json.loads(tool_call.function.arguments)
            result = execute_tool(tool_call.function.name, args)
            tool_log.append({"tool": tool_call.function.name, "input": args, "result": result})
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })
        current.extend(tool_results)
        try:
            response = _call(current)
        except Exception as exc:
            logger.error("Tool continuation failed: %s", exc)
            break

    text = response.choices[0].message.content or ""
    # Strip Qwen3 thinking tokens — <think>...</think> — before returning
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
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
    ck = _cache_key(augmented, settings.gemini_model)
    if use_cache:
        cached = _get_cached(ck, settings.cache_ttl_seconds)
        if cached:
            logger.info("Response cache hit.")
            return {**cached, "cached": True}

    try:
        logger.info("Gemini primary starting: model=%s", settings.gemini_model)
        text, tool_log = _run_tool_loop(
            _gemini_client(), settings.gemini_model, augmented,
            settings.llama_max_tokens, temp, top_p, with_tools=settings.enable_tools,
        )
        model_used = settings.gemini_model
        logger.info("Gemini primary responded: %s", model_used)
    except Exception as gemini_err:
        logger.warning(
            "Gemini primary failed (%s). Qwen fallback configured=%s, model=%s.",
            gemini_err, bool(settings.llama_base_url), settings.llama_model,
        )
        if not settings.llama_base_url:
            logger.error("Qwen fallback skipped: LLAMA_BASE_URL is not configured.")
            raise RuntimeError(
                f"Gemini failed and LLAMA_BASE_URL not set. Gemini error: {gemini_err}"
            ) from gemini_err
        try:
            logger.info("Qwen fallback starting: model=%s", settings.llama_model)
            text, tool_log = _run_tool_loop(
                _qwen_client(), settings.llama_model, augmented,
                settings.llama_max_tokens, temp, top_p,
                with_tools=settings.enable_tools,
            )
            model_used = settings.llama_model
            logger.info("Qwen fallback succeeded: model=%s", model_used)
        except Exception as qwen_err:
            logger.exception("Qwen fallback failed: model=%s error=%s", settings.llama_model, qwen_err)
            raise RuntimeError(
                f"All models failed. Gemini: {gemini_err}. Qwen: {qwen_err}"
            ) from qwen_err

    result: dict[str, Any] = {
        "response": text,
        "tool_calls": tool_log,
        "model_used": model_used,
        "cached": False,
    }
    if use_cache:
        _set_cached(ck, result)
    return result