"""LLM backends. OpenAI when OPENAI_API_KEY is set, otherwise a deterministic offline mock."""
import json
import os

from .output_guards import _content_tokens

SYSTEM_PROMPT = (
    "You are Acme Cloud's support assistant. Answer ONLY from the reference sections provided. "
    "If the answer is not in them, say you don't have that information. "
    'Reply with JSON only: {"answer": str, "confidence": number 0-1, "sources": [section ids], '
    '"category": one of the section ids or "other"}.'
)
NO_INFO = "I'm sorry, I don't have information about that in our policy."


def mock_llm(question: str, reference: dict[str, str], attempt: int = 0) -> str:
    q = _content_tokens(question)
    best, score = None, 0
    for section, text in reference.items():
        s = len(q & _content_tokens(text))
        if s > score:
            best, score = section, s
    if best is None:
        return json.dumps({"answer": NO_INFO, "confidence": 0.2, "sources": [], "category": "other"})
    return json.dumps({"answer": reference[best], "confidence": 0.9, "sources": [best], "category": best})


def openai_llm(question: str, reference: dict[str, str], attempt: int = 0) -> str:
    from openai import OpenAI

    ctx = "\n".join(f"[{k}] {v}" for k, v in reference.items())
    resp = OpenAI().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Reference:\n{ctx}\n\nQuestion: {question}"},
        ],
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def get_llm():
    return openai_llm if os.getenv("OPENAI_API_KEY") else mock_llm
