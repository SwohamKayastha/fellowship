from openai import OpenAI

from config import get_settings

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        api_key = settings.llm_api_key or settings.openai_api_key or "local"
        _client = OpenAI(base_url=settings.llm_base_url, api_key=api_key)
    return _client


def call_llm(system: str, user: str, max_tokens: int = 600) -> str:
    settings = get_settings()
    client = get_client()
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    message = response.choices[0].message
    return (message.content or "").strip()
