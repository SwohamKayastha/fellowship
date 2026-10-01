from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Primary LLM — OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Fallback 1 — Gemini via OpenAI-compatible endpoint
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # Fallback 2 — local vLLM (http://vllm:8000/v1 inside compose, or Modal URL)
    llama_base_url: str = "http://vllm:8000/v1"
    llama_model: str = "Qwen/Qwen3-0.6B"
    llama_api_key: str = "none"
    llama_max_tokens: int = 768
    llama_temp: float = 0.1

    # Vector store
    chroma_host: str = "chromadb"
    chroma_port: int = 8000

    # Reliability
    max_retries: int = 3
    cache_ttl_seconds: int = 300

    # Feature flags
    enable_tools: bool = True
    cost_tracking_enabled: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
