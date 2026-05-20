import os
from dataclasses import dataclass

from dotenv import load_dotenv

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)


@dataclass(frozen=True)
class Settings:
    database_url: str
    llm_base_url: str
    llm_model: str
    llm_api_key: str | None
    openai_api_key: str | None
    gemini_api_key: str | None


def get_settings() -> Settings:
    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@db:5432/classicmodels",
    )
    llm_base_url = os.getenv("LLM_BASE_URL", "http://host.docker.internal:8080/v1")
    llm_model = os.getenv("LLM_MODEL", "local")
    llm_api_key = os.getenv("LLM_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    return Settings(
        database_url=database_url,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
    )
