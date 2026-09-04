from pydantic import BaseModel, Field
from typing import Any


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    use_rag: bool = True
    structured_output: bool = False
    temperature: float = Field(default=-1.0)  # -1 means use server default
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class Source(BaseModel):
    content: str
    metadata: dict[str, Any] = {}


class ChatResponse(BaseModel):
    response: str
    sources: list[Source] = []
    tool_calls: list[dict[str, Any]] = []
    model_used: str
    cached: bool = False


class IngestRequest(BaseModel):
    text: str
    metadata: dict[str, Any] = {}


class IngestResponse(BaseModel):
    status: str
    chunks_added: int


class HealthResponse(BaseModel):
    status: str
    primary_model: str
    fallback_model: str


class BatchChatRequest(BaseModel):
    requests: list[ChatRequest]
    max_concurrency: int = Field(default=5, ge=1, le=20)


class BatchChatResult(BaseModel):
    index: int
    response: ChatResponse | None = None
    error: str | None = None


class BatchChatResponse(BaseModel):
    results: list[BatchChatResult]
    total: int
    succeeded: int
    failed: int
