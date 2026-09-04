"""
FastAPI backend for the AI Assistant.

Endpoints:
  GET  /health        — liveness + model info
  POST /ingest        — add documents to the RAG knowledge base
  POST /chat          — chat with RAG + tool calling
  POST /chat/json     — same as /chat but instructs model to return JSON
  POST /chat/batch    — process multiple chat requests concurrently

Rate limits: 10 req/min on /chat, 5 req/min on /ingest, 3 req/min on /chat/batch.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.llm.client import chat
from app.models import (
    BatchChatRequest,
    BatchChatResponse,
    BatchChatResult,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    Source,
)
from app.rag.embeddings import get_model
from app.rag.ingestion import chunk_text
from app.rag.retriever import add_documents, retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up embedding model so first request isn't slow
    logger.info("Warming up embedding model...")
    get_model()
    logger.info("Ready.")
    yield


app = FastAPI(
    title="AI Assistant API",
    description="RAG-powered assistant backed by Qwen3-8B-AWQ on Modal + Gemini fallback.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    from app.config import settings

    return HealthResponse(
        status="ok",
        primary_model=settings.llama_model,
        fallback_model=settings.gemini_model,
    )


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


@app.post("/ingest", response_model=IngestResponse)
@limiter.limit("5/minute")
async def ingest(request: Request, body: IngestRequest):
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text field is empty.")

    try:
        chunks = chunk_text(body.text)
        metadatas = [body.metadata for _ in chunks]
        count = add_documents(chunks, metadatas)
        logger.info(f"Ingested {count} chunks.")
        return IngestResponse(status="success", chunks_added=count)
    except Exception as exc:
        logger.error(f"Ingest error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Document ingestion failed.")


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def _run_chat(body: ChatRequest, force_json: bool = False) -> ChatResponse:
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages list is empty.")

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    # Append JSON instruction when requested
    if force_json or body.structured_output:
        messages[-1]["content"] += "\n\nRespond with a single valid JSON object only."

    # RAG retrieval
    sources: list[Source] = []
    context = ""
    if body.use_rag:
        retrieved = retrieve(messages[-1]["content"])
        sources = [Source(content=r["content"], metadata=r["metadata"]) for r in retrieved]
        context = "\n\n".join(r["content"] for r in retrieved)

    try:
        result = chat(
            messages=messages,
            context=context,
            temperature=body.temperature,
            top_p=body.top_p,
        )
    except RuntimeError as exc:
        logger.error(f"LLM error: {exc}")
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error(f"Unexpected chat error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Chat request failed.")

    return ChatResponse(
        response=result["response"],
        sources=sources,
        tool_calls=result["tool_calls"],
        model_used=result["model_used"],
        cached=result.get("cached", False),
    )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, body: ChatRequest):
    return _run_chat(body)


@app.post("/chat/json", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat_json_endpoint(request: Request, body: ChatRequest):
    """Same as /chat but instructs model to return structured JSON."""
    return _run_chat(body, force_json=True)


# ---------------------------------------------------------------------------
# Batch chat — concurrent processing via asyncio.gather + thread pool
# ---------------------------------------------------------------------------


async def _run_chat_async(index: int, body: ChatRequest) -> BatchChatResult:
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(None, partial(_run_chat, body))
        return BatchChatResult(index=index, response=response)
    except HTTPException as exc:
        return BatchChatResult(index=index, error=exc.detail)
    except Exception as exc:
        logger.error(f"Batch item {index} failed: {exc}", exc_info=True)
        return BatchChatResult(index=index, error=str(exc))


@app.post("/chat/batch", response_model=BatchChatResponse)
@limiter.limit("3/minute")
async def chat_batch_endpoint(request: Request, body: BatchChatRequest):
    """
    Process multiple independent chat requests concurrently.
    Results are returned in the same order as the input requests.
    max_concurrency controls the semaphore (default 5, max 20).
    """
    if not body.requests:
        raise HTTPException(status_code=400, detail="requests list is empty.")
    if len(body.requests) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 requests per batch.")

    semaphore = asyncio.Semaphore(body.max_concurrency)

    async def _guarded(index: int, req: ChatRequest) -> BatchChatResult:
        async with semaphore:
            return await _run_chat_async(index, req)

    results: list[BatchChatResult] = await asyncio.gather(
        *[_guarded(i, req) for i, req in enumerate(body.requests)]
    )

    succeeded = sum(1 for r in results if r.error is None)
    return BatchChatResponse(
        results=results,
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
    )
