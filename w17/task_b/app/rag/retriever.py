import logging
import time
from typing import Any

import chromadb
from app.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

_collection = None


def _connect() -> chromadb.Collection:
    """Create fresh ChromaDB client and return collection."""
    from app.config import settings

    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )
    # Ping to verify connection is alive
    client.heartbeat()
    return client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is not None:
        return _collection

    for attempt in range(1, 6):
        try:
            _collection = _connect()
            logger.info("Connected to ChromaDB.")
            return _collection
        except Exception as exc:
            logger.warning("ChromaDB connect attempt %d/5 failed: %s", attempt, exc)
            time.sleep(3)

    raise RuntimeError("Cannot connect to ChromaDB after 5 attempts.")


def _reset_collection() -> None:
    global _collection
    _collection = None


def add_documents(texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> int:
    col = get_collection()
    try:
        embeddings = get_embeddings(texts)
        base = col.count()
        ids = [f"doc_{base + i}" for i in range(len(texts))]
        col.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas or [{} for _ in texts],
            ids=ids,
        )
        return len(texts)
    except Exception as exc:
        _reset_collection()
        raise RuntimeError(f"ChromaDB add failed: {exc}") from exc


def retrieve(query: str, n_results: int = 3) -> list[dict[str, Any]]:
    try:
        col = get_collection()
        total = col.count()
        if total == 0:
            return []

        query_emb = get_embeddings([query])[0]
        results = col.query(
            query_embeddings=[query_emb],
            n_results=min(n_results, total),
        )
        return [
            {
                "content": doc,
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
            }
            for i, doc in enumerate(results["documents"][0])
        ]
    except Exception as exc:
        logger.error("ChromaDB retrieve failed: %s. Resetting connection.", exc)
        _reset_collection()
        return []  # degrade gracefully — chat continues without RAG
