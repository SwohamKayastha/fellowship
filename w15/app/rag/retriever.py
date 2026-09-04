import logging
import time
from typing import Any
import chromadb
from app.rag.embeddings import get_embeddings

logger = logging.getLogger(__name__)

_collection = None


def get_collection():
    global _collection
    if _collection is not None:
        return _collection

    from app.config import settings

    for attempt in range(1, 6):
        try:
            client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
            _collection = client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Connected to ChromaDB.")
            return _collection
        except Exception as exc:
            logger.warning(f"ChromaDB connect attempt {attempt}/5 failed: {exc}")
            time.sleep(3)

    raise RuntimeError("Cannot connect to ChromaDB after 5 attempts.")


def add_documents(texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> int:
    col = get_collection()
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


def retrieve(query: str, n_results: int = 3) -> list[dict[str, Any]]:
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
        {"content": doc, "metadata": results["metadatas"][0][i] if results.get("metadatas") else {}}
        for i, doc in enumerate(results["documents"][0])
    ]
