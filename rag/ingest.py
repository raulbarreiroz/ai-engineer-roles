"""Build a local vector index from docs/.

Prefers Chroma; falls back to a JSON store that mimics FAISS cosine search.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from rag.chunking import INDEX, get_embedder, load_docs

COLLECTION = "legal_docs"


def _write_fallback(chunks, vectors, meta_path: Path) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunks": [
            {"id": c.id, "text": c.text, "source": c.source, "vector": vectors[i]}
            for i, c in enumerate(chunks)
        ]
    }
    meta_path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote fallback index -> {meta_path} ({len(chunks)} chunks)")


def ingest() -> None:
    chunks = load_docs()
    if not chunks:
        raise SystemExit("no docs found under docs/")

    embed, backend = get_embedder()
    vectors = [embed(c.text) for c in chunks]
    print(f"embedded {len(chunks)} chunks with {backend}")

    INDEX.mkdir(parents=True, exist_ok=True)
    fallback = INDEX / "fallback_index.json"

    if os.getenv("USE_MOCK_EMBEDDINGS", "0") in {"1", "true", "True"}:
        _write_fallback(chunks, vectors, fallback)
        return

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(INDEX / "chroma"))
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
        col.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source} for c in chunks],
            embeddings=vectors,
        )
        print(f"chroma collection '{COLLECTION}' ready at {INDEX / 'chroma'}")
    except Exception as exc:  # noqa: BLE001
        print(f"chroma unavailable ({exc}); writing fallback index")
        _write_fallback(chunks, vectors, fallback)


if __name__ == "__main__":
    ingest()
