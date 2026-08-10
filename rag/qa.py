"""Ask a question over the legal corpus."""

from __future__ import annotations

import json
import os
import sys

from rag.chunking import INDEX, get_embedder

COLLECTION = "legal_docs"
FALLBACK = INDEX / "fallback_index.json"


def _cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def retrieve_fallback(query: str, k: int) -> list[dict]:
    if not FALLBACK.exists():
        raise FileNotFoundError("run python -m rag.ingest first")
    data = json.loads(FALLBACK.read_text(encoding="utf-8"))
    embed, _ = get_embedder()
    q = embed(query)
    scored = []
    for row in data["chunks"]:
        scored.append((_cosine(q, row["vector"]), row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"text": row["text"], "source": row["source"], "score": round(score, 4)}
        for score, row in scored[:k]
    ]


def retrieve_chroma(query: str, k: int) -> list[dict]:
    import chromadb

    embed, _ = get_embedder()
    client = chromadb.PersistentClient(path=str(INDEX / "chroma"))
    col = client.get_collection(COLLECTION)
    res = col.query(query_embeddings=[embed(query)], n_results=k)
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({"text": doc, "source": meta.get("source"), "score": round(1 - dist, 4)})
    return out


def retrieve(query: str, k: int | None = None) -> list[dict]:
    k = k or int(os.getenv("TOP_K", "3"))
    if (INDEX / "chroma").exists() and os.getenv("USE_MOCK_EMBEDDINGS", "0") not in {"1", "true", "True"}:
        try:
            return retrieve_chroma(query, k)
        except Exception as exc:
            print(f"chroma retrieve failed ({exc}); trying fallback")
    return retrieve_fallback(query, k)


def answer(query: str) -> str:
    hits = retrieve(query)
    if not hits:
        return "No relevant passages found. Did you run ingest?"

    best = hits[0]
    lines = [
        f"Q: {query}",
        "",
        "Answer (grounded stub):",
        best["text"].strip(),
        "",
        "Sources:",
    ]
    for h in hits:
        lines.append(f"- {h['source']} (score={h['score']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print('usage: python -m rag.qa "your question"')
        return 2
    query = " ".join(argv)
    print(answer(query))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
