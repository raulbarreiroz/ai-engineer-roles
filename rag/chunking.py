from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INDEX = ROOT / "index"


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    start: int


def chunk_markdown(text: str, source: str, size: int = 420, overlap: int = 80) -> list[Chunk]:
    parts = re.split(r"(?m)(?=^##\s)", text)
    chunks: list[Chunk] = []
    cursor = 0

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= size:
            chunks.append(Chunk(id=f"{source}:{cursor}", text=part, source=source, start=cursor))
            cursor += len(part)
            continue

        i = 0
        while i < len(part):
            piece = part[i : i + size]
            chunks.append(Chunk(id=f"{source}:{cursor + i}", text=piece, source=source, start=cursor + i))
            if i + size >= len(part):
                break
            i += max(1, size - overlap)

        cursor += len(part)

    return chunks


def load_docs(docs_dir: Path = DOCS) -> list[Chunk]:
    out: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        out.extend(chunk_markdown(path.read_text(encoding="utf-8"), source=path.name))
    return out


def mock_embed(text: str, dim: int = 64) -> list[float]:
    """Deterministic bag-of-hashes vector - good enough for local demos."""
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
        vec[(h // dim) % dim] += 0.5
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def get_embedder():
    if os.getenv("USE_MOCK_EMBEDDINGS", "0") in {"1", "true", "True"}:
        return mock_embed, "mock"

    try:
        from sentence_transformers import SentenceTransformer

        model_id = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer(model_id)

        def _embed(text: str) -> list[float]:
            return model.encode(text, normalize_embeddings=True).tolist()

        return _embed, model_id
    except Exception as exc:  # noqa: BLE001
        print(f"sentence-transformers unavailable ({exc}); using mock embeddings")
        return mock_embed, "mock"
