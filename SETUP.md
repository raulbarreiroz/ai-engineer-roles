# Setup - Legal docs RAG (local)

## Idea

Chunk markdown contracts under `docs/`, embed them (sentence-transformers or a
deterministic mock), store in Chroma (or an in-memory FAISS-like store), then
answer questions with a retrieve + extractive/stub generator.

LoRA notes are in `notes/lora_finetune.md` - not required to run the demo.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m rag.ingest
python -m rag.qa "What is the notice period for termination?"
```

If sentence-transformers / chroma are heavy or fail to install:

```bash
USE_MOCK_EMBEDDINGS=1 python -m rag.ingest
USE_MOCK_EMBEDDINGS=1 python -m rag.qa "Who owns the intellectual property?"
```

## Layout

```
docs/           sample legal markdown
rag/            chunk -> embed -> retrieve -> answer
notes/          LoRA / PEFT pointers
index/          local chroma (gitignored)
```

## Env

| Variable | Default | Notes |
|---|---|---|
| `USE_MOCK_EMBEDDINGS` | `0` | hash-based vectors, no model download |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HF id |
| `TOP_K` | `3` | retrieved chunks |
