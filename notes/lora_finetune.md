# LoRA / PEFT notes (optional path)

The RAG demo answers from retrieved chunks with a lightweight generator.
In production you'd often fine-tune a small instruct model on (question, citation, answer)
triples from your legal corpus.

Suggested stack (not wired here):

1. Collect ~500-2k Q/A pairs grounded in `docs/`.
2. Use Hugging Face PEFT LoRA on a 7B instruct base (or Unsloth for speed).
3. Rank targets: r=8..16, alpha=16..32, dropout=0.05 on q/v projections.
4. Merge adapters or serve adapter + base behind vLLM / TGI.
5. Keep the retriever frozen; LoRA mainly improves grounded phrasing and citation style.

This keeps retrieval quality separate from generation style - easier to debug.
