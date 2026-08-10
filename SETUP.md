# Setup - Sentiment API (FastAPI + optional BERT)

## What it does

`POST /sentiment` scores a review as positive / negative / neutral.

By default the app tries Hugging Face `transformers.pipeline("sentiment-analysis")`
(DistilBERT SST-2). If the model download fails or `FORCE_KEYWORD=1`, it falls back
to a keyword scorer so the API always boots.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# lightweight mode (no model download)
FORCE_KEYWORD=1 uvicorn app.main:app --reload --port 8000
```

With model (first start may download ~250MB):

```bash
uvicorn app.main:app --reload --port 8000
```

## Try it

```bash
curl -X POST http://127.0.0.1:8000/sentiment \
  -H "content-type: application/json" \
  -d "{\"text\": \"The packaging was damaged but support was great.\"}"
```

Open docs at http://127.0.0.1:8000/docs

## Env

| Variable | Default | Meaning |
|---|---|---|
| `FORCE_KEYWORD` | `0` | Force keyword fallback |
| `HF_MODEL` | `distilbert-base-uncased-finetuned-sst-2-english` | Pipeline model id |
| `NEUTRAL_BAND` | `0.55` | BERT scores below this become neutral |
