from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.sentiment import SentimentEngine

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sentiment-api")

engine: SentimentEngine | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global engine
    engine = SentimentEngine()
    log.info("ready - backend=%s", engine.backend)
    yield


app = FastAPI(
    title="Review Sentiment API",
    version="0.1.0",
    lifespan=lifespan,
)


class ReviewIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000, examples=["Loved the delivery speed."])


class SentimentOut(BaseModel):
    label: str
    score: float
    backend: str


@app.get("/health")
def health():
    return {"ok": True, "backend": engine.backend if engine else None}


@app.post("/sentiment", response_model=SentimentOut)
def sentiment(payload: ReviewIn):
    assert engine is not None
    result = engine.score(payload.text)
    return SentimentOut(label=result.label, score=result.score, backend=result.backend)
