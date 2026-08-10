"""Sentiment scoring helpers.

Tries a HF pipeline first; keyword fallback keeps demos working offline.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

POS_WORDS = {
    "good", "great", "excellent", "love", "amazing", "happy", "fast",
    "helpful", "recommend", "perfect", "awesome", "satisfied",
}
NEG_WORDS = {
    "bad", "terrible", "awful", "hate", "slow", "broken", "poor",
    "worst", "refund", "disappointed", "rude", "damaged",
}


@dataclass
class SentimentResult:
    label: str
    score: float
    backend: str


class SentimentEngine:
    def __init__(self) -> None:
        self._pipe = None
        self.backend = "keyword"
        self._neutral_band = float(os.getenv("NEUTRAL_BAND", "0.55"))

        if os.getenv("FORCE_KEYWORD", "0") in {"1", "true", "True"}:
            log.info("FORCE_KEYWORD set - skipping transformers load")
            return

        try:
            from transformers import pipeline

            model_id = os.getenv(
                "HF_MODEL",
                "distilbert-base-uncased-finetuned-sst-2-english",
            )
            # device=-1 keeps CPU; fine for a portfolio demo
            self._pipe = pipeline(
                "sentiment-analysis",
                model=model_id,
                device=-1,
            )
            self.backend = "transformers"
            log.info("loaded transformers pipeline: %s", model_id)
        except Exception as exc:  # noqa: BLE001 - intentional soft-fail
            log.warning("transformers unavailable (%s); using keyword fallback", exc)
            self._pipe = None
            self.backend = "keyword"

    def score(self, text: str) -> SentimentResult:
        text = (text or "").strip()
        if not text:
            return SentimentResult(label="neutral", score=0.0, backend=self.backend)

        if self._pipe is not None:
            return self._score_bert(text)
        return self._score_keywords(text)

    def _score_bert(self, text: str) -> SentimentResult:
        out = self._pipe(text[:512])[0]
        raw = out["label"].lower()
        score = float(out["score"])

        # SST-2 is binary; map low-confidence to neutral for the API contract
        if score < self._neutral_band:
            label = "neutral"
        elif "pos" in raw:
            label = "positive"
        else:
            label = "negative"

        return SentimentResult(label=label, score=round(score, 4), backend="transformers")

    def _score_keywords(self, text: str) -> SentimentResult:
        tokens = set(re.findall(r"[a-zA-Z']+", text.lower()))
        pos = len(tokens & POS_WORDS)
        neg = len(tokens & NEG_WORDS)

        if pos == neg:
            return SentimentResult(label="neutral", score=0.5, backend="keyword")
        if pos > neg:
            conf = min(0.95, 0.55 + 0.1 * (pos - neg))
            return SentimentResult(label="positive", score=round(conf, 4), backend="keyword")
        conf = min(0.95, 0.55 + 0.1 * (neg - pos))
        return SentimentResult(label="negative", score=round(conf, 4), backend="keyword")
