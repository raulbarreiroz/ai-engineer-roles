"""Optional LLM client. Falls back to None when USE_LLM is off or requests fail."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def llm_enabled() -> bool:
    return os.getenv("USE_LLM", "0") in {"1", "true", "True"}


def complete(prompt: str, system: str = "") -> str | None:
    if not llm_enabled():
        return None

    base = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8001/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "local-model")
    key = os.getenv("OPENAI_API_KEY", "sk-local")

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system or "You are a careful Python coding assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]
    except (urllib.error.URLError, KeyError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"[llm] stubbing - request failed: {exc}")
        return None
