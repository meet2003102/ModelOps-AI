"""Minimal Ollama client (local LLM). Uses only the standard library.

Settings (environment variables):
  OLLAMA_URL    default http://localhost:11434
  OLLAMA_MODEL  default qwen2.5:3b   (use qwen2.5:7b if the machine has ~8 GB free RAM)
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


class LLMUnavailable(RuntimeError):
    pass


def _post(path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(f"{OLLAMA_URL}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise LLMUnavailable(f"Ollama not reachable at {OLLAMA_URL}: {e}") from e


def status(timeout: float = 2.0) -> dict:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=timeout) as r:
            names = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception as e:  # noqa: BLE001
        return {"available": False, "model": OLLAMA_MODEL, "message": f"Ollama not running ({e.__class__.__name__})"}
    ok = any(n == OLLAMA_MODEL or n.startswith(OLLAMA_MODEL + ":") for n in names)
    return {"available": ok, "model": OLLAMA_MODEL, "installed_models": names,
            "message": "ready" if ok else f"run: ollama pull {OLLAMA_MODEL}"}


def chat(system: str, user: str, temperature: float = 0.1, timeout: float = 180.0) -> tuple[str, float]:
    """Returns (text, seconds). Low temperature: we want faithful summaries, not creativity."""
    t0 = time.time()
    out = _post("/api/chat", {"model": OLLAMA_MODEL, "stream": False,
                              "options": {"temperature": temperature, "num_ctx": 4096},
                              "messages": [{"role": "system", "content": system},
                                           {"role": "user", "content": user}]}, timeout)
    return out["message"]["content"].strip(), round(time.time() - t0, 2)
