"""
Fireworks backend: every call here goes through FIREWORKS_BASE_URL using the
harness-provided FIREWORKS_API_KEY, and only ever uses a model ID from
ALLOWED_MODELS (never hardcoded — calling anything else invalidates the
submission per the rules doc). These calls count toward the token score, so
prompts are kept short and max_tokens is capped per category.
"""

import os
import sys
import time
import requests

REQUEST_TIMEOUT_SECONDS = 25  # stay under the 30s per-request rule with margin


def _log(msg: str) -> None:
    print(f"[fireworks_backend] {msg}", file=sys.stderr, flush=True)


def _get_config():
    # .strip() guards against stray whitespace/newlines from shell line-wraps
    # or copy-paste — an API key with an embedded \n produces a cryptic
    # "Invalid header value" error that's easy to misread as an auth problem.
    api_key = (os.environ.get("FIREWORKS_API_KEY") or "").strip()
    base_url = (os.environ.get("FIREWORKS_BASE_URL") or "").strip()
    allowed = os.environ.get("ALLOWED_MODELS", "")
    models = [m.strip() for m in allowed.split(",") if m.strip()]
    if not api_key or not base_url or not models:
        raise RuntimeError(
            "Fireworks env not configured: need FIREWORKS_API_KEY, FIREWORKS_BASE_URL, "
            "and a non-empty ALLOWED_MODELS. (Expected — the harness injects these at "
            "evaluation time; for local dev, set them yourself in a .env file.)"
        )
    return api_key, base_url.rstrip("/"), models


def pick_model(models, category: str) -> str:
    """Pick a model ID from the published allow-list at runtime. Prefers a
    larger/more capable-looking model (by name heuristics) for the hardest
    categories, otherwise just uses the first published model. Never
    hardcodes a specific model ID."""
    if len(models) == 1:
        return models[0]

    def size_hint(name: str) -> int:
        low = name.lower()
        for token, val in [("70b", 70), ("72b", 72), ("34b", 34), ("32b", 32),
                           ("13b", 13), ("14b", 14), ("8b", 8), ("7b", 7), ("3b", 3)]:
            if token in low:
                return val
        return 0

    hardest = {"Mathematical reasoning", "Logical / deductive reasoning",
               "Code debugging", "Code generation"}
    if category in hardest:
        return max(models, key=size_hint)
    return min(models, key=size_hint)


def generate(system_prompt: str, user_prompt: str, category: str, max_tokens: int = 300,
             max_retries: int = 2) -> str:
    api_key, base_url, models = _get_config()
    model = pick_model(models, category)

    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            last_err = e
            _log(f"attempt {attempt + 1} failed for model={model}: {e}")
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Fireworks call failed after {max_retries + 1} attempts: {last_err}")
