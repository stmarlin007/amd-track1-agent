"""
Local backend: runs a small quantized instruct model in-process via
llama-cpp-python. All calls here are FREE — they never touch
FIREWORKS_BASE_URL and never count toward the token score.

Model: Qwen2.5-3B-Instruct, Q4_K_M GGUF (~2.1GB on disk, fits comfortably
inside the 4GB RAM / 2 vCPU grading box alongside the Python process).
Swap MODEL_PATH / MODEL_URL in download_model.sh if you prefer a different
2B-3B instruct model — just keep it 4-bit quantized.
"""

import os
import sys
import threading

MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH", "/app/models/qwen2.5-3b-instruct-q4_k_m.gguf")

_llm = None
_lock = threading.Lock()


def _log(msg: str) -> None:
    print(f"[local_backend] {msg}", file=sys.stderr, flush=True)


def _get_llm():
    global _llm
    if _llm is not None:
        return _llm
    with _lock:
        if _llm is not None:
            return _llm
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Local model not found at {MODEL_PATH}. Did the Docker build run "
                f"download_model.sh? See README for how to bundle the model file."
            )
        from llama_cpp import Llama
        _log(f"Loading local model from {MODEL_PATH} ...")
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=int(os.environ.get("LOCAL_CTX", "4096")),
            n_threads=int(os.environ.get("LOCAL_THREADS", "2")),
            chat_format="chatml",  # Qwen2.5 instruct models use the ChatML template
            verbose=False,
        )
        _log("Local model loaded.")
        return _llm


def generate(system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
    """Run one chat completion locally. Returns plain text, never raises for
    model-quality reasons (a bad answer is still an answer); raises only on
    hard infra failures (missing model file) so the caller can fall back."""
    llm = _get_llm()
    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    text = result["choices"][0]["message"]["content"]
    return (text or "").strip()


_CLASSIFY_SYSTEM = (
    "You are a text classifier. Reply with ONLY one of these exact labels, nothing else:\n"
    "Factual knowledge | Mathematical reasoning | Sentiment classification | "
    "Text summarisation | Named entity recognition | Code debugging | "
    "Logical / deductive reasoning | Code generation"
)


def llm_classify(prompt: str):
    """Free, local second opinion for prompts the heuristic classifier
    wasn't confident about. Returns a category string, or None if the
    local model isn't available / didn't return a recognizable label —
    callers should keep the heuristic's default in that case."""
    import categories  # local import avoids a circular import at module load
    try:
        raw = generate(_CLASSIFY_SYSTEM, prompt, max_tokens=20).strip()
    except Exception as e:  # noqa: BLE001
        _log(f"llm_classify failed: {e}")
        return None
    for cat in categories.ALL_CATEGORIES:
        if cat.lower() in raw.lower():
            return cat
    return None


def is_low_quality(answer: str) -> bool:
    """Cheap heuristic to catch local-model answers that are effectively a
    non-answer (empty, truncated to nothing, or a refusal/confusion),
    signalling the caller should escalate this task to Fireworks instead of
    accepting a free-but-useless answer."""
    if not answer or len(answer.strip()) < 3:
        return True
    low = answer.strip().lower()
    bad_starts = (
        "i cannot", "i can't", "i'm not able", "i am not able",
        "as an ai", "i don't understand", "unclear question",
    )
    return any(low.startswith(b) for b in bad_starts)


def warm_up() -> bool:
    """Load the model once at startup so the first real task isn't slowed
    down by model load time (helps stay under the 60s ready / 30s per-request
    limits). Returns True on success, False if unavailable (caller should
    then route everything to Fireworks instead)."""
    try:
        _get_llm()
        return True
    except Exception as e:  # noqa: BLE001
        _log(f"warm_up failed, local backend unavailable: {e}")
        return False
