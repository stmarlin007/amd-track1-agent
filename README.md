# AMD Hackathon — Track 1 Submission (General-Purpose AI Agent)

A router + backend agent that answers all 8 required categories, keeping
Fireworks token usage as low as possible by handling the "easy" categories
with a bundled free local model and only paying for Fireworks on the
categories that genuinely need a stronger model.

## How it works

1. **Classify** (`categories.py`) — a fast, deterministic, zero-cost
   keyword/regex classifier assigns each task to one of the 8 categories.
   No model call, so this never costs tokens and never fails.
   - If nothing matched strongly (low confidence, defaulted to Factual),
     `main.py` asks the **local model** for a free second opinion
     (`local_backend.llm_classify`) before committing to a route — this
     catches unseen phrasing the regex list wasn't written for, at zero
     token cost.
2. **Route**:
   - **Local (free)** → Factual knowledge, Sentiment classification, Text
     summarisation, Named entity recognition. These run entirely on a
     bundled quantized Qwen2.5-3B-Instruct model via `llama-cpp-python`.
   - **Fireworks (counts toward score)** → Mathematical reasoning, Code
     debugging, Logical/deductive reasoning, Code generation. These are the
     categories where a 3B quantized model is known to be unreliable
     (multi-step arithmetic, constraint satisfaction, correct code), so
     accuracy is prioritized over token cost here.
   - **Quality-gated escalation**: even for a locally-routed task, if the
     local model's answer looks like a non-answer (empty, truncated, a
     refusal) — checked by `local_backend.is_low_quality` — the task is
     escalated to Fireworks instead of submitting a useless free answer.
     This spends a few extra tokens only when the free path actually failed.
3. **Post-process**: `categories.enforce_constraints` deterministically
   trims summaries to an explicit sentence/word limit the prompt asked for
   (e.g. "in exactly one sentence"), since models routinely drift by a
   sentence even when told not to, and the guide specifically grades
   summarisation on following the stated format/length constraint.
4. **Answer & write**: `main.py` reads `/input/tasks.json`, dispatches each
   task, and writes `/output/results.json` before exiting 0. If the local
   model fails to load, everything automatically falls back to Fireworks
   instead of crashing. If a single Fireworks call fails, that one task
   gets a graceful fallback answer rather than aborting the whole run. A
   final validation pass guarantees every result has a non-empty string
   answer and a string task_id before the file is written, so a stray bad
   value can't trigger `INVALID_RESULTS_SCHEMA`.

## Files

```
Dockerfile             Build recipe (linux/amd64)
download_model.sh      Fetches the local GGUF model at BUILD time
requirements.txt       llama-cpp-python + requests
main.py                Entrypoint / orchestration
categories.py          Classifier + routing table + prompts
local_backend.py       Local model wrapper (free)
fireworks_backend.py   Fireworks API wrapper (env-driven, never hardcoded)
test/practice_tasks.json   The 8 practice tasks from the guide
test/mock_test_run.py      End-to-end test with both backends mocked
```

## Local testing — no model file, no API key needed

Verifies the I/O contract, routing decisions, and error handling in seconds:

```bash
cd agent
python3 test/mock_test_run.py
```

You should see all 8 practice tasks routed as:
`local, fireworks, local, local, local, fireworks, fireworks, fireworks`
and a valid `results.json` schema printed at the end.

## Local testing — real local model, no Fireworks

```bash
pip install -r requirements.txt
bash download_model.sh                 # downloads ~2.1GB into ./models
LOCAL_MODEL_PATH=./models/qwen2.5-3b-instruct-q4_k_m.gguf \
TASKS_INPUT_PATH=test/practice_tasks.json \
RESULTS_OUTPUT_PATH=/tmp/results.json \
python3 main.py
cat /tmp/results.json
```

This will actually call Fireworks for the 4 "hard" categories too — set
dummy env vars first if you just want to see the local-only categories
work and let the other four fail gracefully:

```bash
export FIREWORKS_API_KEY=dummy
export FIREWORKS_BASE_URL=https://example.invalid
export ALLOWED_MODELS=dummy-model
```

## Local testing — full Docker build

```bash
docker buildx build --platform linux/amd64 -t amd-track1-agent:test --load .

docker run --rm \
  -v "$(pwd)/test/practice_tasks.json:/input/tasks.json:ro" \
  -v "$(pwd)/out:/output" \
  -e FIREWORKS_API_KEY="<your real key if you have one>" \
  -e FIREWORKS_BASE_URL="<harness base url, or a real Fireworks endpoint for testing>" \
  -e ALLOWED_MODELS="<comma,separated,model,ids>" \
  amd-track1-agent:test

cat out/results.json
```

Before launch day you won't have the real `FIREWORKS_BASE_URL` /
`ALLOWED_MODELS` — that's fine, point `FIREWORKS_BASE_URL` at the real
Fireworks API (`https://api.fireworks.ai/inference/v1`) with your own key and
any Fireworks model you can access, purely to confirm the plumbing works
end-to-end. Swap in the harness-published values for the actual submission —
don't hardcode them either way, they're always read from the environment.

## Building & pushing the real submission

```bash
docker login ghcr.io   # or Docker Hub
docker buildx build --platform linux/amd64 \
  -t ghcr.io/<your-username>/amd-track1-agent:latest \
  --push .
```

`--platform linux/amd64` is mandatory even if you're on Apple Silicon — the
grading VM is amd64 only, and this is the #1 cause of `PULL_ERROR`.

Check the pushed image size:
```bash
docker manifest inspect ghcr.io/<your-username>/amd-track1-agent:latest
```
Must stay under 10GB compressed (the current build is roughly 2.1GB model +
~1-1.5GB Python/llama.cpp layers — comfortable headroom).

## Mapping this build to the guide's troubleshooting table

| Status | Why it shouldn't happen here | What to double check anyway |
|---|---|---|
| `PULL_ERROR` | Dockerfile is amd64-only by default | Confirm you built with `--platform linux/amd64` and the registry is public |
| `RUNTIME_ERROR` | `main.py` catches per-task and fatal errors, returns 0/1 explicitly | Check container logs locally first: `docker run ... 2>&1 \| tee run.log` |
| `TIMEOUT` | 9-minute soft budget inside the 10-minute hard limit, per-request Fireworks timeout of 25s | If your real task set is much larger than 8 tasks, consider lowering `OVERALL_BUDGET_SECONDS` margin further |
| `OUTPUT_MISSING` | `main.py` always writes `results.json` in a `finally`-equivalent path before returning | N/A |
| `INVALID_RESULTS_SCHEMA` | Every result is `{task_id, answer}` | Confirmed via `test/mock_test_run.py` schema assertions |
| `MODEL_VIOLATION` | Model ID is only ever read from `ALLOWED_MODELS` at runtime, never hardcoded | Verify `pick_model()` in `fireworks_backend.py` still only reads from env if you edit it |
| `IMAGE_TOO_LARGE` | ~3-4GB expected total | Re-check with `docker manifest inspect` before submitting |
| `ACCURACY_GATE_FAILED` | Hard categories are routed to Fireworks by design | If this still fails, it's the model/prompt quality — see "Tuning" below |

## Tuning for accuracy vs. tokens (once the plumbing works)

- **If you're not clearing the accuracy gate**: move more categories from
  local to Fireworks in `categories.ROUTE_MAP` (e.g. try NER or summarisation
  on Fireworks too) until you clear the gate, then optimize tokens back down
  from there — the guide explicitly recommends nailing accuracy first.
- **If you're comfortably over the accuracy gate and want fewer tokens**:
  trim `MAX_TOKENS` per category in `categories.py`, and/or move
  Named entity recognition or Text summarisation back to local if the 3B
  model handles your own test prompts well.
- **Do not** hardcode any specific answers or try to special-case the
  practice prompts — the guide states evaluation uses unseen variants, and
  the LLM-Judge is checking genuine correctness, not pattern matches.

## About the AI-Studio prototype in your original zip

The `amd-ai-agent-suite` app you already had (React + Express + Gemini) is a
**simulator/playground** — it uses Gemini to *pretend* to be the router and
the two model tiers, so it can't be submitted as-is: it never reads
`/input/tasks.json` or writes `/output/results.json`, and it calls Gemini
instead of a real local model + Fireworks. It's genuinely useful for
demoing your routing *idea* to teammates, but this `agent/` folder is the
actual submittable artifact. If it's helpful, the category list and
local/Fireworks split in `categories.py` mirror the same routing logic your
prototype's `ROUTER_SYSTEM_INSTRUCTION` already encoded — just wired to real
inference instead of a Gemini simulation of it.
