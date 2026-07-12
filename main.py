"""
Track 1 entrypoint.

Reads /input/tasks.json, answers every task, writes /output/results.json,
then exits 0. Designed to degrade gracefully:
  - If the local model can't load, everything routes to Fireworks instead
    of crashing (logged, not silent).
  - If a single task's backend call fails, that task gets a best-effort
    fallback answer rather than aborting the whole run (one bad task
    shouldn't zero out every other answer).
  - A soft overall time budget keeps the whole run inside the 10-minute
    hard limit even if the task set is larger than expected.

Two-phase execution:
  Phase 1 (sequential): classification + local-model answers. Sequential
  because there's a single shared local model instance — running it
  concurrently wouldn't be faster (CPU-bound, one model) and risks
  thread-safety issues in llama.cpp.
  Phase 2 (concurrent): every task that needs Fireworks is sent at the
  same time via a thread pool, since these are network calls — most of
  the wall-clock time is spent waiting on the API, not on CPU, so running
  them concurrently instead of one-by-one meaningfully cuts total runtime
  for task sets larger than the 8 we've tested with.
"""

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import categories
import local_backend
import fireworks_backend

INPUT_PATH = os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("RESULTS_OUTPUT_PATH", "/output/results.json")

OVERALL_BUDGET_SECONDS = 9 * 60  # leave a margin inside the 10-minute hard limit
MAX_PARALLEL_FIREWORKS_CALLS = 6  # concurrent requests; keeps well within a 30s-per-request limit


def log(msg: str) -> None:
    print(f"[main] {msg}", file=sys.stderr, flush=True)


def load_tasks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError("tasks.json must contain a JSON array")
    return tasks


def classify_and_maybe_answer_locally(task: dict, local_available: bool) -> dict:
    """Phase 1, run sequentially for each task. Returns a dict describing
    what's already been answered and what's still pending Fireworks."""
    task_id = task.get("task_id")
    prompt = task.get("prompt", "")
    category, confident = categories.classify(prompt, return_confidence=True)

    if not confident and local_available:
        refined = local_backend.llm_classify(prompt)
        if refined:
            log(f"task_id={task_id} heuristic was unsure "
                f"(defaulted to {category!r}); local tie-break -> {refined!r}")
            category = refined

    route = categories.route_for(category, prompt)
    system_prompt = categories.system_prompt_for(category)
    max_tokens = categories.max_tokens_for(category)
    log(f"task_id={task_id} category={category!r} route={route}")

    plan = {
        "task_id": task_id, "prompt": prompt, "category": category,
        "system_prompt": system_prompt, "max_tokens": max_tokens, "answer": None,
    }

    if route == categories.LOCAL_ROUTE and local_available:
        try:
            candidate = local_backend.generate(system_prompt, prompt, max_tokens=max_tokens)
            if local_backend.is_low_quality(candidate):
                log(f"local answer for {task_id} looked low-quality, escalating to Fireworks.")
            else:
                plan["answer"] = candidate
        except Exception as e:  # noqa: BLE001
            log(f"local backend failed for {task_id}, will try Fireworks: {e}")

    return plan


def run_fireworks_phase(pending: list, local_available: bool) -> None:
    """Phase 2: resolve every plan still missing an answer, concurrently."""
    if not pending:
        return

    def call_one(plan):
        try:
            return fireworks_backend.generate(
                plan["system_prompt"], plan["prompt"], plan["category"], max_tokens=plan["max_tokens"]
            )
        except Exception as e:  # noqa: BLE001
            log(f"fireworks backend failed for {plan['task_id']}: {e}")
            if local_available:
                try:
                    return local_backend.generate(plan["system_prompt"], plan["prompt"], max_tokens=plan["max_tokens"])
                except Exception as e2:  # noqa: BLE001
                    log(f"local fallback also failed for {plan['task_id']}: {e2}")
            return "Unable to generate an answer due to a backend error."

    workers = min(MAX_PARALLEL_FIREWORKS_CALLS, len(pending))
    log(f"Sending {len(pending)} task(s) to Fireworks concurrently ({workers} at a time).")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_plan = {pool.submit(call_one, plan): plan for plan in pending}
        for future in as_completed(future_to_plan):
            plan = future_to_plan[future]
            plan["answer"] = future.result()


def main() -> int:
    start = time.time()
    log("Starting Track 1 agent run.")

    try:
        tasks = load_tasks(INPUT_PATH)
    except Exception as e:  # noqa: BLE001
        log(f"FATAL: could not read {INPUT_PATH}: {e}")
        return 1

    log(f"Loaded {len(tasks)} task(s) from {INPUT_PATH}")

    local_available = local_backend.warm_up()
    if not local_available:
        log("Local model unavailable — ALL tasks will route to Fireworks.")

    # --- Phase 1: sequential classification + local-model attempts ---
    plans = []
    for task in tasks:
        elapsed = time.time() - start
        if elapsed > OVERALL_BUDGET_SECONDS * 0.6:
            # Reserve the back part of the budget for the Fireworks phase;
            # if phase 1 alone is already eating most of it, stop early.
            log(f"Time budget getting tight after phase 1 ({elapsed:.0f}s); "
                f"stopping classification early with {len(tasks) - len(plans)} task(s) left.")
            for remaining in tasks[len(plans):]:
                plans.append({
                    "task_id": remaining.get("task_id"), "prompt": remaining.get("prompt", ""),
                    "category": None, "system_prompt": None, "max_tokens": None,
                    "answer": "No answer produced: time budget exceeded before this task could run.",
                })
            break
        try:
            plans.append(classify_and_maybe_answer_locally(task, local_available))
        except Exception:  # noqa: BLE001
            log(f"Unexpected error on task {task.get('task_id')}:\n{traceback.format_exc()}")
            plans.append({
                "task_id": task.get("task_id"), "prompt": task.get("prompt", ""),
                "category": None, "system_prompt": None, "max_tokens": None,
                "answer": "Unable to generate an answer due to an unexpected error.",
            })

    # --- Phase 2: concurrent Fireworks calls for everything still pending ---
    pending = [p for p in plans if p["answer"] is None]
    try:
        run_fireworks_phase(pending, local_available)
    except Exception:  # noqa: BLE001
        log(f"Unexpected error during Fireworks phase:\n{traceback.format_exc()}")
        for p in pending:
            if p["answer"] is None:
                p["answer"] = "Unable to generate an answer due to an unexpected error."

    # --- Assemble final results, applying constraint enforcement ---
    results = []
    for p in plans:
        answer = p["answer"] or "No answer produced."
        if p["category"]:
            answer = categories.enforce_constraints(p["prompt"], answer, p["category"])
        results.append({"task_id": p["task_id"], "answer": answer})

    # Final safety net: guarantee schema validity even if something upstream
    # produced a weird value, so we never trade a good run for INVALID_RESULTS_SCHEMA.
    for r in results:
        if not isinstance(r.get("answer"), str) or not r["answer"].strip():
            r["answer"] = "No answer produced."
        r["task_id"] = str(r.get("task_id"))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    log(f"Wrote {len(results)} result(s) to {OUTPUT_PATH} in {time.time() - start:.1f}s total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
