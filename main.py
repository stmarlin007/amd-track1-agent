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
"""

import json
import os
import sys
import time
import traceback

import categories
import local_backend
import fireworks_backend

INPUT_PATH = os.environ.get("TASKS_INPUT_PATH", "/input/tasks.json")
OUTPUT_PATH = os.environ.get("RESULTS_OUTPUT_PATH", "/output/results.json")

OVERALL_BUDGET_SECONDS = 9 * 60  # leave a margin inside the 10-minute hard limit


def log(msg: str) -> None:
    print(f"[main] {msg}", file=sys.stderr, flush=True)


def load_tasks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list):
        raise ValueError("tasks.json must contain a JSON array")
    return tasks


def answer_one(task: dict, local_available: bool) -> str:
    prompt = task.get("prompt", "")
    category, confident = categories.classify(prompt, return_confidence=True)

    if not confident and local_available:
        # Free second opinion — the heuristic didn't strongly match anything,
        # so ask the local model to classify instead of silently defaulting
        # to Factual knowledge. Costs zero Fireworks tokens either way.
        refined = local_backend.llm_classify(prompt)
        if refined:
            log(f"task_id={task.get('task_id')} heuristic was unsure "
                f"(defaulted to {category!r}); local tie-break -> {refined!r}")
            category = refined

    route = categories.route_for(category, prompt)
    system_prompt = categories.system_prompt_for(category)
    max_tokens = categories.max_tokens_for(category)

    log(f"task_id={task.get('task_id')} category={category!r} route={route}")

    answer = None
    if route == categories.LOCAL_ROUTE and local_available:
        try:
            candidate = local_backend.generate(system_prompt, prompt, max_tokens=max_tokens)
            if local_backend.is_low_quality(candidate):
                log(f"local answer for {task.get('task_id')} looked low-quality, "
                    f"escalating to Fireworks instead of accepting it.")
            else:
                answer = candidate
        except Exception as e:  # noqa: BLE001
            log(f"local backend failed for {task.get('task_id')}, falling back to Fireworks: {e}")

    if answer is None:
        # Either the category is Fireworks-routed, local was unavailable/failed,
        # or the local answer didn't clear the quality bar.
        try:
            answer = fireworks_backend.generate(system_prompt, prompt, category, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001
            log(f"fireworks backend failed for {task.get('task_id')}: {e}")
            # Last-resort: try local even for a Fireworks-preferred category, so
            # we still emit *something* non-empty rather than nothing.
            if local_available:
                try:
                    answer = local_backend.generate(system_prompt, prompt, max_tokens=max_tokens)
                except Exception as e2:  # noqa: BLE001
                    log(f"local fallback also failed for {task.get('task_id')}: {e2}")
            if answer is None:
                answer = "Unable to generate an answer due to a backend error."

    return categories.enforce_constraints(prompt, answer, category)


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

    results = []
    for i, task in enumerate(tasks):
        elapsed = time.time() - start
        remaining_tasks = len(tasks) - i
        if elapsed > OVERALL_BUDGET_SECONDS:
            log(f"Time budget exceeded ({elapsed:.0f}s) with {remaining_tasks} task(s) left; "
                f"answering remaining tasks locally-only (fast) to finish within the limit.")
            for remaining in tasks[i:]:
                task_id = remaining.get("task_id")
                try:
                    if local_available:
                        ans = local_backend.generate(
                            categories.system_prompt_for(categories.classify(remaining.get("prompt", ""))),
                            remaining.get("prompt", ""),
                            max_tokens=200,
                        )
                    else:
                        ans = "No answer produced: time budget exceeded before this task could run."
                except Exception:  # noqa: BLE001
                    ans = "No answer produced: time budget exceeded before this task could run."
                results.append({"task_id": task_id, "answer": ans})
            break

        task_id = task.get("task_id")
        try:
            answer = answer_one(task, local_available)
        except Exception:  # noqa: BLE001
            log(f"Unexpected error on task {task_id}:\n{traceback.format_exc()}")
            answer = "Unable to generate an answer due to an unexpected error."
        results.append({"task_id": task_id, "answer": answer})

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
