"""
Runs main.py's full pipeline with the local and Fireworks backends mocked out,
so you can verify the I/O contract (tasks.json -> results.json), routing
decisions, and error handling WITHOUT needing the model file downloaded or
real Fireworks credentials.

Usage:
    python3 test/mock_test_run.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import local_backend
import fireworks_backend

# --- Mock the two backends -------------------------------------------------
def fake_local_generate(system_prompt, user_prompt, max_tokens=300):
    return f"[LOCAL MOCK] would answer: {user_prompt[:60]}..."

def fake_local_warm_up():
    return True

def fake_fireworks_generate(system_prompt, user_prompt, category, max_tokens=300, max_retries=2):
    return f"[FIREWORKS MOCK, category={category}] would answer: {user_prompt[:60]}..."

local_backend.generate = fake_local_generate
local_backend.warm_up = fake_local_warm_up
fireworks_backend.generate = fake_fireworks_generate

# --- Point main.py at a temp input/output pair ------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "practice_tasks.json")) as f:
    tasks = json.load(f)

tmpdir = tempfile.mkdtemp()
input_path = os.path.join(tmpdir, "tasks.json")
output_path = os.path.join(tmpdir, "results.json")
with open(input_path, "w") as f:
    json.dump(tasks, f)

os.environ["TASKS_INPUT_PATH"] = input_path
os.environ["RESULTS_OUTPUT_PATH"] = output_path

import main
rc = main.main()

print(f"\nexit code: {rc}")
with open(output_path) as f:
    results = json.load(f)

assert isinstance(results, list), "results.json must be a JSON array"
assert len(results) == len(tasks), "must return one result per task"
for r in results:
    assert "task_id" in r and "answer" in r, "each result needs task_id and answer"

print(f"OK: {len(results)} results written, schema valid.\n")
print(json.dumps(results, indent=2))

# --- Extra regression checks: classifier robustness + constraint enforcement ---
import categories

unseen_variants = [
    ("A car travels 60 km/h for 2.5 hours, then 40 km/h for 1 hour. What total distance did it cover?",
     "Mathematical reasoning"),
    ("Four people sit in a row; Al is not at either end; Bo is left of Cy; who is in the middle?",
     "Logical / deductive reasoning"),
    ("Give the key points of this article in bullet points.", "Text summarisation"),
    ("Is this review positive or negative: The food arrived cold but the staff were lovely.",
     "Sentiment classification"),
    ("Identify the people, organizations, and locations in: Elon Musk visited Tesla's Berlin plant.",
     "Named entity recognition"),
]
print("\n--- classifier robustness on unseen-style prompts ---")
failures = 0
for prompt, expected in unseen_variants:
    cat = categories.classify(prompt)
    ok = cat == expected
    failures += 0 if ok else 1
    print(f"{'OK' if ok else 'MISS':4s} got={cat!r:32s} expected={expected!r}")
assert failures == 0, f"{failures} classifier regression(s) — see above"

print("\n--- constraint enforcement ---")
one_sentence = categories.enforce_constraints(
    "Summarize in exactly one sentence: text.",
    "Sentence one. Sentence two. Sentence three.",
    categories.SUMMARY,
)
assert one_sentence.count(".") == 1, f"expected exactly one sentence, got: {one_sentence!r}"
print(f"OK: {one_sentence!r}")

five_words = categories.enforce_constraints(
    "Summarize in at most 5 words: text.",
    "one two three four five six seven",
    categories.SUMMARY,
)
assert len(five_words.rstrip(".").split()) == 5, f"expected 5 words, got: {five_words!r}"
print(f"OK: {five_words!r}")

print("\nAll checks passed.")
