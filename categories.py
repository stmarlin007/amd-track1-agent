"""
Category definitions, heuristic classifier, and per-category prompt templates
for the AMD Hackathon Track 1 general-purpose agent.

Design:
- 8 capability categories from the participant guide.
- A fast, free, deterministic heuristic classifier (regex/keyword based) picks
  a category for each incoming prompt. No model call, no tokens, near-zero
  latency, and fully reproducible.
- Each category is pre-assigned to a "route": LOCAL (bundled quantized model,
  zero score-cost) or FIREWORKS (premium API, counts toward token score).
  The split favors accuracy on tasks that quantized 3B models are known to be
  weak at (multi-step math, constraint logic, code) while keeping the more
  pattern-like tasks (factual lookups, sentiment, summarization, NER) local.
"""

import re

FACTUAL = "Factual knowledge"
MATH = "Mathematical reasoning"
SENTIMENT = "Sentiment classification"
SUMMARY = "Text summarisation"
NER = "Named entity recognition"
CODE_DEBUG = "Code debugging"
LOGIC = "Logical / deductive reasoning"
CODE_GEN = "Code generation"

ALL_CATEGORIES = [FACTUAL, MATH, SENTIMENT, SUMMARY, NER, CODE_DEBUG, LOGIC, CODE_GEN]

LOCAL_ROUTE = "local"
FIREWORKS_ROUTE = "fireworks"

# Which categories default to which backend.
ROUTE_MAP = {
    FACTUAL: LOCAL_ROUTE,
    SENTIMENT: LOCAL_ROUTE,
    SUMMARY: LOCAL_ROUTE,
    NER: LOCAL_ROUTE,
    MATH: FIREWORKS_ROUTE,
    CODE_DEBUG: FIREWORKS_ROUTE,
    LOGIC: FIREWORKS_ROUTE,
    CODE_GEN: FIREWORKS_ROUTE,
}


def classify(prompt: str, return_confidence: bool = False):
    """Deterministic, zero-cost heuristic classifier.

    Order matters: more specific / higher-precision signals are checked
    first so a prompt that could match two patterns lands on the category
    where getting it right matters most (e.g. code with a bug should hit
    CODE_DEBUG, not CODE_GEN).

    If return_confidence=True, returns (category, confident: bool).
    confident=False only when nothing matched and we fell through to the
    FACTUAL default — the caller can use that as a signal to get a free
    second opinion from the local model (see local_backend.llm_classify).
    """
    p = prompt.lower()

    def hit(category):
        return (category, True) if return_confidence else category

    # --- Code debugging: existing code + something wrong with it ---
    debug_signals = [
        "bug", "fix", "doesn't work", "does not work", "incorrect output",
        "raises an error", "throws an error", "traceback", "fails when",
        "find and fix", "what's wrong", "whats wrong", "off by one",
        "wrong output", "unexpected result", "crashes", "not returning",
        "isn't returning", "produces the wrong",
    ]
    has_code_block = bool(re.search(r"def\s+\w+\s*\(|class\s+\w+|\{|\};|function\s+\w+\s*\(", prompt))
    if has_code_block and any(s in p for s in debug_signals):
        return hit(CODE_DEBUG)

    # --- Code generation: asked to write a function/program ---
    codegen_signals = [
        "write a function", "write a python", "write a program", "implement a function",
        "write code", "write a script", "implement the following", "return a function",
        "write a class", "write a method", "write an algorithm", "code a function",
    ]
    if any(s in p for s in codegen_signals):
        return hit(CODE_GEN)
    if has_code_block and not any(s in p for s in debug_signals):
        # Code present but no "something's wrong" signal -> likely a codegen spec/snippet.
        if any(s in p for s in ["write", "implement", "create a function", "define a function"]):
            return hit(CODE_GEN)

    # --- Logical / deductive reasoning: constraint puzzles ---
    logic_signals = [
        r"\bowns\b", "each own", "different pet", "who owns", "which one is",
        "exactly one of", "if and only if", "must be true", "puzzle",
        "three friends", "seating arrangement", "who is telling the truth",
        "either .* or", "cannot both", "at most one", "no two", "everyone except",
        "which of the following must", "who is lying", "who is the liar",
        "arrange the", "in what order", "next to each other",
        "sit in a row", "sits in a row", "seated", "in the middle",
        "at either end", "left of", "right of", "in front of", "behind",
        "not at either end", "who is in position",
    ]
    if any(re.search(s, p) for s in logic_signals):
        return hit(LOGIC)

    # --- Mathematical reasoning: numbers + arithmetic/word-problem cues ---
    math_signals = [
        "%", "percent", "how many", "how much", "total cost", "average",
        "projection", "compound interest", "ratio of", "remainder",
        "sold", "items remain", "calculate", "how old", "years old",
        "increase by", "decrease by", "discount", "budget", "left over",
        "remaining", "per hour", "per day", "miles per", "km/h", "price of",
    ]
    has_digit = bool(re.search(r"\d", prompt))
    if has_digit and any(s in p for s in math_signals):
        return hit(MATH)

    # --- Named entity recognition ---
    ner_signals = [
        "named entities", "extract all entities", "extract entities",
        "label the entities", "person, org", "entity types", "and their types",
        "identify the people, organizations", "extract people, places",
    ]
    if any(s in p for s in ner_signals):
        return hit(NER)

    # --- Sentiment classification ---
    sentiment_signals = [
        "sentiment", "classify the sentiment", "positive, negative", "how does the reviewer feel",
        "review:", "tone of this", "positive or negative", "opinion expressed",
        "star rating", "is this review",
    ]
    if any(s in p for s in sentiment_signals):
        return hit(SENTIMENT)

    # --- Text summarisation ---
    summary_signals = [
        "summarise", "summarize", "condense", "tl;dr", "in one sentence",
        "in exactly", "shorten this", "give a summary", "key points",
        "bullet points", "briefly describe", "main takeaways", "abstract of",
    ]
    if any(s in p for s in summary_signals):
        return hit(SUMMARY)

    # Nothing matched — default to factual, but flag as low-confidence so
    # the caller can optionally get a free second opinion from the local model.
    return (FACTUAL, False) if return_confidence else FACTUAL


# --- Length/format constraint extraction (for summarisation compliance) ----

_SENTENCE_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def extract_sentence_limit(prompt: str):
    """Looks for 'in exactly one sentence' / 'in 2 sentences' style
    constraints. Returns an int or None."""
    p = prompt.lower()
    m = re.search(r"in (?:exactly )?(\d+|one|two|three|four|five|six|seven|eight|nine|ten) sentences?", p)
    if not m:
        return None
    token = m.group(1)
    return int(token) if token.isdigit() else _SENTENCE_WORDS.get(token)


def extract_word_limit(prompt: str):
    """Looks for 'in at most 50 words' / 'in 30 words or fewer' style caps.
    Returns an int or None. Only matches explicit maximum-style phrasing to
    avoid misfiring on unrelated numbers."""
    p = prompt.lower()
    m = re.search(r"(?:in |within )(?:at most |up to )?(\d+) words?(?: or (?:fewer|less))?", p)
    if m:
        return int(m.group(1))
    return None


def is_multi_part_question(prompt: str) -> bool:
    """Detects questions that ask for more than one distinct thing (e.g. 'what
    is X, and what Y is it near'). Observed in testing: the small local model
    reliably answers only the first part and drops the rest even when told
    explicitly not to. Rather than rely on prompt-tuning to fix that, these
    get routed to Fireworks instead, where the stronger model handles both
    parts correctly."""
    p = prompt.lower()
    if p.count("?") >= 2:
        return True
    multi_part_signals = [
        " and what ", " and how ", " and why ", " and where ", " and when ",
        " as well as ", ", and ",
    ]
    return any(s in p for s in multi_part_signals)


def route_for(category: str, prompt: str = "") -> str:
    if category == FACTUAL and prompt and is_multi_part_question(prompt):
        return FIREWORKS_ROUTE
    return ROUTE_MAP.get(category, LOCAL_ROUTE)


# --- Per-category system prompts -------------------------------------------
# Kept short deliberately: for FIREWORKS-routed categories every word here is
# counted as input tokens by the judging proxy. LOCAL prompts can be a little
# more generous since local tokens are free, but conciseness also helps a 3B
# model stay focused.

_COMMON_SUFFIX = " Answer directly. No preamble, no restating the question, no meta-commentary."

SYSTEM_PROMPTS = {
    FACTUAL: (
        "Answer the question accurately and concisely. If the question has multiple "
        "parts (e.g. asks for two different things), make sure you explicitly answer "
        "every part — do not skip or substitute one part for another." + _COMMON_SUFFIX
    ),
    SENTIMENT: (
        "Classify the sentiment (positive, negative, neutral, or mixed) and give a "
        "one-sentence justification." + _COMMON_SUFFIX
    ),
    SUMMARY: "Summarise the given text, following any length or format constraint exactly." + _COMMON_SUFFIX,
    NER: (
        "Extract every named entity from the text and label each with its type "
        "(Person, Organization, Location, Date, etc.) as a simple list." + _COMMON_SUFFIX
    ),
    MATH: (
        "Solve the math word problem step by step internally, then give the final "
        "numeric answer clearly. Show only the minimal necessary working, ending with "
        "the final answer on its own." + _COMMON_SUFFIX
    ),
    CODE_DEBUG: (
        "Find the bug in the given code and provide the corrected, working version. "
        "Briefly state what was wrong, then give the fixed code." + _COMMON_SUFFIX
    ),
    LOGIC: (
        "Solve the constraint-based logic puzzle. Work through each condition step by "
        "step FIRST, then state your final answer LAST, after the reasoning — never "
        "state the answer before you've verified it against every condition. If your "
        "reasoning contradicts an earlier guess, trust the reasoning and correct the "
        "final answer accordingly." + _COMMON_SUFFIX
    ),
    CODE_GEN: (
        "Write a correct, well-structured function implementing the specification. "
        "Handle edge cases explicitly mentioned in the spec. Return only the code and, "
        "if needed, a one-line explanation." + _COMMON_SUFFIX
    ),
}


def system_prompt_for(category: str) -> str:
    return SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS[FACTUAL])


# Max output tokens per category — capped to control the token score without
# truncating answers that genuinely need the room (code, logic proofs).
MAX_TOKENS = {
    FACTUAL: 200,
    SENTIMENT: 120,
    SUMMARY: 150,
    NER: 200,
    MATH: 350,
    CODE_DEBUG: 400,
    LOGIC: 350,
    CODE_GEN: 450,
}


def max_tokens_for(category: str) -> int:
    return MAX_TOKENS.get(category, 300)


# --- Deterministic post-processing --------------------------------------
# Free, zero-latency fixups applied after generation. These never call a
# model — they just trim output to match an explicit constraint the prompt
# asked for, since "text summarisation" is graded partly on following a
# specific format/length constraint exactly, and models routinely drift
# by a sentence or two even when told not to.

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def enforce_constraints(prompt: str, answer: str, category: str) -> str:
    if category != SUMMARY or not answer:
        return answer

    sentence_limit = extract_sentence_limit(prompt)
    if sentence_limit:
        parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(answer.strip()) if s.strip()]
        if len(parts) > sentence_limit:
            answer = " ".join(parts[:sentence_limit])

    word_limit = extract_word_limit(prompt)
    if word_limit:
        words = answer.split()
        if len(words) > word_limit:
            answer = " ".join(words[:word_limit])
            if answer and answer[-1] not in ".!?":
                answer += "."

    return answer
