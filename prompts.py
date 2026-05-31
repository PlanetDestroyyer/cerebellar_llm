"""
Evaluation prompts for cerebellar generation quality test.

Three tiers:
  ID  — Wikipedia-style text (close to WikiText-103 training distribution)
  OOD — Different domains: science, code, dialogue, math reasoning
  STRESS — Long context + topic shifts (maximum pressure on error correction)

The cerebellar module should help most on OOD and STRESS prompts —
these are where the base model's pattern-completion fails and
the Purkinje error correction signal should kick in.
"""

# ── In-distribution (Wikipedia style) ────────────────────────────────────────
ID_PROMPTS = [
    {
        "id": "id_history",
        "text": "The Roman Empire at its greatest extent encompassed territories from",
        "domain": "history",
        "expected": "factual, structured encyclopedic continuation",
    },
    {
        "id": "id_science",
        "text": "Photosynthesis is the process by which plants convert sunlight into",
        "domain": "science",
        "expected": "accurate biological description",
    },
    {
        "id": "id_biography",
        "text": "Marie Curie was born in Warsaw in 1867 and later became the first",
        "domain": "biography",
        "expected": "accurate biographical facts",
    },
]

# ── Out-of-distribution ───────────────────────────────────────────────────────
OOD_PROMPTS = [
    {
        "id": "ood_code_reasoning",
        "text": "To implement a binary search tree in Python, the key insight is that",
        "domain": "code",
        "expected": "coherent technical explanation, not hallucinated syntax",
    },
    {
        "id": "ood_math",
        "text": "The proof that there are infinitely many prime numbers proceeds by",
        "domain": "mathematics",
        "expected": "logically sound proof steps",
    },
    {
        "id": "ood_medical",
        "text": "The mechanism by which CRISPR-Cas9 edits DNA involves the guide RNA",
        "domain": "molecular biology",
        "expected": "accurate molecular mechanism description",
    },
    {
        "id": "ood_philosophy",
        "text": "Kant's categorical imperative differs from utilitarian ethics because",
        "domain": "philosophy",
        "expected": "philosophically coherent argument",
    },
    {
        "id": "ood_economics",
        "text": "The 2008 financial crisis was triggered by the collapse of mortgage-backed",
        "domain": "economics",
        "expected": "accurate causal chain, not confabulation",
    },
]

# ── Stress tests (topic drift + long context) ────────────────────────────────
STRESS_PROMPTS = [
    {
        "id": "stress_shift",
        "text": (
            "The history of computing began with mechanical calculators in the "
            "19th century. Charles Babbage designed the Difference Engine, which "
            "could tabulate polynomial functions. Ada Lovelace wrote what many "
            "consider the first algorithm. By the 1940s, electronic computers "
            "emerged. However, the deeper question this raises about consciousness is"
        ),
        "domain": "topic_shift",
        "expected": "handles the pivot to philosophy without hallucinating",
    },
    {
        "id": "stress_uncertainty",
        "text": (
            "While the exact cause remains debated among historians, the most "
            "plausible explanation for the fall of the Western Roman Empire involves"
        ),
        "domain": "uncertainty",
        "expected": "expresses appropriate uncertainty, does not overclaim",
    },
]

ALL_PROMPTS = ID_PROMPTS + OOD_PROMPTS + STRESS_PROMPTS
