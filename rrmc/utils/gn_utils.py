"""
Guessing Numbers (Bulls & Cows) utilities for RRMC.
====================================================

Ported from ``_external/AR-Bench_Test/arbench_old/utils/utils_gn.py``
and ``_external/AR-Bench_Test/src/pipeline/tasks/gn_evaluator.py``.

Provides deterministic game logic needed for:
- Computing bulls & cows feedback
- Generating / filtering the answer set (all 5040 valid numbers)
- Estimating information gain (entropy reduction) of candidate guesses

All functions are **pure** (no LLM calls) and fully self-contained.
"""

import math
import random
import re
from collections import Counter
from itertools import permutations
from typing import List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Core game logic
# ---------------------------------------------------------------------------

def compare_guess(
    guess: List[int], secret: List[int]
) -> Tuple[int, int, int]:
    """Compute (bulls, cows, score) for a guess against a secret.

    Parameters
    ----------
    guess, secret : list of int
        Each is a 4-element list of digits.

    Returns
    -------
    same_pos : int
        Digits in the correct position ("bulls").
    diff_pos : int
        Correct digits but in wrong position ("cows").
    score : int
        ``2 * same_pos + diff_pos`` (AR-Bench scoring).
    """
    if isinstance(guess, str):
        guess = [int(d) for d in guess]
    if isinstance(secret, str):
        secret = [int(d) for d in secret]
    if len(guess) != len(secret):
        raise ValueError("guess and secret must have the same length")

    same_pos = sum(g == s for g, s in zip(guess, secret))
    gc, sc = Counter(guess), Counter(secret)
    common = sum((gc & sc).values())
    diff_pos = common - same_pos
    return same_pos, diff_pos, 2 * same_pos + diff_pos


def extract_guess(text: str) -> Optional[str]:
    """Extract a 4-digit guess (unique digits) from text.

    Looks for ``Guess: XXXX`` first, then falls back to any 4-digit
    number with unique digits.

    Returns
    -------
    str or None
        The 4-digit string, or None if nothing valid found.
    """
    text = text.replace("[", "").replace("]", "")

    # "Guess: 1234"
    m = re.search(r"guess:\s*(\d{4})", text, re.IGNORECASE)
    if m and len(set(m.group(1))) == 4:
        return m.group(1)

    # Any 4-digit number with unique digits
    for n in re.findall(r"\d{4}", text):
        if len(set(n)) == 4:
            return n

    # Fallback: any 4-digit number
    nums = re.findall(r"\d{4}", text)
    return nums[-1] if nums else None


# ---------------------------------------------------------------------------
# Answer-set management
# ---------------------------------------------------------------------------

def generate_all_answers() -> List[str]:
    """Generate all 5040 valid 4-digit numbers with unique digits."""
    return [
        "".join(map(str, p))
        for p in permutations(range(10), 4)
    ]


def filter_remaining(
    answer_set: List[str],
    guess: str,
    bulls: int,
    cows: int,
) -> List[str]:
    """Filter the answer set to keep only numbers consistent with feedback.

    Parameters
    ----------
    answer_set : list of str
        Current possible answers (each a 4-digit string).
    guess : str
        The guess that was made.
    bulls, cows : int
        Feedback received for *guess*.

    Returns
    -------
    list of str
        Filtered answer set.
    """
    gl = [int(d) for d in guess]
    out = []
    for ans in answer_set:
        al = [int(d) for d in ans]
        b, c, _ = compare_guess(gl, al)
        if b == bulls and c == cows:
            out.append(ans)
    return out


# ---------------------------------------------------------------------------
# Entropy / information gain
# ---------------------------------------------------------------------------

def estimate_entropy_reduction(
    answer_set: List[str],
    guess: str,
) -> float:
    """Estimate the expected entropy reduction for a given guess.

    For each possible feedback outcome (bulls, cows), counts how many
    answers in the set would produce that feedback, then computes the
    expected remaining entropy.

    Returns
    -------
    float
        Expected entropy reduction (bits).  Higher is better.
    """
    if not answer_set:
        return 0.0

    gl = [int(d) for d in guess]
    total = len(answer_set)

    # current entropy
    current_entropy = math.log2(total) if total > 1 else 0.0

    # partition by feedback
    buckets: dict = {}
    for ans in answer_set:
        al = [int(d) for d in ans]
        b, c, _ = compare_guess(gl, al)
        key = (b, c)
        buckets[key] = buckets.get(key, 0) + 1

    # expected remaining entropy
    expected = 0.0
    for count in buckets.values():
        p = count / total
        if count > 1:
            expected += p * math.log2(count)
        # count == 1 → 0 entropy contribution

    return current_entropy - expected


def generate_random_number() -> List[int]:
    """Generate a random 4-digit number with unique digits."""
    digits = list(range(10))
    random.shuffle(digits)
    if digits[0] == 0:
        for i in range(1, 10):
            if digits[i] != 0:
                digits[0], digits[i] = digits[i], digits[0]
                break
    return digits[:4]
