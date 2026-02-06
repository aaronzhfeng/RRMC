"""
Text processing utilities for RRMC.
====================================

Ported from ``_external/AR-Bench_Test/wrappers/inference_wrapper.py``
and ``_external/AR-Bench_Test/scripts/score_sp_baseline.py``.

Provides helper functions that may be needed when interfacing with
models that emit reasoning / thinking tags (e.g. Qwen3).
"""

import re
from typing import Optional


def strip_thinking_tags(text: str) -> str:
    """Strip Qwen3-style ``<think>…</think>`` blocks from model output.

    Parameters
    ----------
    text : str
        Raw model output that may contain ``<think>`` blocks.

    Returns
    -------
    str
        Cleaned text with thinking blocks removed.
    """
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def clean_prediction(text: str) -> str:
    """Clean a model prediction by removing think tags and answer prefixes.

    Parameters
    ----------
    text : str
        Raw prediction text.

    Returns
    -------
    str
        Cleaned prediction.
    """
    if text is None:
        return ""
    text = str(text)
    text = strip_thinking_tags(text)
    text = re.sub(r"^(A:|Answer:)\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def messages_to_chatml(
    messages: list,
    model: str = "",
) -> str:
    """Convert chat messages to a ChatML prompt string.

    Useful when running models in offline / raw-completion mode
    (e.g. vLLM offline with Qwen).

    Parameters
    ----------
    messages : list of dict
        ``[{"role": "user", "content": "…"}, …]``
    model : str, optional
        Model identifier (currently unused but kept for future routing).

    Returns
    -------
    str
        ChatML-formatted prompt.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)
