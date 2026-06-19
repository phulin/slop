from __future__ import annotations

import re
from typing import Literal

from slop_sftdiv.features.tier1_matchers import TOKEN_RE


FeatureTextMode = Literal["raw", "final_answer"]


THINKING_BLOCK_RE = re.compile(
    r"<\s*(think|reasoning)\s*>.*?<\s*/\s*\1\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
THINKING_OPEN_RE = re.compile(r"<\s*(think|reasoning)\s*>", flags=re.IGNORECASE)
THINKING_FENCE_RE = re.compile(
    r"```(?:thinking|reasoning)\b.*?```",
    flags=re.IGNORECASE | re.DOTALL,
)
THINKING_FENCE_OPEN_RE = re.compile(r"```(?:thinking|reasoning)\b", flags=re.IGNORECASE)
CHAT_ROLE_LINE_RE = re.compile(
    r"(?im)^\s*(?:assistant|user|system)\s*$\n?",
)


def final_answer_text(text: str) -> str:
    """Remove common explicit reasoning traces while preserving visible answer text."""
    without_blocks = THINKING_BLOCK_RE.sub("", text)
    without_fences = THINKING_FENCE_RE.sub("", without_blocks)
    opening_match = THINKING_OPEN_RE.search(without_fences)
    if opening_match is not None:
        without_fences = without_fences[: opening_match.start()]
    fence_match = THINKING_FENCE_OPEN_RE.search(without_fences)
    if fence_match is not None:
        without_fences = without_fences[: fence_match.start()]
    return without_fences.strip()


def strip_chat_role_markers(text: str) -> str:
    """Remove standalone chat-role marker lines left by some generation backends."""
    return CHAT_ROLE_LINE_RE.sub("", text).strip()


def feature_text_for_mode(text: str, mode: FeatureTextMode) -> str:
    if mode == "raw":
        return text
    if mode == "final_answer":
        return final_answer_text(text)
    raise ValueError(f"unsupported feature text mode: {mode}")


def feature_text_token_count(
    *,
    raw_generated_tokens: int,
    feature_text: str,
    mode: FeatureTextMode,
) -> int:
    if mode == "raw":
        return max(1, raw_generated_tokens)
    return max(1, len(TOKEN_RE.findall(feature_text)))
