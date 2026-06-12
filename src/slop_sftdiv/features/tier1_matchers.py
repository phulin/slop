"""Tier-1 regex matchers for the Phase 1 frequency census.

The matchers in this module are intentionally lightweight and deterministic.
They are designed for first-pass corpus counting, not final precision claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping


TOKEN_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
SENTENCE_RE = re.compile(r"[.!?]+(?:\s+|$)")


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.UNICODE)


CONTRASTIVE_NEGATION_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "not_x_but_y": _compile(
        r"\b(?:is|are|was|were|be|being|been|it's|it is|this is|that is)?"
        r"\s*not\s+(?!only\b|just\b|merely\b|simply\b)"
        r"[^.!?\n;:]{1,120}?\s+but\s+[^.!?\n]{1,160}"
    ),
    "not_just_but": _compile(
        r"\b(?:it'?s|it is|this is|that is|they are|we are|you are)?"
        r"\s*not\s+(?:just|merely|simply|only)\s+[^.!?\n;:]{1,140}?"
        r"\s+(?:but|but also)\s+[^.!?\n]{1,160}"
    ),
    "isnt_just_its": _compile(
        r"\b(?:it|this|that)\s+isn'?t\s+(?:just|merely|simply|only)\s+"
        r"[^.!?\n;:]{1,140}?\s*(?:--|—|-|;|,)?\s*"
        r"(?:it'?s|it is|this is|that is)\s+[^.!?\n]{1,160}"
    ),
}


SLOP_LEXICON: Mapping[str, re.Pattern[str]] = {
    "delve": _compile(r"\bdelv(?:e|es|ed|ing)\b"),
    "tapestry": _compile(r"\btapestr(?:y|ies)\b"),
    "testament_to": _compile(r"\btestament\s+to\b"),
    "important_to_note": _compile(r"\bit'?s\s+important\s+to\s+note\b"),
    "worth_noting": _compile(r"\b(?:it'?s\s+)?worth\s+noting\b"),
    "underscore": _compile(r"\bunderscor(?:e|es|ed|ing)\b"),
    "nuanced": _compile(r"\bnuanc(?:e|ed|es)\b"),
    "multifaceted": _compile(r"\bmultifaceted\b"),
    "intricate": _compile(r"\bintric(?:ate|ately|acies|acy)\b"),
    "robust": _compile(r"\brobust(?:ly|ness)?\b"),
    "seamless": _compile(r"\bseamless(?:ly)?\b"),
    "landscape": _compile(r"\blandscape\b"),
    "realm": _compile(r"\brealm\b"),
    "journey": _compile(r"\bjourney\b"),
    "navigate": _compile(r"\bnavigat(?:e|es|ed|ing|ion)\b"),
    "foster": _compile(r"\bfoster(?:s|ed|ing)?\b"),
    "elevate": _compile(r"\belevat(?:e|es|ed|ing|ion)\b"),
    "unlock": _compile(r"\bunlock(?:s|ed|ing)?\b"),
    "comprehensive": _compile(r"\bcomprehensive(?:ly)?\b"),
    "moreover": _compile(r"\bmoreover\b"),
    "ultimately": _compile(r"\bultimately\b"),
}


STRUCTURE_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "bullet_line": _compile(r"^\s*(?:[-*+]|[•‣])\s+\S"),
    "numbered_line": _compile(r"^\s*\d{1,3}[.)]\s+\S"),
    "header_line": _compile(r"^\s{0,3}#{1,6}\s+\S"),
    "bold_lead_in": _compile(r"^\s*(?:[-*+]\s+)?\*\*[^*\n]{1,80}\*\*\s*[:—-]\s*"),
}


STOCK_OPENERS: Mapping[str, re.Pattern[str]] = {
    "great_question": _compile(r"^\s*(?:great|excellent|good)\s+question[.!]?\b"),
    "certainly": _compile(r"^\s*certainly[,.!]\s+"),
    "sure": _compile(r"^\s*sure[,.!]\s+"),
    "happy_to_help": _compile(r"^\s*(?:i'?d\s+be\s+)?happy\s+to\s+help\b"),
    "here_are": _compile(r"^\s*here\s+(?:are|is)\s+"),
}


STOCK_CLOSERS: Mapping[str, re.Pattern[str]] = {
    "hope_this_helps": _compile(r"\bi\s+hope\s+(?:this|that)\s+helps[.!]?\s*$"),
    "in_conclusion": _compile(r"\bin\s+conclusion[,:]?\s+[^.!?]{0,180}[.!?]?\s*$"),
    "overall": _compile(r"\boverall[,:]?\s+[^.!?]{0,180}[.!?]?\s*$"),
    "to_summarize": _compile(r"\bto\s+summari[sz]e[,:]?\s+[^.!?]{0,180}[.!?]?\s*$"),
    "let_me_know": _compile(r"\blet\s+me\s+know\s+if\s+you(?:'d)?\s+(?:like|need|want)\b[^.!?]*[.!?]?\s*$"),
}


PUNCTUATION_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "em_dash": re.compile(r"—|--"),
    "colon": re.compile(r":"),
    "semicolon": re.compile(r";"),
    "ellipsis": re.compile(r"\.\.\.|…"),
    "exclamation": re.compile(r"!"),
    "question": re.compile(r"\?"),
}


RULE_OF_THREE_PATTERN = _compile(
    r"\b(?!and\b|or\b)(?:[a-z][\w'-]*)(?:\s+(?:of|the|a|an|to|for|with|in|on|"
    r"and|or|[a-z][\w'-]*)){0,3}"
    r"\s*,\s*"
    r"(?!and\b|or\b)(?:[a-z][\w'-]*)(?:\s+(?:of|the|a|an|to|for|with|in|on|"
    r"and|or|[a-z][\w'-]*)){0,3}"
    r"\s*,?\s+(?:and|or)\s+"
    r"(?!and\b|or\b)(?:[a-z][\w'-]*)(?:\s+(?:of|the|a|an|to|for|with|in|on|"
    r"and|or|[a-z][\w'-]*)){0,3}\b"
)


@dataclass(frozen=True)
class Tier1Features:
    """Counts and normalized helper values for one input text."""

    counts: dict[str, int]
    rates_per_1k_tokens: dict[str, float]
    item_counts: dict[str, dict[str, int]]
    helpers: dict[str, float | int | list[int]]


@dataclass(frozen=True)
class FeatureHit:
    """One regex-level feature hit for validation sampling."""

    feature: str
    subtype: str
    start: int
    end: int
    text: str
    context: str


def extract_tier1_features(text: str) -> Tier1Features:
    """Extract Tier-1 feature counts from one text.

    Args:
        text: Raw document or response text.

    Returns:
        A ``Tier1Features`` bundle with pooled feature counts, per-1k-token
        rates, per-item counts, and token/rhythm helper data.
    """

    token_count = count_tokens(text)
    sentence_count = max(1, len(SENTENCE_RE.findall(text)) or 1)
    denominator = max(1, token_count)

    item_counts = {
        "contrastive_negation": _count_patterns(text, CONTRASTIVE_NEGATION_PATTERNS),
        "slop_lexicon": _count_patterns(text, SLOP_LEXICON),
        "structure": _count_patterns(text, STRUCTURE_PATTERNS),
        "stock_openers": _count_patterns(text, STOCK_OPENERS),
        "stock_closers": _count_patterns(text, STOCK_CLOSERS),
        "punctuation": _count_patterns(text, PUNCTUATION_PATTERNS),
        "rule_of_three": {"approx_triple": len(RULE_OF_THREE_PATTERN.findall(text))},
    }

    paragraph_token_counts = _paragraph_token_counts(text)
    paragraph_mean = _mean(paragraph_token_counts)
    paragraph_stdev = _stdev(paragraph_token_counts, paragraph_mean)
    rhythm_uniformity = _rhythm_uniformity(paragraph_token_counts, paragraph_mean, paragraph_stdev)

    counts = {
        "contrastive_negation": sum(item_counts["contrastive_negation"].values()),
        "slop_lexicon": sum(item_counts["slop_lexicon"].values()),
        "list_header_bold_lead_in": sum(item_counts["structure"].values()),
        "stock_openers": sum(item_counts["stock_openers"].values()),
        "stock_closers": sum(item_counts["stock_closers"].values()),
        "punctuation_rhythm": sum(item_counts["punctuation"].values()),
        "rule_of_three_approx": sum(item_counts["rule_of_three"].values()),
    }
    counts["stock_openers_closers"] = counts["stock_openers"] + counts["stock_closers"]

    rates_per_1k_tokens = {
        name: count / denominator * 1000.0 for name, count in counts.items()
    }

    helpers: dict[str, float | int | list[int]] = {
        "token_count": token_count,
        "sentence_count": sentence_count,
        "paragraph_count": len(paragraph_token_counts),
        "paragraph_token_counts": paragraph_token_counts,
        "paragraph_token_mean": paragraph_mean,
        "paragraph_token_stdev": paragraph_stdev,
        "paragraph_rhythm_uniformity": rhythm_uniformity,
        "rule_of_three_per_sentence": counts["rule_of_three_approx"] / sentence_count,
    }

    for name, count in item_counts["punctuation"].items():
        helpers[f"{name}_per_1k_tokens"] = count / denominator * 1000.0

    return Tier1Features(
        counts=counts,
        rates_per_1k_tokens=rates_per_1k_tokens,
        item_counts=item_counts,
        helpers=helpers,
    )


def count_tokens(text: str) -> int:
    """Count simple word-like tokens for matcher normalization."""

    return len(TOKEN_RE.findall(text))


def iter_tier1_hits(text: str, *, context_chars: int = 120) -> list[FeatureHit]:
    """Return regex-level Tier-1 hits with bounded context for human validation."""

    hits: list[FeatureHit] = []
    for feature, patterns in _validation_patterns().items():
        for subtype, pattern in patterns.items():
            for match in pattern.finditer(text):
                start, end = match.span()
                if feature == "list_header_bold_lead_in":
                    start, end = _line_span(text, start=start, end=end)
                hits.append(
                    FeatureHit(
                        feature=feature,
                        subtype=subtype,
                        start=start,
                        end=end,
                        text=text[start:end],
                        context=_context_window(text, start=start, end=end, context_chars=context_chars),
                    )
                )
    return sorted(hits, key=lambda hit: (hit.start, hit.end, hit.feature, hit.subtype))


def _count_patterns(text: str, patterns: Mapping[str, re.Pattern[str]]) -> dict[str, int]:
    return {name: len(pattern.findall(text)) for name, pattern in patterns.items()}


def _validation_patterns() -> dict[str, Mapping[str, re.Pattern[str]]]:
    return {
        "contrastive_negation": CONTRASTIVE_NEGATION_PATTERNS,
        "slop_lexicon": SLOP_LEXICON,
        "list_header_bold_lead_in": STRUCTURE_PATTERNS,
        "stock_openers": STOCK_OPENERS,
        "stock_closers": STOCK_CLOSERS,
        "punctuation_rhythm": PUNCTUATION_PATTERNS,
        "rule_of_three_approx": {"approx_triple": RULE_OF_THREE_PATTERN},
    }


def _context_window(text: str, *, start: int, end: int, context_chars: int) -> str:
    left = max(0, start - context_chars)
    right = min(len(text), end + context_chars)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    return prefix + text[left:right] + suffix


def _line_span(text: str, *, start: int, end: int) -> tuple[int, int]:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return line_start, line_end


def _paragraph_token_counts(text: str) -> list[int]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]
    return [count_tokens(paragraph) for paragraph in paragraphs]


def _mean(values: list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _stdev(values: list[int], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _rhythm_uniformity(values: list[int], mean: float, stdev: float) -> float:
    if not values:
        return 0.0
    if mean <= 0.0:
        return 0.0
    return 1.0 / (1.0 + stdev / mean)
