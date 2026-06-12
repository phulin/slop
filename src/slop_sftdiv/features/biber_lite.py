from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Mapping

from slop_sftdiv.features.tier1_matchers import count_tokens


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.UNICODE)


BIBER_LITE_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "biber_lite_first_person_pronouns": _compile(r"\b(?:i|me|my|mine|we|us|our|ours)\b"),
    "biber_lite_second_person_pronouns": _compile(r"\b(?:you|your|yours|yourself|yourselves)\b"),
    "biber_lite_third_person_pronouns": _compile(
        r"\b(?:he|him|his|she|her|hers|they|them|their|theirs)\b"
    ),
    "biber_lite_demonstratives": _compile(r"\b(?:this|that|these|those)\b"),
    "biber_lite_possibility_modals": _compile(r"\b(?:can|could|may|might)\b"),
    "biber_lite_necessity_modals": _compile(r"\b(?:must|should|ought)\b"),
    "biber_lite_prediction_modals": _compile(r"\b(?:will|would|shall)\b"),
    "biber_lite_hedges": _compile(r"\b(?:often|typically|usually|generally|perhaps|maybe)\b"),
    "biber_lite_amplifiers": _compile(r"\b(?:very|really|highly|extremely|completely|totally)\b"),
    "biber_lite_private_verbs": _compile(
        r"\b(?:think|thinks|thought|believe|believes|believed|know|knows|knew|"
        r"feel|feels|felt|understand|understands|understood)\b"
    ),
    "biber_lite_public_verbs": _compile(
        r"\b(?:say|says|said|tell|tells|told|report|reports|reported|"
        r"argue|argues|argued|claim|claims|claimed)\b"
    ),
    "biber_lite_suasive_verbs": _compile(
        r"\b(?:suggest|suggests|suggested|recommend|recommends|recommended|"
        r"propose|proposes|proposed|urge|urges|urged)\b"
    ),
    "biber_lite_nominalizations": _compile(
        r"\b[a-z][a-z'-]{3,}(?:tion|sion|ment|ness|ity|ance|ence|ism|ship)\b"
    ),
    "biber_lite_that_complements": _compile(r"\bthat\s+(?:the|a|an|this|these|it|they|we|you|he|she)\b"),
    "biber_lite_infinitives": _compile(r"\bto\s+[a-z][a-z'-]+\b"),
    "biber_lite_causal_subordinators": _compile(r"\b(?:because|since|therefore|thus|hence)\b"),
    "biber_lite_conditional_subordinators": _compile(r"\b(?:if|unless|provided that)\b"),
    "biber_lite_wh_questions": _compile(r"\b(?:what|why|how|when|where|who|which)\b[^.!?\n]*\?"),
    "biber_lite_passive_voice_approx": _compile(
        r"\b(?:is|are|was|were|be|been|being)\s+(?:[a-z][a-z'-]+ed|"
        r"known|made|given|taken|seen|found|used|built|shown|done)\b"
    ),
}


@dataclass(frozen=True)
class BiberLiteFeatures:
    counts: Counter[str]
    token_count: int
    rates_per_1k_tokens: dict[str, float]


def extract_biber_lite_features(text: str) -> BiberLiteFeatures:
    token_count = count_tokens(text)
    denominator = max(token_count, 1)
    counts = Counter(
        {feature: len(pattern.findall(text)) for feature, pattern in BIBER_LITE_PATTERNS.items()}
    )
    rates = {feature: count / denominator * 1000.0 for feature, count in counts.items()}
    return BiberLiteFeatures(counts=counts, token_count=token_count, rates_per_1k_tokens=rates)
