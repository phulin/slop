from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from slop_sftdiv.features.tier1_matchers import TOKEN_RE, iter_tier1_hits


BOUNDARY_RE = re.compile(r"(?:^|[.!?;:\n]\s+)")
NEUTRAL_CONTROL_PATTERNS: dict[str, re.Pattern[str]] = {
    "neutral_for_example": re.compile(r"\bfor\s+example\b", re.IGNORECASE | re.UNICODE),
    "neutral_such_as": re.compile(r"\bsuch\s+as\b", re.IGNORECASE | re.UNICODE),
    "neutral_in_particular": re.compile(r"\bin\s+particular\b", re.IGNORECASE | re.UNICODE),
    "neutral_as_a_result": re.compile(r"\bas\s+a\s+result\b", re.IGNORECASE | re.UNICODE),
}
NEUTRAL_COMMON_PATTERNS: dict[str, re.Pattern[str]] = {
    "neutral_common_the": re.compile(r"\bthe\b", re.IGNORECASE | re.UNICODE),
    "neutral_common_a": re.compile(r"\ba\b", re.IGNORECASE | re.UNICODE),
    "neutral_common_of_the": re.compile(r"\bof\s+the\b", re.IGNORECASE | re.UNICODE),
    "neutral_common_number_of": re.compile(r"\bnumber\s+of\b", re.IGNORECASE | re.UNICODE),
    "neutral_common_in_the": re.compile(r"\bin\s+the\b", re.IGNORECASE | re.UNICODE),
    "neutral_common_to_the": re.compile(r"\bto\s+the\b", re.IGNORECASE | re.UNICODE),
    "neutral_common_is_a": re.compile(r"\bis\s+a\b", re.IGNORECASE | re.UNICODE),
}
NEUTRAL_CONTROL_INITIATORS: dict[str, tuple[str, ...]] = {
    "neutral_for_example": ("for example",),
    "neutral_such_as": ("such as",),
    "neutral_in_particular": ("in particular",),
    "neutral_as_a_result": ("as a result",),
}
NEUTRAL_COMMON_INITIATORS: dict[str, tuple[str, ...]] = {
    "neutral_common_the": ("the",),
    "neutral_common_a": ("a",),
    "neutral_common_of_the": ("of the",),
    "neutral_common_number_of": ("number of",),
    "neutral_common_in_the": ("in the",),
    "neutral_common_to_the": ("to the",),
    "neutral_common_is_a": ("is a",),
}


@dataclass(frozen=True)
class OpportunitySpec:
    feature: str
    opportunity_kind: str
    initiators: tuple[str, ...]


@dataclass(frozen=True)
class FeatureOpportunity:
    feature: str
    opportunity_kind: str
    char_offset: int
    reference_initiates: bool
    matched_subtype: str | None = None


PHASE2_OPPORTUNITY_SPECS: dict[str, OpportunitySpec] = {
    "contrastive_negation": OpportunitySpec(
        feature="contrastive_negation",
        opportunity_kind="clause_boundary",
        initiators=(
            "not just",
            "not only",
            "it is not",
            "it's not",
            "this is not",
            "that is not",
        ),
    ),
    "slop_lexicon": OpportunitySpec(
        feature="slop_lexicon",
        opportunity_kind="token_start",
        initiators=(
            "delve",
            "tapestry",
            "testament",
            "important",
            "worth",
            "underscore",
            "nuanced",
            "multifaceted",
            "intricate",
            "robust",
            "seamless",
            "landscape",
            "realm",
            "journey",
            "navigate",
            "foster",
            "elevate",
            "unlock",
            "comprehensive",
            "moreover",
            "ultimately",
        ),
    ),
    "stock_openers": OpportunitySpec(
        feature="stock_openers",
        opportunity_kind="document_start",
        initiators=("great", "excellent", "good", "certainly", "sure", "happy", "here"),
    ),
    "stock_closers": OpportunitySpec(
        feature="stock_closers",
        opportunity_kind="final_clause_boundary",
        initiators=("i", "in", "overall", "to", "let"),
    ),
    "stock_openers_closers": OpportunitySpec(
        feature="stock_openers_closers",
        opportunity_kind="document_or_final_boundary",
        initiators=(
            "great",
            "excellent",
            "good",
            "certainly",
            "sure",
            "happy",
            "here",
            "i",
            "in",
            "overall",
            "to",
            "let",
        ),
    ),
    "neutral_for_example": OpportunitySpec(
        feature="neutral_for_example",
        opportunity_kind="token_start",
        initiators=NEUTRAL_CONTROL_INITIATORS["neutral_for_example"],
    ),
    "neutral_such_as": OpportunitySpec(
        feature="neutral_such_as",
        opportunity_kind="token_start",
        initiators=NEUTRAL_CONTROL_INITIATORS["neutral_such_as"],
    ),
    "neutral_in_particular": OpportunitySpec(
        feature="neutral_in_particular",
        opportunity_kind="token_start",
        initiators=NEUTRAL_CONTROL_INITIATORS["neutral_in_particular"],
    ),
    "neutral_as_a_result": OpportunitySpec(
        feature="neutral_as_a_result",
        opportunity_kind="token_start",
        initiators=NEUTRAL_CONTROL_INITIATORS["neutral_as_a_result"],
    ),
    "neutral_controls": OpportunitySpec(
        feature="neutral_controls",
        opportunity_kind="token_start",
        initiators=tuple(
            phrase
            for phrases in NEUTRAL_CONTROL_INITIATORS.values()
            for phrase in phrases
        ),
    ),
    "neutral_common_the": OpportunitySpec(
        feature="neutral_common_the",
        opportunity_kind="token_start",
        initiators=NEUTRAL_COMMON_INITIATORS["neutral_common_the"],
    ),
    "neutral_common_a": OpportunitySpec(
        feature="neutral_common_a",
        opportunity_kind="token_start",
        initiators=NEUTRAL_COMMON_INITIATORS["neutral_common_a"],
    ),
    "neutral_common_of_the": OpportunitySpec(
        feature="neutral_common_of_the",
        opportunity_kind="token_start",
        initiators=NEUTRAL_COMMON_INITIATORS["neutral_common_of_the"],
    ),
    "neutral_common_number_of": OpportunitySpec(
        feature="neutral_common_number_of",
        opportunity_kind="token_start",
        initiators=NEUTRAL_COMMON_INITIATORS["neutral_common_number_of"],
    ),
    "neutral_common_in_the": OpportunitySpec(
        feature="neutral_common_in_the",
        opportunity_kind="token_start",
        initiators=NEUTRAL_COMMON_INITIATORS["neutral_common_in_the"],
    ),
    "neutral_common_to_the": OpportunitySpec(
        feature="neutral_common_to_the",
        opportunity_kind="token_start",
        initiators=NEUTRAL_COMMON_INITIATORS["neutral_common_to_the"],
    ),
    "neutral_common_is_a": OpportunitySpec(
        feature="neutral_common_is_a",
        opportunity_kind="token_start",
        initiators=NEUTRAL_COMMON_INITIATORS["neutral_common_is_a"],
    ),
    "neutral_common_controls": OpportunitySpec(
        feature="neutral_common_controls",
        opportunity_kind="token_start",
        initiators=tuple(
            phrase
            for phrases in NEUTRAL_COMMON_INITIATORS.values()
            for phrase in phrases
        ),
    ),
}


def iter_feature_opportunities(
    text: str,
    *,
    features: Iterable[str] | None = None,
    max_token_start_opportunities: int | None = None,
) -> list[FeatureOpportunity]:
    selected = set(features or PHASE2_OPPORTUNITY_SPECS)
    hits_by_feature = _hit_starts_by_feature(text, selected)
    opportunities: list[FeatureOpportunity] = []
    for feature in sorted(selected):
        spec = PHASE2_OPPORTUNITY_SPECS.get(feature)
        if spec is None:
            continue
        offsets = _opportunity_offsets(
            text,
            spec.opportunity_kind,
            max_token_start_opportunities=max_token_start_opportunities,
        )
        feature_hits = hits_by_feature.get(feature, {})
        for offset in offsets:
            opportunities.append(
                FeatureOpportunity(
                    feature=feature,
                    opportunity_kind=spec.opportunity_kind,
                    char_offset=offset,
                    reference_initiates=offset in feature_hits,
                    matched_subtype=feature_hits.get(offset),
                )
            )
    return sorted(opportunities, key=lambda item: (item.char_offset, item.feature))


def _hit_starts_by_feature(text: str, selected: set[str]) -> dict[str, dict[int, str]]:
    starts: dict[str, dict[int, str]] = {}
    tier1_features = {
        feature
        for feature in selected
        if feature
        in {
            "contrastive_negation",
            "slop_lexicon",
            "stock_openers",
            "stock_closers",
            "rule_of_three_approx",
        }
    }
    if "stock_openers_closers" in selected:
        tier1_features.update({"stock_openers", "stock_closers"})
    if tier1_features:
        for hit in iter_tier1_hits(text, features=tier1_features):
            starts.setdefault(hit.feature, {})[hit.start] = hit.subtype
            if hit.feature in {"stock_openers", "stock_closers"}:
                starts.setdefault("stock_openers_closers", {})[hit.start] = hit.subtype
    for feature, pattern in NEUTRAL_CONTROL_PATTERNS.items():
        if feature not in selected and "neutral_controls" not in selected:
            continue
        for match in pattern.finditer(text):
            starts.setdefault(feature, {})[match.start()] = feature.removeprefix("neutral_")
            starts.setdefault("neutral_controls", {})[match.start()] = feature.removeprefix("neutral_")
    for feature, pattern in NEUTRAL_COMMON_PATTERNS.items():
        if feature not in selected and "neutral_common_controls" not in selected:
            continue
        for match in pattern.finditer(text):
            subtype = feature.removeprefix("neutral_common_")
            starts.setdefault(feature, {})[match.start()] = subtype
            starts.setdefault("neutral_common_controls", {})[match.start()] = subtype
    return starts


def _opportunity_offsets(
    text: str,
    opportunity_kind: str,
    *,
    max_token_start_opportunities: int | None,
) -> list[int]:
    if opportunity_kind == "document_start":
        return [0] if text.strip() else []
    if opportunity_kind == "token_start":
        offsets = [match.start() for match in TOKEN_RE.finditer(text)]
        if max_token_start_opportunities is not None:
            return offsets[:max_token_start_opportunities]
        return offsets
    if opportunity_kind == "clause_boundary":
        return _boundary_offsets(text)
    if opportunity_kind == "final_clause_boundary":
        offsets = _boundary_offsets(text)
        threshold = int(len(text) * 0.6)
        final_offsets = [offset for offset in offsets if offset >= threshold]
        return final_offsets or offsets[-1:]
    if opportunity_kind == "document_or_final_boundary":
        return sorted({0, *_opportunity_offsets(
            text,
            "final_clause_boundary",
            max_token_start_opportunities=max_token_start_opportunities,
        )})
    raise ValueError(f"unsupported opportunity kind: {opportunity_kind}")


def _boundary_offsets(text: str) -> list[int]:
    if not text.strip():
        return []
    offsets = {0}
    for match in BOUNDARY_RE.finditer(text):
        offsets.add(match.end())
    return sorted(offset for offset in offsets if offset < len(text))
