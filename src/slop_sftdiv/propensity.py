from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from slop_sftdiv.features.tier1_matchers import TOKEN_RE, iter_tier1_hits


BOUNDARY_RE = re.compile(r"(?:^|[.!?;:\n]\s+)")


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
            "not",
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
}


def iter_feature_opportunities(
    text: str,
    *,
    features: Iterable[str] | None = None,
    max_token_start_opportunities: int | None = None,
) -> list[FeatureOpportunity]:
    selected = set(features or PHASE2_OPPORTUNITY_SPECS)
    hits_by_feature = _hit_starts_by_feature(text)
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


def _hit_starts_by_feature(text: str) -> dict[str, dict[int, str]]:
    starts: dict[str, dict[int, str]] = {}
    for hit in iter_tier1_hits(text):
        starts.setdefault(hit.feature, {})[hit.start] = hit.subtype
        if hit.feature in {"stock_openers", "stock_closers"}:
            starts.setdefault("stock_openers_closers", {})[hit.start] = hit.subtype
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
