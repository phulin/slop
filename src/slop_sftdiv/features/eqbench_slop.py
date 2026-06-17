"""Portable EQ-Bench Slop Score feature helpers.

The bundled data snapshot comes from ``sam-paech/slop-score`` and reproduces
the public EQ-Bench Slop Score word/trigram lists and leaderboard normalization
ranges. The JavaScript benchmark also contains an optional POS-tagged contrast
stage; this module ports the deterministic public list matching and Stage-1
surface contrast regexes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from functools import lru_cache
from importlib import resources
import subprocess
from typing import Any, Iterable


WORD_RE = re.compile(r"[a-z']+")
ALPHA_TOKEN_RE = re.compile(r"^[a-z]+(?:'[a-z]+)?$")


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.UNICODE)


MAXG = 160
PRON = r"(?:it|they|this|that)"
BE = r"(?:is|are|was|were)"
BE_NEG = r"(?:is\s+not|are\s+not|was\s+not|were\s+not|isn't|aren't|wasn't|weren't|ain't)"


EQBENCH_STAGE1_CONTRAST_PATTERNS: dict[str, re.Pattern[str]] = {
    "RE_NOT_BUT": _compile(
        rf"\b(?:(?:{BE_NEG})|not(?!\s+(?:that|only)\b))\s+"
        rf"(?:(?!\bbut\b|[.?!]).){{1,100}}?"
        rf"[,;:]\s*but\s+"
        rf"(?!when\b|while\b|which\b|who\b|whom\b|whose\b|where\b|if\b|that\b|as\b|because\b|although\b|though\b|till\b|until\b|unless\b|"
        rf"here\b|there\b|then\b|my\b|we\b|I\b|you\b|it\s+seems\b|it\s+appears\b|it\s+felt\b|it\s+looks?\b|anything\b)"
    ),
    "RE_NOT_DASH": _compile(
        rf"\b(?:\w+n't|not)\s+(?:just|only|merely)?\s+"
        rf"(?:(?![.?!]).){{1,{MAXG}}}?"
        rf"(?:-|\s-\s|[\u2014\u2013])\s*"
        rf"{PRON}\s+(?:(?:'re|are|'s|is|were|was)\b|(?!'re|are|'s|is|were|was)[*_~]*[a-z]\w*)"
    ),
    "RE_PRON_BE_NOT_SEP_BE": _compile(
        rf"(?:(?<=^)|(?<=[.?!]\s))\s*[\"']?"
        rf"(?:(?:{PRON}\s+{BE}\s+not)|(?:{PRON}\s+{BE}n't)|(?:it's|they're|that's)\s+not)\b"
        rf"[^.?!]{{0,{MAXG}}}[.;:?!]\s*[\"']?"
        rf"{PRON}\s+(?:{BE}|(?:'s|'re))\b(?!\s+not\b)"
    ),
    "RE_NP_BE_NOT_SEP_THEY_BE": _compile(
        rf"(?:(?<=^)|(?<=[.?!]\s))\s*"
        rf"(?![^.?!]{{0,80}}\b(?:knew|know|thought|think|said|says|told|heard|learned)\b[^.?!]{{0,40}}?\bthat\b)"
        rf"(?!\s*not\s+without\b)"
        rf"(?![^.?!]{{0,50}}\bnot\s+put\b)"
        rf"[^.?!]{{0,{MAXG}}}?\b(?:{BE_NEG})\b[^.?!]{{0,{MAXG}}}[.;:?!]\s*"
        rf"[\"']?{PRON}\b(?:'re|\s+(?:are|were|is|was))\b(?!\s+not\b)"
    ),
    "RE_NO_LONGER": _compile(
        rf"(?:(?<=^)|(?<=[.?!]\s))\s*[^.?!]{{0,{MAXG}}}\bno\s+longer\b[^.;:?!]{{0,{MAXG}}}"
        rf"[.;:?!]\s*(?:it|they|this|that)\s+(?:is|are|was|were)\b(?!\s+not\b)"
    ),
    "RE_NOT_JUST_SEP": _compile(
        rf"(?:(?<=^)|(?<=[.?!]\s))\s*[\"']?"
        rf"{PRON}\b(?:'s|'re|\s+(?:is|are|was|were))?\s+not\s+just\b[^.?!]{{0,{MAXG}}}[.?!]\s*[\"']?"
        rf"{PRON}\b(?:'s|'re|\s+(?:is|are|was|were))\b(?!\s+not\b)"
    ),
    "RE_NOT_PERIOD_SAMEVERB": _compile(
        rf"(?:(?<=^)|(?<=[.?!]\s))[^.?!]*?\b(?:do|does|did)n't\b\s+"
        rf"(?:(?:\w+\s+){{0,2}})([a-z]{{3,}})\b[^.?!]*[.?!]\s*"
        rf"{PRON}\s+\1(?:ed|es|s|ing)?\b"
    ),
    "RE_SIMPLE_BE_NOT_IT_BE": _compile(
        rf"(?:(?<=^)|(?<=[.?!]\s))\s*[\"']?"
        rf"(?!he\b|she\b|i\b|you\b|we\b)"
        rf"(?![^.?!]{{0,80}}\b(?:knew|know|thought|think|said|says|told|heard|learned)\b[^.?!]{{0,40}}?\bthat\b)"
        rf"[^.?!]{{0,{MAXG}}}?\b{BE_NEG}\b[^.?!]{{0,{MAXG}}}[.;:?!]\s*"
        rf"[\"']?it(?:'s|\s+(?:is|are|was|were))\b"
    ),
    "RE_EMBEDDED_NOT_JUST_SEP": _compile(
        rf"(?:(?<=^)|(?<=[.?!]\s))"
        rf"[^.?!]{{0,80}}?\b(?:(?:it|they)\s+(?:is|are)|(?:it's|they're))\s+not\s+just\b"
        rf"[^.?!]{{0,{MAXG}}}[.?!]\s*"
        rf"(?:(?:it|they)\s+(?:is|are)|(?:it's|they're))\b"
    ),
    "RE_DIALOGUE_NOT_JUST": _compile(
        rf"[\"']?{PRON}(?:'re|'s|\s+(?:are|is|was|were))\s+not\s+just\b[^\"']{{0,{MAXG}}}[\"']?\s*"
        rf"(?:[^.?!]{{0,80}}\b(?:said|asked|whispered|muttered|replied|added|shouted|cried)\b[^.?!]{{0,80}}[.?!]\s*)?"
        rf"[\"']?{PRON}(?:'re|'s|\s+(?:are|is|was|were))\s+[*_~]?[a-z]\w*"
    ),
}


@dataclass(frozen=True)
class EqbenchSlopMetrics:
    """EQ-Bench Slop Score components for one text."""

    slop_score: float
    word_matches_per_1k_words: float
    trigram_matches_per_1k_words: float
    not_x_but_y_per_1k_chars: float
    word_hit_count: int
    trigram_hit_count: int
    contrast_hit_count: int
    token_count: int


@dataclass(frozen=True)
class EqbenchSlopScorecard:
    """Aggregate EQ-Bench Slop Score result for a benchmark text set."""

    metrics: EqbenchSlopMetrics
    contrast_mode: str
    sample_count: int
    valid_sample_count: int
    total_chars: int
    min_chars: int
    slop_word_hits: list[tuple[str, int]]
    slop_trigram_hits: list[tuple[str, int]]
    contrast_matches: list[tuple[str, str]]


def eqbench_tokens(text: str) -> list[str]:
    """Tokenize text the same way EQ-Bench's ``wordsOnlyLower`` + ``alphaTokens`` path does."""

    lowered = _normalize_quotes(text.lower())
    tokens = [item.strip("'") for item in WORD_RE.findall(lowered)]
    return [token for token in tokens if token and ALPHA_TOKEN_RE.match(token)]


def extract_eqbench_slop_metrics(text: str) -> EqbenchSlopMetrics:
    """Compute the portable EQ-Bench Slop Score components."""

    tokens = eqbench_tokens(text)
    token_count = len(tokens)
    word_hits = _eqbench_word_hit_count(tokens)
    trigram_hits = _eqbench_trigram_hit_count(tokens)
    contrast_hits = len(iter_eqbench_contrast_hits(text))
    chars = len(text)

    word_score = (word_hits / token_count) * 1000.0 if token_count else 0.0
    trigram_score = (trigram_hits / token_count) * 1000.0 if token_count else 0.0
    contrast_score = (contrast_hits / chars) * 1000.0 if chars else 0.0
    score = 0.0 if not text else _eqbench_weighted_score(word_score, trigram_score, contrast_score)
    return EqbenchSlopMetrics(
        slop_score=score,
        word_matches_per_1k_words=word_score,
        trigram_matches_per_1k_words=trigram_score,
        not_x_but_y_per_1k_chars=contrast_score,
        word_hit_count=word_hits,
        trigram_hit_count=trigram_hits,
        contrast_hit_count=contrast_hits,
        token_count=token_count,
    )


def score_eqbench_outputs(
    outputs: list[str],
    *,
    min_chars: int = 300,
    contrast_mode: str = "portable",
) -> EqbenchSlopScorecard:
    """Score a list of generated outputs using the EQ-Bench aggregate shape.

    EQ-Bench's leaderboard filters samples with ``output.length > 300`` and
    scores the concatenated remaining text. This helper mirrors that aggregate
    path while keeping ``min_chars`` configurable for small smoke tests.
    """

    valid_outputs = [output for output in outputs if len(output) > min_chars]
    concatenated = "\n\n".join(valid_outputs)
    tokens = eqbench_tokens(concatenated)
    token_count = len(tokens)
    word_hits = iter_eqbench_word_hits(concatenated)
    trigram_hits = iter_eqbench_trigram_hits(concatenated)
    contrast_hits = _scorecard_contrast_hits(concatenated, contrast_mode=contrast_mode)
    chars = len(concatenated)
    word_score = (len(word_hits) / token_count) * 1000.0 if token_count else 0.0
    trigram_score = (len(trigram_hits) / token_count) * 1000.0 if token_count else 0.0
    contrast_score = (len(contrast_hits) / chars) * 1000.0 if chars else 0.0
    score = (
        0.0
        if not concatenated
        else _eqbench_weighted_score(word_score, trigram_score, contrast_score)
    )
    metrics = EqbenchSlopMetrics(
        slop_score=score,
        word_matches_per_1k_words=word_score,
        trigram_matches_per_1k_words=trigram_score,
        not_x_but_y_per_1k_chars=contrast_score,
        word_hit_count=len(word_hits),
        trigram_hit_count=len(trigram_hits),
        contrast_hit_count=len(contrast_hits),
        token_count=token_count,
    )
    return EqbenchSlopScorecard(
        metrics=metrics,
        contrast_mode=contrast_mode,
        sample_count=len(outputs),
        valid_sample_count=len(valid_outputs),
        total_chars=len(concatenated),
        min_chars=min_chars,
        slop_word_hits=_top_counts(word for word, _, _ in word_hits),
        slop_trigram_hits=_top_counts(trigram for trigram, _, _ in trigram_hits),
        contrast_matches=[
            (pattern_name, match_text)
            for pattern_name, match_text in contrast_hits
        ][:100],
    )


def iter_eqbench_word_hits(text: str) -> list[tuple[str, int, int]]:
    """Return EQ-Bench slop-list word hits as ``(word, start, end)`` tuples."""

    lowered = _normalize_quotes(text.lower())
    slop_words = _eqbench_slop_words()
    hits: list[tuple[str, int, int]] = []
    for match in WORD_RE.finditer(lowered):
        token = match.group(0).strip("'")
        if token and ALPHA_TOKEN_RE.match(token) and token in slop_words:
            hits.append((token, match.start(), match.end()))
    return hits


def iter_eqbench_trigram_hits(text: str) -> list[tuple[str, int, int]]:
    """Return EQ-Bench slop trigram hits as ``(trigram, start, end)`` tuples."""

    lowered = _normalize_quotes(text.lower())
    tokens: list[tuple[str, int, int]] = []
    for match in WORD_RE.finditer(lowered):
        token = match.group(0).strip("'")
        if token and ALPHA_TOKEN_RE.match(token):
            tokens.append((token, match.start(), match.end()))
    slop_trigrams = _eqbench_slop_trigrams()
    hits: list[tuple[str, int, int]] = []
    for i in range(max(0, len(tokens) - 2)):
        trigram = f"{tokens[i][0]} {tokens[i + 1][0]} {tokens[i + 2][0]}"
        if trigram in slop_trigrams:
            hits.append((trigram, tokens[i][1], tokens[i + 2][2]))
    return hits


def iter_eqbench_contrast_hits(text: str) -> list[tuple[str, int, int, str]]:
    """Return portable EQ-Bench Stage-1 contrast hits."""

    normalized = _normalize_text(text)
    hits: list[tuple[str, int, int, str]] = []
    for name, pattern in EQBENCH_STAGE1_CONTRAST_PATTERNS.items():
        for match in pattern.finditer(normalized):
            hits.append((name, match.start(), match.end(), match.group(0)))
    return _dedupe_interval_hits(hits)


def iter_eqbench_exact_contrast_matches(text: str) -> list[tuple[str, str]]:
    """Return exact EQ-Bench contrast matches using the upstream wink POS stage.

    This path shells out to the bundled Node helper, which depends on the
    repository's ``wink-pos-tagger`` npm dependency. It mirrors the upstream
    Stage-1 + Stage-2 contrast detector for benchmark-level scorecards.
    """

    script_path = resources.files("slop_sftdiv.resources.eqbench_slop").joinpath(
        "exact_contrast.mjs"
    )
    try:
        result = subprocess.run(
            ["node", str(script_path)],
            input=json.dumps({"text": text}),
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Node.js is required for exact EQ-Bench contrast mode") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"exact EQ-Bench contrast mode failed: {detail}") from exc
    payload = json.loads(result.stdout)
    matches = payload.get("matches", [])
    return [
        (str(item.get("pattern_name", "")), str(item.get("match_text", "")))
        for item in matches
        if isinstance(item, dict)
    ]


def _scorecard_contrast_hits(text: str, *, contrast_mode: str) -> list[tuple[str, str]]:
    if contrast_mode == "portable":
        return [
            (pattern_name, match_text)
            for pattern_name, _, _, match_text in iter_eqbench_contrast_hits(text)
        ]
    if contrast_mode == "exact":
        return iter_eqbench_exact_contrast_matches(text)
    if contrast_mode == "auto":
        try:
            return iter_eqbench_exact_contrast_matches(text)
        except RuntimeError:
            return [
                (pattern_name, match_text)
                for pattern_name, _, _, match_text in iter_eqbench_contrast_hits(text)
            ]
    raise ValueError("contrast_mode must be one of: portable, exact, auto")


def _eqbench_word_hit_count(tokens: list[str]) -> int:
    words = _eqbench_slop_words()
    return sum(1 for token in tokens if token in words)


def _eqbench_trigram_hit_count(tokens: list[str]) -> int:
    if len(tokens) < 3:
        return 0
    trigrams = _eqbench_slop_trigrams()
    return sum(
        1
        for i in range(len(tokens) - 2)
        if f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}" in trigrams
    )


def _eqbench_weighted_score(word_score: float, trigram_score: float, contrast_score: float) -> float:
    ranges = _eqbench_metadata()["normalization_ranges"]
    norm_words = _normalize_value(word_score, ranges["slop_words"])
    norm_trigrams = _normalize_value(trigram_score, ranges["slop_trigrams"])
    norm_contrast = _normalize_value(contrast_score, ranges["contrast"])
    return (norm_words * 0.60 + norm_contrast * 0.25 + norm_trigrams * 0.15) * 100.0


def _normalize_value(value: float, range_payload: dict[str, float]) -> float:
    denominator = range_payload["max"] - range_payload["min"]
    if denominator <= 0:
        return 0.0
    normalized = (value - range_payload["min"]) / denominator
    return max(0.0, min(1.0, normalized))


def _normalize_quotes(text: str) -> str:
    return re.sub(r"[\u2018\u2019\u201A\u201B\u2032\u02BC\uFF07`]", "'", text)


def _normalize_text(text: str) -> str:
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2014", "-")
        .replace("\u2013", "-")
    )


def _dedupe_interval_hits(
    hits: list[tuple[str, int, int, str]]
) -> list[tuple[str, int, int, str]]:
    sorted_hits = sorted(hits, key=lambda item: (item[1], item[2], item[0]))
    deduped: list[tuple[str, int, int, str]] = []
    last_end = -1
    for hit in sorted_hits:
        if hit[1] < last_end:
            continue
        deduped.append(hit)
        last_end = hit[2]
    return deduped


def _top_counts(items: Iterable[str], *, limit: int = 50) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]


@lru_cache(maxsize=1)
def _eqbench_slop_words() -> frozenset[str]:
    return frozenset(_load_phrase_list("slop_list.json"))


@lru_cache(maxsize=1)
def _eqbench_slop_trigrams() -> frozenset[str]:
    return frozenset(_load_phrase_list("slop_list_trigrams.json"))


@lru_cache(maxsize=1)
def _eqbench_metadata() -> dict[str, Any]:
    return json.loads(_resource_text("metadata.json"))


def eqbench_prompt_manifest() -> list[Any]:
    """Return the vendored 300-prompt EQ-Bench prompt manifest."""

    return json.loads(_resource_text("prompts.json"))


def _load_phrase_list(filename: str) -> list[str]:
    raw = json.loads(_resource_text(filename))
    phrases = []
    for item in raw:
        if not item:
            continue
        phrase_match = re.search(r"[a-z]+(?:'[a-z]+)?(?:\s+[a-z]+(?:'[a-z]+)?)*", str(item[0]).lower())
        if phrase_match:
            phrases.append(phrase_match.group(0))
    return phrases


def _resource_text(filename: str) -> str:
    return (
        resources.files("slop_sftdiv.resources.eqbench_slop")
        .joinpath(filename)
        .read_text(encoding="utf-8")
    )
