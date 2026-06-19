from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from slop_sftdiv.features.tier1_matchers import count_tokens
from slop_sftdiv.generation_text import FeatureTextMode, feature_text_for_mode


TOKEN_RE = re.compile(r"[a-z][a-z0-9'_-]*|[0-9]+|#{1,6}|\*\*|[-*+]", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
ASSISTANT_PREFIX_RE = re.compile(r"^\s*assistant\s*", re.IGNORECASE)
EPSILON = 1e-9

KNOWN_TIER1_TERMS = {
    "delve",
    "tapestry",
    "testament to",
    "important to note",
    "worth noting",
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
    "great question",
    "hope this helps",
    "in conclusion",
    "overall",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class CorpusSpec:
    label: str
    path: Path
    field: str
    kind: str


@dataclass
class Document:
    source: str
    corpus: str
    path: str
    record_id: str
    text: str
    token_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bounded Phase 4 feature discovery: rank model-skewed n-gram spans from "
            "existing generations against reference texts, cluster them heuristically, "
            "and emit matcher specs plus a rerun census."
        )
    )
    parser.add_argument(
        "--reference-jsonl",
        action="append",
        default=[],
        metavar="LABEL=PATH[:FIELD]",
        help="Reference JSONL. Defaults to field 'text'. Repeatable.",
    )
    parser.add_argument(
        "--generation-jsonl",
        action="append",
        default=[],
        metavar="LABEL=PATH[:FIELD]",
        help="Generation JSONL. Defaults to field 'generation'. Repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-docs-per-source", type=int, default=2048)
    parser.add_argument("--max-examples-per-candidate", type=int, default=4)
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=4)
    parser.add_argument("--min-model-docs", type=int, default=8)
    parser.add_argument("--min-model-count", type=int, default=12)
    parser.add_argument("--min-lift", type=float, default=1.5)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--matcher-top-k", type=int, default=12)
    parser.add_argument(
        "--feature-text-mode",
        choices=["raw", "final_answer"],
        default="final_answer",
        help="Text cleaning mode for generated responses.",
    )
    return parser


def _parse_spec(raw: str, *, default_field: str, kind: str) -> CorpusSpec:
    if "=" not in raw:
        raise ValueError(f"corpus spec must be LABEL=PATH[:FIELD], got {raw!r}")
    label, rest = raw.split("=", 1)
    path_text, field = _split_path_field(rest, default_field=default_field)
    if not label.strip():
        raise ValueError(f"corpus spec label is empty: {raw!r}")
    return CorpusSpec(label=label.strip(), path=Path(path_text), field=field, kind=kind)


def _split_path_field(value: str, *, default_field: str) -> tuple[str, str]:
    path = Path(value)
    if path.exists():
        return value, default_field
    if ":" in value:
        maybe_path, maybe_field = value.rsplit(":", 1)
        if maybe_path and maybe_field:
            return maybe_path, maybe_field
    return value, default_field


def _read_jsonl(spec: CorpusSpec, *, max_docs: int, feature_text_mode: str) -> list[Document]:
    docs: list[Document] = []
    with spec.path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if len(docs) >= max_docs:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get(spec.field, "") or "")
            if spec.kind == "generation":
                text = feature_text_for_mode(
                    _clean_generation_text(text),
                    cast(FeatureTextMode, feature_text_mode),
                )
            text = _normalize_text(text)
            if not text:
                continue
            record_id = str(
                row.get("record_id")
                or row.get("doc_id")
                or row.get("phase2_prompt_id")
                or row.get("id")
                or row_index
            )
            docs.append(
                Document(
                    source=spec.label,
                    corpus=spec.kind,
                    path=str(spec.path),
                    record_id=record_id,
                    text=text,
                    token_count=count_tokens(text),
                )
            )
    return docs


def _clean_generation_text(text: str) -> str:
    return ASSISTANT_PREFIX_RE.sub("", text).strip()


def _normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.replace("\u00a0", " ")).strip()


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def _ngrams(tokens: list[str], *, ngram_min: int, ngram_max: int) -> list[str]:
    spans: list[str] = []
    for n in range(ngram_min, ngram_max + 1):
        if n <= 0 or len(tokens) < n:
            continue
        for index in range(0, len(tokens) - n + 1):
            gram_tokens = tokens[index : index + n]
            if _skip_ngram(gram_tokens):
                continue
            spans.append(" ".join(gram_tokens))
    return spans


def _skip_ngram(tokens: list[str]) -> bool:
    if not tokens:
        return True
    if any(token in {"-", "*", "+", "**", "##", "###", "####", "#####", "######"} for token in tokens):
        return True
    if len(tokens) == 1:
        token = tokens[0]
        return token in STOPWORDS or len(token) < 3 or token.isdigit()
    if all(token in STOPWORDS for token in tokens):
        return True
    if tokens[0] in {"and", "or", "the", "a", "an", "to", "of"}:
        return True
    if tokens[-1] in {"and", "or", "the", "a", "an", "to", "of"}:
        return True
    return False


def _corpus_counts(
    docs: list[Document],
    *,
    ngram_min: int,
    ngram_max: int,
) -> tuple[Counter[str], Counter[str], dict[str, list[Document]], int]:
    total_counts: Counter[str] = Counter()
    doc_counts: Counter[str] = Counter()
    example_docs: dict[str, list[Document]] = defaultdict(list)
    total_tokens = 0
    for doc in docs:
        tokens = _tokens(doc.text)
        total_tokens += len(tokens)
        grams = _ngrams(tokens, ngram_min=ngram_min, ngram_max=ngram_max)
        total_counts.update(grams)
        for gram in set(grams):
            doc_counts[gram] += 1
            if len(example_docs[gram]) < 20:
                example_docs[gram].append(doc)
    return total_counts, doc_counts, example_docs, total_tokens


def _candidate_rows(
    *,
    model_docs: list[Document],
    reference_docs: list[Document],
    ngram_min: int,
    ngram_max: int,
    min_model_docs: int,
    min_model_count: int,
    min_lift: float,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, list[Document]], dict[str, Any]]:
    model_counts, model_doc_counts, examples, model_tokens = _corpus_counts(
        model_docs, ngram_min=ngram_min, ngram_max=ngram_max
    )
    ref_counts, ref_doc_counts, _ref_examples, ref_tokens = _corpus_counts(
        reference_docs, ngram_min=ngram_min, ngram_max=ngram_max
    )
    vocabulary = set(model_counts) | set(ref_counts)
    prior = 0.5
    rows: list[dict[str, Any]] = []
    model_doc_total = max(1, len(model_docs))
    ref_doc_total = max(1, len(reference_docs))
    for phrase, model_count in model_counts.items():
        model_docs_with_phrase = model_doc_counts[phrase]
        if model_count < min_model_count or model_docs_with_phrase < min_model_docs:
            continue
        ref_count = ref_counts.get(phrase, 0)
        model_rate = model_count / max(1, model_tokens) * 1000.0
        ref_rate = ref_count / max(1, ref_tokens) * 1000.0
        smoothed_model_rate = (model_count + prior) / (model_tokens + prior * len(vocabulary))
        smoothed_ref_rate = (ref_count + prior) / (ref_tokens + prior * len(vocabulary))
        lift = smoothed_model_rate / max(EPSILON, smoothed_ref_rate)
        if lift < min_lift:
            continue
        model_doc_rate = model_docs_with_phrase / model_doc_total
        ref_doc_rate = ref_doc_counts.get(phrase, 0) / ref_doc_total
        score = math.log2(lift) * math.sqrt(model_docs_with_phrase) * (1.0 + model_doc_rate)
        family = _cluster_family(phrase)
        rows.append(
            {
                "candidate": phrase,
                "cluster": family,
                "auto_label": _cluster_label(family),
                "model_count": model_count,
                "reference_count": ref_count,
                "model_doc_count": model_docs_with_phrase,
                "reference_doc_count": ref_doc_counts.get(phrase, 0),
                "model_per_1k_tokens": model_rate,
                "reference_per_1k_tokens": ref_rate,
                "model_doc_rate": model_doc_rate,
                "reference_doc_rate": ref_doc_rate,
                "lift_vs_reference": lift,
                "log2_lift_vs_reference": math.log2(lift),
                "score": score,
                "known_tier1_overlap": _known_tier1_overlap(phrase),
                "auto_perceptibility_proxy": _perceptibility_proxy(phrase, family),
                "matcher_regex": _phrase_regex(phrase),
                "opportunity_definition": _opportunity_definition(phrase, family),
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["auto_perceptibility_proxy"] == "likely_perceptible"),
            float(row["score"]),
        ),
        reverse=True,
    )
    summary = {
        "model_documents": len(model_docs),
        "reference_documents": len(reference_docs),
        "model_tokens": model_tokens,
        "reference_tokens": ref_tokens,
        "vocabulary_size": len(vocabulary),
        "candidate_count_before_top_k": len(rows),
    }
    return rows[:top_k], examples, summary


def _known_tier1_overlap(phrase: str) -> str:
    phrase_norm = phrase.lower()
    for term in KNOWN_TIER1_TERMS:
        if phrase_norm == term or phrase_norm in term or term in phrase_norm:
            return "yes"
    return "no"


def _cluster_family(phrase: str) -> str:
    tokens = phrase.split()
    phrase_norm = " ".join(tokens)
    if any(token in {"##", "###", "**", "-", "*"} for token in tokens):
        return "format_scaffolding"
    if phrase_norm.startswith(("here are", "are a few", "are some")):
        return "offer_examples"
    if phrase_norm.startswith(("do you have", "you have any", "what are some", "what do you")):
        return "followup_question_prompt"
    if phrase_norm.startswith(("thank you", "thanks for")):
        return "politeness_thanks"
    if phrase_norm.startswith(("i hope", "hope this")):
        return "stock_helpful_closure"
    if phrase_norm.startswith(("i'm not sure", "i'm glad")):
        return "conversational_disclaimer"
    if phrase_norm.startswith(("let me know", "me know if", "know if you")):
        return "followup_offer"
    if phrase_norm.startswith(("let's", "lets", "we can", "we'll", "we will")):
        return "assistant_process_framing"
    if phrase_norm.startswith(("it is important", "it's important", "remember", "note that")):
        return "advice_caveat_framing"
    if phrase_norm.startswith(("in conclusion", "overall", "to summarize", "ultimately")):
        return "summary_closure"
    if phrase_norm.startswith(("for example", "such as", "in other words")):
        return "explanatory_transition"
    if any(token in {"ensure", "make", "consider", "avoid", "should", "need"} for token in tokens):
        return "prescriptive_instruction"
    if any(token in {"comprehensive", "effective", "important", "robust", "clear", "specific"} for token in tokens):
        return "evaluative_adjectives"
    if len(tokens) >= 3 and tokens[0] in {"first", "second", "third", "finally"}:
        return "ordered_reasoning"
    if len(tokens) >= 3:
        return f"phrase_family_{tokens[0]}"
    return "lexical_lift"


def _cluster_label(family: str) -> str:
    labels = {
        "format_scaffolding": "Markdown/list scaffolding and visible answer formatting",
        "assistant_process_framing": "Assistant process framing",
        "advice_caveat_framing": "Caveat and importance framing",
        "summary_closure": "Summary or conclusion closure",
        "explanatory_transition": "Explanatory transition phrase",
        "prescriptive_instruction": "Prescriptive instruction style",
        "evaluative_adjectives": "Evaluative assistant-register adjectives",
        "ordered_reasoning": "Explicit ordered reasoning",
        "lexical_lift": "Single lexical marker with model-side lift",
        "offer_examples": "Offer-of-examples framing",
        "followup_question_prompt": "Conversational follow-up question prompt",
        "politeness_thanks": "Politeness and thanks formula",
        "stock_helpful_closure": "Stock helpful closure",
        "conversational_disclaimer": "Conversational uncertainty or affirmation",
        "followup_offer": "Follow-up assistance offer",
    }
    if family.startswith("phrase_family_"):
        return f"Repeated phrase family rooted at '{family.removeprefix('phrase_family_')}'"
    return labels.get(family, family.replace("_", " "))


def _perceptibility_proxy(phrase: str, family: str) -> str:
    if family in {
        "format_scaffolding",
        "assistant_process_framing",
        "advice_caveat_framing",
        "summary_closure",
        "prescriptive_instruction",
        "ordered_reasoning",
    }:
        return "likely_perceptible"
    if len(phrase.split()) >= 2 and family != "lexical_lift":
        return "likely_perceptible"
    return "machine_detectability_only_or_uncertain"


def _phrase_regex(phrase: str) -> str:
    parts = [re.escape(part) for part in phrase.split()]
    body = r"\s+".join(parts)
    prefix = r"\b" if phrase[:1].isalnum() else ""
    suffix = r"\b" if phrase[-1:].isalnum() else ""
    return prefix + body + suffix


def _opportunity_definition(phrase: str, family: str) -> str:
    if family in {"summary_closure", "assistant_process_framing", "format_scaffolding", "ordered_reasoning"}:
        return "line_or_sentence_start"
    if family in {"advice_caveat_framing", "prescriptive_instruction"}:
        return "sentence_start_or_clause_boundary"
    return "any_token_position"


def _cluster_rows(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[str(row["cluster"])].append(row)
    rows = []
    for cluster, items in grouped.items():
        top_items = sorted(items, key=lambda row: float(row["score"]), reverse=True)
        rows.append(
            {
                "cluster": cluster,
                "auto_label": _cluster_label(cluster),
                "candidate_count": len(items),
                "likely_perceptible_count": sum(
                    1 for row in items if row["auto_perceptibility_proxy"] == "likely_perceptible"
                ),
                "top_candidate": top_items[0]["candidate"],
                "top_score": top_items[0]["score"],
                "top_lift_vs_reference": top_items[0]["lift_vs_reference"],
                "candidate_examples": "; ".join(str(row["candidate"]) for row in top_items[:6]),
            }
        )
    return sorted(rows, key=lambda row: (int(row["likely_perceptible_count"]), float(row["top_score"])), reverse=True)


def _example_rows(
    candidates: list[dict[str, Any]],
    examples: dict[str, list[Document]],
    *,
    max_examples: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        phrase = str(candidate["candidate"])
        pattern = re.compile(_phrase_regex(phrase), re.IGNORECASE)
        for doc in examples.get(phrase, [])[:max_examples]:
            rows.append(
                {
                    "candidate": phrase,
                    "cluster": candidate["cluster"],
                    "source": doc.source,
                    "record_id": doc.record_id,
                    "snippet": _snippet(doc.text, pattern),
                }
            )
    return rows


def _snippet(text: str, pattern: re.Pattern[str], *, context_chars: int = 120) -> str:
    match = pattern.search(text)
    if match is None:
        return text[: context_chars * 2]
    start = max(0, match.start() - context_chars)
    end = min(len(text), match.end() + context_chars)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end] + suffix


def _matcher_specs(candidates: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    specs = []
    rows = _representative_candidate_rows(candidates, limit=limit)
    for index, row in enumerate(rows, start=1):
        name = re.sub(r"[^a-z0-9]+", "_", str(row["candidate"]).lower()).strip("_")
        if not name:
            name = str(row["cluster"])
        specs.append(
            {
                "feature": f"phase4_candidate_{index:02d}_{name[:40]}",
                "candidate": row["candidate"],
                "cluster": row["cluster"],
                "auto_label": row["auto_label"],
                "regex": row["matcher_regex"],
                "opportunity_definition": row["opportunity_definition"],
                "precision_status": "requires_200_hit_manual_validation_before_core_claim",
                "promotion_status": (
                    "candidate_perceptible"
                    if row["auto_perceptibility_proxy"] == "likely_perceptible"
                    else "candidate_machine_detectability_only"
                ),
            }
        )
    return specs


def _representative_candidate_rows(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_cluster[str(row["cluster"])].append(row)

    representatives: list[dict[str, Any]] = []
    for rows in by_cluster.values():
        sorted_rows = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
        multiword = [row for row in sorted_rows if len(str(row["candidate"]).split()) >= 2]
        representatives.append(multiword[0] if multiword else sorted_rows[0])

    representatives.sort(
        key=lambda row: (
            int(row["auto_perceptibility_proxy"] == "likely_perceptible"),
            float(row["score"]),
        ),
        reverse=True,
    )
    return representatives[:limit]


def _census_rows(specs: list[dict[str, Any]], docs: list[Document]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_source: dict[tuple[str, str], list[Document]] = defaultdict(list)
    for doc in docs:
        by_source[(doc.corpus, doc.source)].append(doc)
    for spec in specs:
        pattern = re.compile(str(spec["regex"]), re.IGNORECASE)
        for (corpus, source), source_docs in sorted(by_source.items()):
            hits = 0
            tokens = 0
            docs_with_hit = 0
            for doc in source_docs:
                doc_hits = len(pattern.findall(doc.text))
                hits += doc_hits
                tokens += max(1, doc.token_count)
                docs_with_hit += int(doc_hits > 0)
            rows.append(
                {
                    "feature": spec["feature"],
                    "candidate": spec["candidate"],
                    "cluster": spec["cluster"],
                    "corpus": corpus,
                    "source": source,
                    "documents": len(source_docs),
                    "documents_with_hit": docs_with_hit,
                    "hits": hits,
                    "tokens": tokens,
                    "per_1k_tokens": hits / max(1, tokens) * 1000.0,
                    "doc_rate": docs_with_hit / max(1, len(source_docs)),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "candidate",
        "feature",
        "cluster",
        "auto_label",
        "source",
        "corpus",
        "model_count",
        "reference_count",
        "model_doc_count",
        "reference_doc_count",
        "model_per_1k_tokens",
        "reference_per_1k_tokens",
        "lift_vs_reference",
        "log2_lift_vs_reference",
        "score",
        "known_tier1_overlap",
        "auto_perceptibility_proxy",
    ]
    columns = [column for column in preferred if column in columns] + [
        column for column in columns if column not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_scope_decision(path: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Phase 4 Bounded Scope Decision",
        "",
        "The literal Phase 4 plan calls for ModernBERT-large fine-tuning, integrated gradients over at least 10k documents, span embedding with GTE-large, HDBSCAN clustering, LLM/manual cluster labeling, and 100 human annotations per cluster. That is not a reliable sub-24h A100 task once data preparation, model training, attribution, and human validation are included.",
        "",
        "This run therefore executes the bounded path allowed by the active compute instruction: no A100 generation or training; reuse existing Phase 2/3 generations; discover candidate spans with an interpretable model-vs-reference n-gram lift detector; cluster with deterministic lexical heuristics; produce candidate matcher specs and a lightweight rerun census.",
        "",
        "## Compute",
        "",
        "- A100 time used: 0 hours.",
        "- New generation: none.",
        "- Detector training: deferred.",
        "- Human annotation: deferred; this run reports only an auto-perceptibility proxy.",
        "",
        "## Inputs",
        "",
    ]
    for spec in args.reference_jsonl:
        lines.append(f"- Reference: `{spec}`")
    for spec in args.generation_jsonl:
        lines.append(f"- Generation: `{spec}`")
    lines.extend(
        [
            "",
            "## Loaded Corpus",
            "",
            f"- Reference documents: {summary['reference_documents']}",
            f"- Model documents: {summary['model_documents']}",
            f"- Reference tokens: {summary['reference_tokens']}",
            f"- Model tokens: {summary['model_tokens']}",
            f"- Vocabulary size: {summary['vocabulary_size']}",
            f"- Candidate count before top-k: {summary['candidate_count_before_top_k']}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(
    path: Path,
    *,
    candidates: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Phase 4 Bounded Discovery Summary",
        "",
        "This is a bounded detector-discovery substitute, not the full ModernBERT/integrated-gradients/human-validation Phase 4.",
        "",
        f"- Reference documents: {summary['reference_documents']}",
        f"- Model documents: {summary['model_documents']}",
        f"- Candidate spans retained: {len(candidates)}",
        f"- Clusters: {len(clusters)}",
        f"- Matcher specs emitted: {len(specs)}",
        "",
        "## Top Candidate Spans",
        "",
        "| Candidate | Cluster | Model /1k | Reference /1k | Lift | Proxy |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in candidates[:20]:
        lines.append(
            "| {candidate} | {cluster} | {model:.3f} | {ref:.3f} | {lift:.2f} | {proxy} |".format(
                candidate=row["candidate"],
                cluster=row["cluster"],
                model=float(row["model_per_1k_tokens"]),
                ref=float(row["reference_per_1k_tokens"]),
                lift=float(row["lift_vs_reference"]),
                proxy=row["auto_perceptibility_proxy"],
            )
        )
    lines.extend(
        [
            "",
            "## Cluster Summary",
            "",
            "| Cluster | Label | Candidates | Top Candidate | Lift |",
            "|---|---|---:|---|---:|",
        ]
    )
    for row in clusters:
        lines.append(
            "| {cluster} | {label} | {count} | {top} | {lift:.2f} |".format(
                cluster=row["cluster"],
                label=row["auto_label"],
                count=row["candidate_count"],
                top=row["top_candidate"],
                lift=float(row["top_lift_vs_reference"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- These candidates are detector-style proposals, not promoted Tier-3 features yet.",
            "- Promotion still requires the original precision target: 200 hand-checked matches with target precision at least 0.9.",
            "- The auto-perceptibility proxy is a triage label only; it is not the Shaib-taxonomy human validation requested in the full Phase 4.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    reference_specs = [
        _parse_spec(raw, default_field="text", kind="reference") for raw in args.reference_jsonl
    ]
    generation_specs = [
        _parse_spec(raw, default_field="generation", kind="generation") for raw in args.generation_jsonl
    ]
    reference_docs = [
        doc
        for spec in reference_specs
        for doc in _read_jsonl(
            spec, max_docs=args.max_docs_per_source, feature_text_mode=args.feature_text_mode
        )
    ]
    model_docs = [
        doc
        for spec in generation_specs
        for doc in _read_jsonl(
            spec, max_docs=args.max_docs_per_source, feature_text_mode=args.feature_text_mode
        )
    ]
    if not reference_docs:
        raise ValueError("no reference documents loaded")
    if not model_docs:
        raise ValueError("no model documents loaded")

    candidates, examples_by_candidate, summary = _candidate_rows(
        model_docs=model_docs,
        reference_docs=reference_docs,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_model_docs=args.min_model_docs,
        min_model_count=args.min_model_count,
        min_lift=args.min_lift,
        top_k=args.top_k,
    )
    clusters = _cluster_rows(candidates)
    examples = _example_rows(
        candidates,
        examples_by_candidate,
        max_examples=args.max_examples_per_candidate,
    )
    specs = _matcher_specs(candidates, limit=args.matcher_top_k)
    census = _census_rows(specs, [*reference_docs, *model_docs])

    output_dir: Path = args.output_dir
    _write_csv(output_dir / "phase4_candidate_spans.csv", candidates)
    _write_csv(output_dir / "phase4_candidate_clusters.csv", clusters)
    _write_jsonl(output_dir / "phase4_candidate_examples.jsonl", examples)
    _write_json(output_dir / "phase4_matcher_specs.json", specs)
    _write_csv(output_dir / "phase4_candidate_census.csv", census)
    _write_scope_decision(output_dir / "phase4_scope_decision.md", summary, args)
    _write_summary(
        output_dir / "phase4_bounded_discovery_summary.md",
        candidates=candidates,
        clusters=clusters,
        specs=specs,
        summary=summary,
    )
    return {
        "candidate_count": len(candidates),
        "cluster_count": len(clusters),
        "matcher_count": len(specs),
        **summary,
    }


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(
        "Wrote Phase 4 bounded discovery outputs to "
        f"{args.output_dir} ({summary['candidate_count']} candidates, "
        f"{summary['cluster_count']} clusters, {summary['matcher_count']} matchers)"
    )


if __name__ == "__main__":
    main()
