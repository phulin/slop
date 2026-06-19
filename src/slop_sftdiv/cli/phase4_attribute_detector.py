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

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from slop_sftdiv.generation_text import (
    FeatureTextMode,
    feature_text_for_mode,
    strip_chat_role_markers,
)


ALPHA_RE = re.compile(r"[A-Za-z]")
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class AttributionDoc:
    source: str
    doc_id: str
    text: str
    label: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run integrated-gradients attribution for the Phase 4 detector."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--generation-jsonl",
        action="append",
        default=[],
        metavar="LABEL=PATH[:FIELD]",
        help="Generation JSONL to attribute as LLM-class examples.",
    )
    parser.add_argument("--max-docs-per-source", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--ig-steps", type=int, default=8)
    parser.add_argument("--top-tokens-per-doc", type=int, default=24)
    parser.add_argument("--top-spans-per-doc", type=int, default=8)
    parser.add_argument("--span-max-tokens", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--feature-text-mode", choices=["raw", "final_answer"], default="final_answer")
    return parser


def _parse_spec(raw: str, *, default_field: str) -> tuple[str, Path, str]:
    if "=" not in raw:
        raise ValueError(f"expected LABEL=PATH[:FIELD], got {raw!r}")
    label, rest = raw.split("=", 1)
    path = Path(rest)
    if path.exists():
        return label, path, default_field
    if ":" in rest:
        path_text, field = rest.rsplit(":", 1)
        return label, Path(path_text), field
    return label, Path(rest), default_field


def _load_generation_docs(
    specs: list[str],
    *,
    max_docs_per_source: int,
    feature_text_mode: str,
) -> list[AttributionDoc]:
    docs: list[AttributionDoc] = []
    for raw in specs:
        source, path, field = _parse_spec(raw, default_field="generation")
        loaded = 0
        with path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if loaded >= max_docs_per_source:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                text = strip_chat_role_markers(
                    feature_text_for_mode(
                        str(row.get(field, "") or ""),
                        cast(FeatureTextMode, feature_text_mode),
                    )
                ).strip()
                if not text:
                    continue
                docs.append(
                    AttributionDoc(
                        source=source,
                        doc_id=str(row.get("record_id") or row.get("doc_id") or row_index),
                        text=text,
                        label=1,
                    )
                )
                loaded += 1
    return docs


def _integrated_gradients(
    *,
    model: Any,
    tokenizer: Any,
    doc: AttributionDoc,
    device: torch.device,
    max_length: int,
    steps: int,
    target_label: int = 1,
) -> tuple[list[float], list[str], float]:
    encoded = tokenizer(
        doc.text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    special_mask = encoded["special_tokens_mask"][0].tolist()
    embedding_layer = model.get_input_embeddings()
    input_embeddings = embedding_layer(input_ids).detach()
    baseline = torch.zeros_like(input_embeddings)
    total_gradients = torch.zeros_like(input_embeddings)

    model.eval()
    for step_index in range(1, steps + 1):
        alpha = float(step_index) / float(steps)
        scaled = (baseline + alpha * (input_embeddings - baseline)).detach()
        scaled.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        outputs = model(inputs_embeds=scaled, attention_mask=attention_mask)
        target_logit = outputs.logits[0, target_label]
        target_logit.backward()
        if scaled.grad is None:
            raise RuntimeError("integrated gradients failed: no gradient for inputs_embeds")
        total_gradients += scaled.grad.detach()

    average_gradients = total_gradients / float(steps)
    attributions = ((input_embeddings - baseline) * average_gradients).sum(dim=-1)[0]
    with torch.inference_mode():
        probabilities = torch.softmax(model(input_ids=input_ids, attention_mask=attention_mask).logits.float(), dim=-1)
    token_scores = attributions.detach().float().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())
    for index, is_special in enumerate(special_mask):
        if is_special:
            token_scores[index] = float("-inf")
    return token_scores, tokens, float(probabilities[0, target_label].detach().cpu())


def _doc_span_rows(
    *,
    tokenizer: Any,
    doc: AttributionDoc,
    token_scores: list[float],
    tokens: list[str],
    llm_probability: float,
    top_tokens: int,
    top_spans: int,
    span_max_tokens: int,
) -> list[dict[str, Any]]:
    ranked_positions = [
        index
        for index, score in sorted(enumerate(token_scores), key=lambda item: item[1], reverse=True)
        if math.isfinite(score)
    ][:top_tokens]
    candidates: dict[str, dict[str, Any]] = {}
    token_count = len(tokens)
    for position in ranked_positions:
        for start in range(max(0, position - span_max_tokens + 1), position + 1):
            for end in range(position + 1, min(token_count, start + span_max_tokens) + 1):
                span_scores = token_scores[start:end]
                if not all(math.isfinite(score) for score in span_scores):
                    continue
                text = _clean_decoded_span(tokenizer.convert_tokens_to_string(tokens[start:end]))
                if _skip_span(text):
                    continue
                score = sum(span_scores) / math.sqrt(end - start)
                existing = candidates.get(text)
                if existing is None or score > existing["attribution_score"]:
                    candidates[text] = {
                        "source": doc.source,
                        "doc_id": doc.doc_id,
                        "span": text,
                        "token_start": start,
                        "token_end": end,
                        "attribution_score": score,
                        "llm_probability": llm_probability,
                    }
    return sorted(candidates.values(), key=lambda row: row["attribution_score"], reverse=True)[
        :top_spans
    ]


def _clean_decoded_span(text: str) -> str:
    return SPACE_RE.sub(" ", text.replace("Ġ", " ").replace("▁", " ")).strip()


def _skip_span(text: str) -> bool:
    if len(text) < 3:
        return True
    if not ALPHA_RE.search(text):
        return True
    if len(text.split()) == 1 and len(text) < 4:
        return True
    return False


def _cluster_for_span(span: str) -> str:
    lower = span.lower()
    if lower.startswith(("here are", "are a few", "are some")):
        return "offer_examples"
    if lower.startswith(("let's", "we can", "we need", "we should")):
        return "assistant_process_framing"
    if "thank" in lower or "thanks" in lower:
        return "politeness_thanks"
    if "hope this" in lower or "hope that" in lower:
        return "stock_helpful_closure"
    if "let me know" in lower or "know if" in lower:
        return "followup_offer"
    if lower.startswith(("overall", "in conclusion", "to summarize")):
        return "summary_closure"
    if lower.startswith(("such as", "for example")):
        return "explanatory_transition"
    if any(word in lower.split() for word in ("should", "need", "consider", "ensure", "make")):
        return "prescriptive_instruction"
    return "attributed_span"


def _aggregate_span_rows(span_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in span_rows:
        grouped[str(row["span"]).lower()].append(row)
    out = []
    for span, rows in grouped.items():
        source_counts = Counter(str(row["source"]) for row in rows)
        out.append(
            {
                "span": span,
                "cluster": _cluster_for_span(span),
                "doc_count": len({(row["source"], row["doc_id"]) for row in rows}),
                "hit_count": len(rows),
                "mean_attribution_score": sum(float(row["attribution_score"]) for row in rows)
                / len(rows),
                "max_attribution_score": max(float(row["attribution_score"]) for row in rows),
                "mean_llm_probability": sum(float(row["llm_probability"]) for row in rows)
                / len(rows),
                "sources": ";".join(f"{source}:{count}" for source, count in sorted(source_counts.items())),
            }
        )
    return sorted(out, key=lambda row: (int(row["doc_count"]), float(row["mean_attribution_score"])), reverse=True)


def _cluster_rows(aggregate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in aggregate_rows:
        grouped[str(row["cluster"])].append(row)
    out = []
    for cluster, rows in grouped.items():
        top = rows[0]
        out.append(
            {
                "cluster": cluster,
                "span_count": len(rows),
                "total_doc_count": sum(int(row["doc_count"]) for row in rows),
                "top_span": top["span"],
                "top_span_doc_count": top["doc_count"],
                "top_span_mean_attribution_score": top["mean_attribution_score"],
            }
        )
    return sorted(out, key=lambda row: (int(row["total_doc_count"]), float(row["top_span_mean_attribution_score"])), reverse=True)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "cluster",
        "span",
        "source",
        "doc_id",
        "doc_count",
        "hit_count",
        "attribution_score",
        "mean_attribution_score",
        "llm_probability",
        "mean_llm_probability",
    ]
    columns = [column for column in preferred if column in columns] + [
        column for column in columns if column not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = cast(Any, AutoTokenizer.from_pretrained(args.checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(device)
    docs = _load_generation_docs(
        args.generation_jsonl,
        max_docs_per_source=args.max_docs_per_source,
        feature_text_mode=args.feature_text_mode,
    )
    span_rows: list[dict[str, Any]] = []
    for doc_index, doc in enumerate(docs, start=1):
        token_scores, tokens, llm_probability = _integrated_gradients(
            model=model,
            tokenizer=tokenizer,
            doc=doc,
            device=device,
            max_length=args.max_length,
            steps=args.ig_steps,
            target_label=1,
        )
        rows = _doc_span_rows(
            tokenizer=tokenizer,
            doc=doc,
            token_scores=token_scores,
            tokens=tokens,
            llm_probability=llm_probability,
            top_tokens=args.top_tokens_per_doc,
            top_spans=args.top_spans_per_doc,
            span_max_tokens=args.span_max_tokens,
        )
        for row in rows:
            row["doc_index"] = doc_index
        span_rows.extend(rows)
        if args.progress_every > 0 and (doc_index == 1 or doc_index % args.progress_every == 0):
            print(
                f"Attributed {doc_index}/{len(docs)} documents "
                f"({len(span_rows)} doc-span rows)",
                flush=True,
            )

    aggregate_rows = _aggregate_span_rows(span_rows)
    cluster_rows = _cluster_rows(aggregate_rows)
    output_dir: Path = args.output_dir
    _write_csv(output_dir / "phase4_detector_ig_doc_spans.csv", span_rows)
    _write_csv(output_dir / "phase4_detector_ig_span_summary.csv", aggregate_rows)
    _write_csv(output_dir / "phase4_detector_ig_clusters.csv", cluster_rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "documents_attributed": len(docs),
        "ig_steps": args.ig_steps,
        "max_length": args.max_length,
        "doc_span_rows": len(span_rows),
        "unique_spans": len(aggregate_rows),
        "clusters": len(cluster_rows),
    }
    _write_summary(output_dir / "phase4_detector_ig_summary.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(
        "Wrote Phase 4 detector attribution outputs to "
        f"{args.output_dir} ({summary['documents_attributed']} docs, "
        f"{summary['unique_spans']} unique spans)"
    )


if __name__ == "__main__":
    main()
