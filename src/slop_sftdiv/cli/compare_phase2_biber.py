from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slop_sftdiv.features.biber_lite import BIBER_LITE_PATTERNS, extract_biber_lite_features
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


ROLE_COLUMNS = {
    "pretrain_document": "pretrain",
    "target_response": "sft_target",
    "chosen": "dpo_chosen",
    "rejected": "dpo_rejected",
}
STAGE_LABELS = {
    "base": "Base",
    "sft": "SFT",
    "dpo": "DPO",
    "final": "Final/RLVR",
}


@dataclass(frozen=True)
class GenerationSpec:
    stage: str
    path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Biber-lite rates in Phase 2 generations to Phase 1 corpus samples."
    )
    parser.add_argument(
        "--generation-cache",
        action="append",
        required=True,
        help="Stage-labelled generation JSONL in the form stage=path.",
    )
    parser.add_argument(
        "--corpus-rates",
        type=Path,
        required=True,
        help="Phase 1 feature-rate table, CSV or Parquet.",
    )
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-analysis")
    parser.add_argument("--wandb-job-type", default="biber-generation-comparison")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _parse_generation_specs(raw_specs: list[str]) -> list[GenerationSpec]:
    specs: list[GenerationSpec] = []
    seen: set[str] = set()
    for raw_spec in raw_specs:
        if "=" not in raw_spec:
            raise ValueError(f"generation cache must be stage=path, got {raw_spec!r}")
        stage, raw_path = raw_spec.split("=", 1)
        stage = stage.strip()
        if not stage:
            raise ValueError(f"empty stage in generation cache spec {raw_spec!r}")
        if stage in seen:
            raise ValueError(f"duplicate stage {stage!r}")
        seen.add(stage)
        specs.append(GenerationSpec(stage=stage, path=Path(raw_path)))
    return specs


def _selected_features(features: list[str]) -> list[str]:
    selected = features or sorted(BIBER_LITE_PATTERNS)
    unknown = sorted(set(selected) - set(BIBER_LITE_PATTERNS))
    if unknown:
        raise ValueError(f"unknown Biber-lite feature(s): {', '.join(unknown)}")
    return sorted(dict.fromkeys(selected))


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_rate(count: int | float, denominator: int | float) -> float:
    return count / denominator if denominator else 0.0


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return math.inf if numerator > 0 else 0.0
    return numerator / denominator


def _load_corpus_rates(
    path: Path, selected_features: list[str]
) -> dict[str, dict[str, dict[str, float | int]]]:
    selected = set(selected_features)
    totals: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for row in _read_rows(path):
        feature = str(row.get("feature", ""))
        role = str(row.get("role", ""))
        if feature not in selected or role not in ROLE_COLUMNS:
            continue
        counter = totals[feature][role]
        counter["count"] += int(float(row.get("count") or 0))
        counter["docs"] += int(float(row.get("docs") or 0))
        counter["tokens"] += int(float(row.get("tokens") or 0))

    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for feature, roles in totals.items():
        out[feature] = {}
        for role, counter in roles.items():
            tokens = int(counter["tokens"])
            docs = int(counter["docs"])
            count = int(counter["count"])
            out[feature][role] = {
                "count": count,
                "docs": docs,
                "tokens": tokens,
                "per_1k_tokens": 1000.0 * _safe_rate(count, tokens),
                "per_doc": _safe_rate(count, docs),
            }
    return out


def _generation_counts(
    specs: list[GenerationSpec], selected_features: list[str]
) -> dict[tuple[str, str], Counter[str]]:
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    selected = set(selected_features)
    for spec in specs:
        with spec.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {spec.path}:{line_number}") from exc
                generation = str(row.get("generation", ""))
                analysis = extract_biber_lite_features(generation)
                for feature in selected_features:
                    counter = counts[(feature, spec.stage)]
                    counter["count"] += int(analysis.counts.get(feature, 0))
                    counter["docs"] += 1
                    counter["tokens"] += int(analysis.token_count)
                    if analysis.counts.get(feature, 0) > 0:
                        counter["docs_with_feature"] += 1
                for feature in set(analysis.counts) - selected:
                    _ = feature
    return counts


def _row(
    *,
    feature: str,
    stage: str,
    counter: Counter[str],
    corpus: dict[str, dict[str, dict[str, float | int]]],
) -> dict[str, Any]:
    tokens = int(counter["tokens"])
    docs = int(counter["docs"])
    count = int(counter["count"])
    generation_per_1k = 1000.0 * _safe_rate(count, tokens)
    generation_per_doc = _safe_rate(count, docs)
    row: dict[str, Any] = {
        "feature": feature,
        "stage": stage,
        "stage_label": STAGE_LABELS.get(stage, stage),
        "generation_count": count,
        "generation_docs": docs,
        "generation_tokens": tokens,
        "generation_docs_with_feature": int(counter["docs_with_feature"]),
        "generation_per_1k_tokens": generation_per_1k,
        "generation_per_doc": generation_per_doc,
        "generation_share_docs_with_feature": _safe_rate(int(counter["docs_with_feature"]), docs),
    }
    for role, prefix in ROLE_COLUMNS.items():
        baseline = corpus.get(feature, {}).get(role, {})
        baseline_per_1k = baseline.get("per_1k_tokens")
        row[f"{prefix}_per_1k_tokens"] = baseline_per_1k
        row[f"delta_vs_{prefix}_per_1k_tokens"] = (
            generation_per_1k - float(baseline_per_1k)
            if baseline_per_1k is not None
            else None
        )
        row[f"ratio_vs_{prefix}_per_1k_tokens"] = _safe_ratio(
            generation_per_1k,
            float(baseline_per_1k) if baseline_per_1k is not None else None,
        )
    return row


def compare_biber(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = _parse_generation_specs(args.generation_cache)
    selected_features = _selected_features(args.feature)
    corpus = _load_corpus_rates(args.corpus_rates, selected_features)
    generation = _generation_counts(specs, selected_features)
    rows = [
        _row(
            feature=feature,
            stage=spec.stage,
            counter=generation[(feature, spec.stage)],
            corpus=corpus,
        )
        for feature in selected_features
        for spec in specs
    ]
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "feature",
        "stage",
        "stage_label",
        "generation_count",
        "generation_docs",
        "generation_tokens",
        "generation_docs_with_feature",
        "generation_per_1k_tokens",
        "generation_per_doc",
        "generation_share_docs_with_feature",
        "pretrain_per_1k_tokens",
        "delta_vs_pretrain_per_1k_tokens",
        "ratio_vs_pretrain_per_1k_tokens",
        "sft_target_per_1k_tokens",
        "delta_vs_sft_target_per_1k_tokens",
        "ratio_vs_sft_target_per_1k_tokens",
        "dpo_chosen_per_1k_tokens",
        "delta_vs_dpo_chosen_per_1k_tokens",
        "ratio_vs_dpo_chosen_per_1k_tokens",
        "dpo_rejected_per_1k_tokens",
        "delta_vs_dpo_rejected_per_1k_tokens",
        "ratio_vs_dpo_rejected_per_1k_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    return f"{float(value):.3f}"


def _top_rows(rows: list[dict[str, Any]], key: str, *, limit: int = 8) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if row.get(key) is not None],
        key=lambda row: abs(float(row[key])),
        reverse=True,
    )[:limit]


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    features = sorted({row["feature"] for row in rows})
    stages = sorted({row["stage"] for row in rows})
    lines = [
        "# Phase 2 Biber-Lite Generation vs Corpus Comparison",
        "",
        f"Features: {len(features)}",
        f"Stages: {', '.join(stages)}",
        "",
        "Rates are regex-token-normalized per 1k tokens, matching the Phase 1 Biber-lite census.",
        "",
        "## Largest Generation-vs-SFT Target Deltas",
        "",
        "| Feature | Stage | Generation /1k | SFT target /1k | Delta | Ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in _top_rows(rows, "delta_vs_sft_target_per_1k_tokens"):
        lines.append(
            "| {feature} | {stage} | {gen} | {sft} | {delta} | {ratio} |".format(
                feature=row["feature"],
                stage=row["stage_label"],
                gen=_fmt(row["generation_per_1k_tokens"]),
                sft=_fmt(row["sft_target_per_1k_tokens"]),
                delta=_fmt(row["delta_vs_sft_target_per_1k_tokens"]),
                ratio=_fmt(row["ratio_vs_sft_target_per_1k_tokens"]),
            )
        )
    lines.extend(
        [
            "",
            "## Largest Generation-vs-DPO Chosen Deltas",
            "",
            "| Feature | Stage | Generation /1k | DPO chosen /1k | Delta | Ratio |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in _top_rows(rows, "delta_vs_dpo_chosen_per_1k_tokens"):
        lines.append(
            "| {feature} | {stage} | {gen} | {chosen} | {delta} | {ratio} |".format(
                feature=row["feature"],
                stage=row["stage_label"],
                gen=_fmt(row["generation_per_1k_tokens"]),
                chosen=_fmt(row["dpo_chosen_per_1k_tokens"]),
                delta=_fmt(row["delta_vs_dpo_chosen_per_1k_tokens"]),
                ratio=_fmt(row["ratio_vs_dpo_chosen_per_1k_tokens"]),
            )
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- These are Biber-lite regex proxies, not full pybiber categories.",
            "- Phase 2 rates are measured on generated completions; Phase 1 rates are corpus samples with different length and source mixtures.",
            "- This is a register comparison layer, not a teacher-forced AF or compounding analysis.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = compare_biber(args)
    _write_csv(args.output, rows)
    _write_summary(args.summary_output, rows)
    run_obj = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage2", "phase2", "biber-lite", *args.wandb_tag],
        config={
            "generation_caches": args.generation_cache,
            "corpus_rates": str(args.corpus_rates),
            "features": args.feature or sorted(BIBER_LITE_PATTERNS),
            "output": str(args.output),
            "summary_output": str(args.summary_output),
        },
    )
    try:
        run_obj.log(
            {
                "phase2_biber/rows": len(rows),
                "phase2_biber/features": len({row["feature"] for row in rows}),
                "phase2_biber/stages": len({row["stage"] for row in rows}),
            }
        )
        log_summary_table(run_obj, "phase2_biber_generation_vs_corpus", rows)
    finally:
        run_obj.finish()
    return rows


def main() -> None:
    args = build_parser().parse_args()
    rows = run(args)
    print(f"Wrote {len(rows)} Biber-lite comparison rows to {args.output}")


if __name__ == "__main__":
    main()
