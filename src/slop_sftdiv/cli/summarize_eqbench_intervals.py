from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slop_sftdiv.cli.eqbench_slop_score import _record_text
from slop_sftdiv.features.eqbench_slop import (
    _eqbench_weighted_score,
    eqbench_tokens,
    iter_eqbench_contrast_hits,
    iter_eqbench_trigram_hits,
    iter_eqbench_word_hits,
)


DEFAULT_PANELS = (
    (
        "olmo3",
        "base",
        "artifacts/phase2/generations/"
        "olmo3_base_promptpkg5000_free_run_512prompt_8comp_t1_batched1024.jsonl",
    ),
    (
        "olmo3",
        "sft",
        "artifacts/phase2/generations/"
        "olmo3_sft_promptpkg5000_free_run_512prompt_8comp_t1_batched1024.jsonl",
    ),
    (
        "olmo3",
        "dpo",
        "artifacts/phase2/generations/"
        "olmo3_dpo_promptpkg5000_free_run_512prompt_8comp_t1_batched1024.jsonl",
    ),
    (
        "olmo3",
        "final",
        "artifacts/phase2/generations/"
        "olmo3_final_promptpkg5000_free_run_512prompt_8comp_t1_batched1024.jsonl",
    ),
    (
        "smollm3_no_think",
        "base",
        "artifacts/phase2/generations/"
        "smollm3_no_think_promptpkg512_chat_production_grid_base_"
        "smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256.jsonl",
    ),
    (
        "smollm3_no_think",
        "sft",
        "artifacts/phase2/generations/"
        "smollm3_no_think_promptpkg512_chat_production_grid_sft_"
        "smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256.jsonl",
    ),
    (
        "smollm3_no_think",
        "dpo",
        "artifacts/phase2/generations/"
        "smollm3_no_think_promptpkg512_chat_production_grid_dpo_"
        "smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256.jsonl",
    ),
    (
        "smollm3_no_think",
        "final",
        "artifacts/phase2/generations/"
        "smollm3_no_think_promptpkg512_chat_production_grid_final_"
        "smoltalk2-everyday-no-think-promptpkg512-chat_free_run_512prompt_8comp_t1_batched256.jsonl",
    ),
)


@dataclass(frozen=True)
class EqbenchCounts:
    words: int
    trigrams: int
    contrasts: int
    tokens: int
    chars: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bootstrap EQ-Bench Slop Score intervals from cached generation panels."
    )
    parser.add_argument(
        "--panel",
        action="append",
        default=[],
        metavar="LADDER,STAGE,PATH",
        help="Generation JSONL panel. Defaults to the paper Figure 2 panels.",
    )
    parser.add_argument("--text-field", action="append", default=["generation"])
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.md"),
    )
    return parser


def run_summarize_eqbench_intervals(args: argparse.Namespace) -> list[dict[str, Any]]:
    panels = _parse_panels(args.panel)
    rows = []
    rng = random.Random(args.seed)
    for ladder, stage, path in panels:
        counts = _load_panel_counts(
            path,
            text_fields=args.text_field,
            min_chars=args.min_chars,
        )
        if not counts:
            raise ValueError(f"{path} has no valid outputs after min_chars={args.min_chars}")
        point = _score_from_counts(counts)
        boot_scores = _bootstrap_scores(counts, args.bootstrap_samples, rng)
        rows.append(
            {
                "ladder": ladder,
                "stage": stage,
                "eqbench_slop_score": point["eqbench_slop_score"],
                "eqbench_slop_score_ci_low": _quantile(boot_scores, 0.025),
                "eqbench_slop_score_ci_high": _quantile(boot_scores, 0.975),
                "slop_list_matches_per_1k_words": point["slop_list_matches_per_1k_words"],
                "slop_trigram_matches_per_1k_words": point[
                    "slop_trigram_matches_per_1k_words"
                ],
                "not_x_but_y_per_1k_chars": point["not_x_but_y_per_1k_chars"],
                "sample_count": _jsonl_record_count(path),
                "valid_sample_count": len(counts),
                "bootstrap_samples": args.bootstrap_samples,
                "min_chars": args.min_chars,
                "seed": args.seed,
                "panel_path": str(path),
            }
        )
    _write_csv(args.output, rows)
    _write_markdown(args.summary_output, rows)
    return rows


def _parse_panels(raw_panels: list[str]) -> list[tuple[str, str, Path]]:
    if not raw_panels:
        return [(ladder, stage, Path(path)) for ladder, stage, path in DEFAULT_PANELS]
    panels = []
    for raw_panel in raw_panels:
        parts = raw_panel.split(",", 2)
        if len(parts) != 3:
            raise ValueError(f"panel must be LADDER,STAGE,PATH: {raw_panel}")
        ladder, stage, raw_path = parts
        panels.append((ladder, stage, Path(raw_path)))
    return panels


def _load_panel_counts(path: Path, *, text_fields: list[str], min_chars: int) -> list[EqbenchCounts]:
    counts = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            text = _record_text(record, text_fields=text_fields)
            if text is None or len(text) <= min_chars:
                continue
            counts.append(_counts_for_text(text))
    return counts


def _counts_for_text(text: str) -> EqbenchCounts:
    return EqbenchCounts(
        words=len(iter_eqbench_word_hits(text)),
        trigrams=len(iter_eqbench_trigram_hits(text)),
        contrasts=len(iter_eqbench_contrast_hits(text)),
        tokens=len(eqbench_tokens(text)),
        chars=len(text),
    )


def _score_from_counts(counts: list[EqbenchCounts]) -> dict[str, float]:
    words = sum(item.words for item in counts)
    trigrams = sum(item.trigrams for item in counts)
    contrasts = sum(item.contrasts for item in counts)
    tokens = sum(item.tokens for item in counts)
    chars = sum(item.chars for item in counts)
    word_rate = (words / tokens) * 1000.0 if tokens else 0.0
    trigram_rate = (trigrams / tokens) * 1000.0 if tokens else 0.0
    contrast_rate = (contrasts / chars) * 1000.0 if chars else 0.0
    return {
        "eqbench_slop_score": _eqbench_weighted_score(word_rate, trigram_rate, contrast_rate),
        "slop_list_matches_per_1k_words": word_rate,
        "slop_trigram_matches_per_1k_words": trigram_rate,
        "not_x_but_y_per_1k_chars": contrast_rate,
    }


def _bootstrap_scores(
    counts: list[EqbenchCounts], bootstrap_samples: int, rng: random.Random
) -> list[float]:
    sample_size = len(counts)
    scores = []
    for _ in range(bootstrap_samples):
        sample = [counts[rng.randrange(sample_size)] for _ in range(sample_size)]
        scores.append(_score_from_counts(sample)["eqbench_slop_score"])
    return scores


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile of empty values")
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _jsonl_record_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "ladder",
        "stage",
        "eqbench_slop_score",
        "eqbench_slop_score_ci_low",
        "eqbench_slop_score_ci_high",
        "slop_list_matches_per_1k_words",
        "slop_trigram_matches_per_1k_words",
        "not_x_but_y_per_1k_chars",
        "sample_count",
        "valid_sample_count",
        "bootstrap_samples",
        "min_chars",
        "seed",
        "panel_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 2 EQ-Bench Slop Score Bootstrap Intervals",
        "",
        "Intervals are nonparametric bootstrap intervals over valid generated outputs. "
        "Each bootstrap replicate resamples valid outputs with replacement, sums EQ-Bench "
        "component counts, recomputes the benchmark-weighted aggregate score, and reports "
        "the 2.5th and 97.5th percentiles. The score remains an aggregate diagnostic, not "
        "the causal propensity estimand.",
        "",
        "| Ladder | Stage | Score | 95% bootstrap CI | Valid / total |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {ladder} | {stage} | {score:.3f} | [{low:.3f}, {high:.3f}] | {valid}/{total} |".format(
                ladder=row["ladder"],
                stage=row["stage"],
                score=row["eqbench_slop_score"],
                low=row["eqbench_slop_score_ci_low"],
                high=row["eqbench_slop_score_ci_high"],
                valid=row["valid_sample_count"],
                total=row["sample_count"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_summarize_eqbench_intervals(args)
    print(f"Wrote {len(rows)} EQ-Bench interval rows to {args.output}")
    if args.summary_output is not None:
        print(f"Wrote interval summary to {args.summary_output}")


if __name__ == "__main__":
    main()
