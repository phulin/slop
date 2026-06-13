from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from slop_sftdiv.propensity import (
    PHASE2_OPPORTUNITY_SPECS,
    FeatureOpportunity,
    _opportunity_offsets,
    iter_feature_opportunities,
)
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


OUTPUT_COLUMNS = [
    "mode",
    "repeat",
    "input",
    "features",
    "measurement_rows",
    "chars",
    "opportunities",
    "reference_initiations",
    "wall_seconds",
    "measurement_rows_per_sec",
    "chars_per_sec",
    "opportunities_per_sec",
    "peak_rss_mb",
]


@dataclass(frozen=True)
class MeasurementText:
    role: str
    text: str


@dataclass
class BenchmarkMetrics:
    mode: str
    repeat: int
    input_name: str
    features: tuple[str, ...]
    measurement_rows: int = 0
    chars: int = 0
    opportunities: int = 0
    reference_initiations: int = 0
    wall_seconds: float = 0.0

    @property
    def measurement_rows_per_sec(self) -> float:
        return self.measurement_rows / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def chars_per_sec(self) -> float:
        return self.chars / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def opportunities_per_sec(self) -> float:
        return self.opportunities / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def row(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "repeat": self.repeat,
            "input": self.input_name,
            "features": ",".join(self.features),
            "measurement_rows": self.measurement_rows,
            "chars": self.chars,
            "opportunities": self.opportunities,
            "reference_initiations": self.reference_initiations,
            "wall_seconds": self.wall_seconds,
            "measurement_rows_per_sec": self.measurement_rows_per_sec,
            "chars_per_sec": self.chars_per_sec,
            "opportunities_per_sec": self.opportunities_per_sec,
            "peak_rss_mb": _peak_rss_mb(),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark CPU-side Phase 2 opportunity extraction on local JSONL inputs."
    )
    parser.add_argument("--input", action="append", required=True, help="Local JSONL input path.")
    parser.add_argument("--text-field", default=None, help="Text field for single-text rows.")
    parser.add_argument("--sample-size", type=int, default=1024, help="Maximum JSONL rows per input.")
    parser.add_argument("--max-scan", type=int, default=None, help="Maximum JSONL rows to scan.")
    parser.add_argument("--feature", action="append", default=[], help="Feature ID; repeatable.")
    parser.add_argument("--max-token-start-opportunities", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--include-offsets-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also time raw opportunity-offset enumeration without matcher hit extraction.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-benchmarks")
    parser.add_argument("--wandb-job-type", default="opportunity-extraction-benchmark")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    _validate_args(args)
    features = _selected_features(args.feature)
    rows: list[dict[str, Any]] = []
    inputs = [Path(raw_input) for raw_input in args.input]
    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage2", "phase2", "benchmark", "opportunity-extraction", *args.wandb_tag],
        config={
            "inputs": [str(path) for path in inputs],
            "text_field": args.text_field,
            "sample_size": args.sample_size,
            "max_scan": args.max_scan,
            "features": features,
            "max_token_start_opportunities": args.max_token_start_opportunities,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "include_offsets_only": args.include_offsets_only,
            "output": str(args.output),
        },
    )

    try:
        for input_path in inputs:
            texts = list(
                _iter_input_texts(
                    input_path,
                    text_field=args.text_field,
                    sample_size=args.sample_size,
                    max_scan=args.max_scan,
                )
            )
            for _index in range(args.warmup):
                _run_full_pass(
                    texts,
                    input_name=str(input_path),
                    repeat=-1,
                    features=features,
                    max_token_start_opportunities=args.max_token_start_opportunities,
                )
                if args.include_offsets_only:
                    _run_offsets_only_pass(
                        texts,
                        input_name=str(input_path),
                        repeat=-1,
                        features=features,
                        max_token_start_opportunities=args.max_token_start_opportunities,
                    )

            for repeat in range(args.repeats):
                rows.append(
                    _run_full_pass(
                        texts,
                        input_name=str(input_path),
                        repeat=repeat,
                        features=features,
                        max_token_start_opportunities=args.max_token_start_opportunities,
                    ).row()
                )
                if args.include_offsets_only:
                    rows.append(
                        _run_offsets_only_pass(
                            texts,
                            input_name=str(input_path),
                            repeat=repeat,
                            features=features,
                            max_token_start_opportunities=args.max_token_start_opportunities,
                        ).row()
                    )

        _write_csv(args.output, rows)
        log_summary_table(run, "opportunity_extraction_benchmark", rows)
        if rows:
            fastest_full = max(
                (
                    row["opportunities_per_sec"]
                    for row in rows
                    if row["mode"] == "full"
                ),
                default=0.0,
            )
            run.log(
                {
                    "opportunity_extraction/rows": len(rows),
                    "opportunity_extraction/fastest_full_opportunities_per_sec": fastest_full,
                }
            )
        return rows
    finally:
        run.finish()


def _validate_args(args: argparse.Namespace) -> None:
    if args.sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if args.max_scan is not None and args.max_scan <= 0:
        raise ValueError("max_scan must be positive when set")
    if args.max_token_start_opportunities is not None and args.max_token_start_opportunities < 0:
        raise ValueError("max_token_start_opportunities must be non-negative when set")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")


def _selected_features(raw_features: Iterable[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(raw_features))
    if not selected:
        return tuple(sorted(PHASE2_OPPORTUNITY_SPECS))
    return tuple(feature for feature in selected if feature in PHASE2_OPPORTUNITY_SPECS)


def _iter_input_texts(
    input_path: Path,
    *,
    text_field: str | None,
    sample_size: int,
    max_scan: int | None,
) -> Iterable[MeasurementText]:
    with input_path.open(encoding="utf-8") as handle:
        yielded_rows = 0
        for line_number, line in enumerate(handle, start=1):
            if max_scan is not None and line_number > max_scan:
                break
            if yielded_rows >= sample_size:
                break
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            yielded_rows += 1
            yield from _measurement_texts(row, text_field=text_field)


def _measurement_texts(row: dict[str, Any], *, text_field: str | None) -> list[MeasurementText]:
    if isinstance(row.get("chosen"), str) and isinstance(row.get("rejected"), str):
        return [
            MeasurementText(role="chosen", text=row["chosen"]),
            MeasurementText(role="rejected", text=row["rejected"]),
        ]
    fields = [text_field] if text_field else ["text", "response", "completion", "content"]
    for field in fields:
        value = row.get(field)
        if isinstance(value, str):
            return [MeasurementText(role=field, text=value)]
    return []


def _run_full_pass(
    texts: list[MeasurementText],
    *,
    input_name: str,
    repeat: int,
    features: tuple[str, ...],
    max_token_start_opportunities: int | None,
) -> BenchmarkMetrics:
    metrics = BenchmarkMetrics(
        mode="full",
        repeat=repeat,
        input_name=input_name,
        features=features,
    )
    started_at = time.perf_counter()
    for item in tqdm(texts, desc=f"opportunity-full:{Path(input_name).name}", unit="text", leave=False):
        opportunities = iter_feature_opportunities(
            item.text,
            features=features,
            max_token_start_opportunities=max_token_start_opportunities,
        )
        _update_metrics_from_opportunities(metrics, item.text, opportunities)
    metrics.wall_seconds = time.perf_counter() - started_at
    return metrics


def _run_offsets_only_pass(
    texts: list[MeasurementText],
    *,
    input_name: str,
    repeat: int,
    features: tuple[str, ...],
    max_token_start_opportunities: int | None,
) -> BenchmarkMetrics:
    metrics = BenchmarkMetrics(
        mode="offsets_only",
        repeat=repeat,
        input_name=input_name,
        features=features,
    )
    specs = [PHASE2_OPPORTUNITY_SPECS[feature] for feature in features]
    started_at = time.perf_counter()
    for item in tqdm(texts, desc=f"opportunity-offsets:{Path(input_name).name}", unit="text", leave=False):
        metrics.measurement_rows += 1
        metrics.chars += len(item.text)
        for spec in specs:
            offsets = _opportunity_offsets(
                item.text,
                spec.opportunity_kind,
                max_token_start_opportunities=max_token_start_opportunities,
            )
            metrics.opportunities += len(offsets)
    metrics.wall_seconds = time.perf_counter() - started_at
    return metrics


def _update_metrics_from_opportunities(
    metrics: BenchmarkMetrics,
    text: str,
    opportunities: list[FeatureOpportunity],
) -> None:
    metrics.measurement_rows += 1
    metrics.chars += len(text)
    metrics.opportunities += len(opportunities)
    metrics.reference_initiations += sum(
        1 for opportunity in opportunities if opportunity.reference_initiates
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    rows = run_benchmark(args)
    print(f"Wrote {len(rows)} benchmark rows to {args.output}")


if __name__ == "__main__":
    main()
