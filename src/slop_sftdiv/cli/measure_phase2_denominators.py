from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from slop_sftdiv.propensity import PHASE2_OPPORTUNITY_SPECS, iter_feature_opportunities
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


OUTPUT_COLUMNS = [
    "input",
    "feature",
    "measurement_rows",
    "documents_with_opportunity",
    "documents_with_reference",
    "opportunities",
    "reference_initiations",
    "reference_rate",
    "references_per_document",
    "matched_subtypes_json",
]


@dataclass(frozen=True)
class MeasurementText:
    record_id: str
    role: str
    text: str


@dataclass
class FeatureSupport:
    feature: str
    measurement_rows: int = 0
    documents_with_opportunity: int = 0
    documents_with_reference: int = 0
    opportunities: int = 0
    reference_initiations: int = 0
    matched_subtypes: Counter[str] = field(default_factory=Counter)

    @property
    def reference_rate(self) -> float:
        return self.reference_initiations / self.opportunities if self.opportunities else 0.0

    @property
    def references_per_document(self) -> float:
        return self.reference_initiations / self.measurement_rows if self.measurement_rows else 0.0

    def row(self, *, input_name: str) -> dict[str, Any]:
        return {
            "input": input_name,
            "feature": self.feature,
            "measurement_rows": self.measurement_rows,
            "documents_with_opportunity": self.documents_with_opportunity,
            "documents_with_reference": self.documents_with_reference,
            "opportunities": self.opportunities,
            "reference_initiations": self.reference_initiations,
            "reference_rate": self.reference_rate,
            "references_per_document": self.references_per_document,
            "matched_subtypes_json": json.dumps(dict(sorted(self.matched_subtypes.items()))),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure per-feature Phase 2 opportunity/reference support."
    )
    parser.add_argument("--input", action="append", required=True, help="Local JSONL input path.")
    parser.add_argument("--text-field", default=None, help="Text field for single-text rows.")
    parser.add_argument("--sample-size", type=int, default=1024, help="Maximum JSONL rows per input.")
    parser.add_argument("--max-scan", type=int, default=None, help="Maximum JSONL rows to scan.")
    parser.add_argument("--feature", action="append", default=[], help="Feature ID; repeatable.")
    parser.add_argument("--max-token-start-opportunities", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-analysis")
    parser.add_argument("--wandb-job-type", default="measure-phase2-denominators")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def run_measure_phase2_denominators(args: argparse.Namespace) -> dict[str, Any]:
    _validate_args(args)
    requested_features, features, omitted_features = _select_features(args.feature)
    input_paths = [Path(raw_input) for raw_input in args.input]
    output_rows: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage2", "phase2", "denominators", *args.wandb_tag],
        config={
            "inputs": [str(path) for path in input_paths],
            "text_field": args.text_field,
            "sample_size": args.sample_size,
            "max_scan": args.max_scan,
            "requested_features": requested_features,
            "features": features,
            "omitted_features": omitted_features,
            "max_token_start_opportunities": args.max_token_start_opportunities,
            "output": str(args.output),
            "summary_output": str(args.summary_output) if args.summary_output else None,
        },
    )

    try:
        for input_path in input_paths:
            output_rows.extend(
                _measure_input(
                    input_path,
                    text_field=args.text_field,
                    sample_size=args.sample_size,
                    max_scan=args.max_scan,
                    features=features,
                    max_token_start_opportunities=args.max_token_start_opportunities,
                )
            )
        _write_csv(args.output, output_rows)
        wall_seconds = time.perf_counter() - started_at
        unique_measurement_texts = _unique_measurement_texts(output_rows)
        summary = {
            "rows": len(output_rows),
            "inputs": len(input_paths),
            "requested_features": len(requested_features),
            "features": len(features),
            "omitted_features": len(omitted_features),
            "omitted_feature_ids": omitted_features,
            "wall_seconds": wall_seconds,
            "measurement_rows": unique_measurement_texts,
            "feature_measurement_rows": sum(row["measurement_rows"] for row in output_rows),
            "opportunities": sum(row["opportunities"] for row in output_rows),
            "reference_initiations": sum(row["reference_initiations"] for row in output_rows),
            "features_with_references": sum(
                1 for row in output_rows if row["reference_initiations"] > 0
            ),
            "peak_rss_mb": _peak_rss_mb(),
            "output": str(args.output),
        }
        if args.summary_output:
            args.summary_output.parent.mkdir(parents=True, exist_ok=True)
            args.summary_output.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        run.log({f"phase2_denominators/{key}": value for key, value in summary.items()})
        log_summary_table(run, "phase2_denominator_support", output_rows)
        return summary
    finally:
        run.finish()


def _validate_args(args: argparse.Namespace) -> None:
    if args.sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if args.max_scan is not None and args.max_scan <= 0:
        raise ValueError("max_scan must be positive when set")
    if args.max_token_start_opportunities is not None and args.max_token_start_opportunities < 0:
        raise ValueError("max_token_start_opportunities must be non-negative when set")


def _select_features(raw_features: Iterable[str]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    selected = tuple(dict.fromkeys(raw_features))
    if not selected:
        all_features = tuple(sorted(PHASE2_OPPORTUNITY_SPECS))
        return all_features, all_features, ()
    measured = tuple(feature for feature in selected if feature in PHASE2_OPPORTUNITY_SPECS)
    omitted = tuple(feature for feature in selected if feature not in PHASE2_OPPORTUNITY_SPECS)
    if not measured:
        omitted_display = ", ".join(omitted)
        raise ValueError(f"none of the requested features has a Phase 2 opportunity spec: {omitted_display}")
    return selected, measured, omitted


def _unique_measurement_texts(rows: list[dict[str, Any]]) -> int:
    per_input: dict[str, int] = defaultdict(int)
    for row in rows:
        per_input[row["input"]] = max(per_input[row["input"]], int(row["measurement_rows"]))
    return sum(per_input.values())


def _measure_input(
    input_path: Path,
    *,
    text_field: str | None,
    sample_size: int,
    max_scan: int | None,
    features: tuple[str, ...],
    max_token_start_opportunities: int | None,
) -> list[dict[str, Any]]:
    supports = {feature: FeatureSupport(feature=feature) for feature in features}
    for item in tqdm(
        _iter_input_texts(
            input_path,
            text_field=text_field,
            sample_size=sample_size,
            max_scan=max_scan,
        ),
        desc=f"phase2-denominators:{input_path.name}",
        unit="text",
    ):
        opportunities = iter_feature_opportunities(
            item.text,
            features=features,
            max_token_start_opportunities=max_token_start_opportunities,
        )
        by_feature: dict[str, list[Any]] = defaultdict(list)
        for opportunity in opportunities:
            by_feature[opportunity.feature].append(opportunity)
        for feature, support in supports.items():
            support.measurement_rows += 1
            feature_opportunities = by_feature.get(feature, [])
            reference_count = sum(1 for item in feature_opportunities if item.reference_initiates)
            support.opportunities += len(feature_opportunities)
            support.reference_initiations += reference_count
            if feature_opportunities:
                support.documents_with_opportunity += 1
            if reference_count:
                support.documents_with_reference += 1
            for opportunity in feature_opportunities:
                if opportunity.reference_initiates and opportunity.matched_subtype:
                    support.matched_subtypes[opportunity.matched_subtype] += 1
    return [support.row(input_name=str(input_path)) for support in supports.values()]


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
            yield from _measurement_texts(row, text_field=text_field, row_index=line_number - 1)


def _measurement_texts(
    row: dict[str, Any],
    *,
    text_field: str | None,
    row_index: int,
) -> list[MeasurementText]:
    record_id = str(row.get("phase2_prompt_id") or row.get("id") or row_index)
    if isinstance(row.get("chosen"), str) and isinstance(row.get("rejected"), str):
        return [
            MeasurementText(record_id=record_id, role="chosen", text=row["chosen"]),
            MeasurementText(record_id=record_id, role="rejected", text=row["rejected"]),
        ]
    fields = [text_field] if text_field else ["text", "response", "completion", "content"]
    for field_name in fields:
        value = row.get(field_name)
        if isinstance(value, str):
            return [MeasurementText(record_id=record_id, role=field_name, text=value)]
    return []


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    args = build_parser().parse_args()
    summary = run_measure_phase2_denominators(args)
    print(
        "Wrote denominator support for "
        f"{summary['features']} features to {summary['output']}"
    )


if __name__ == "__main__":
    main()
