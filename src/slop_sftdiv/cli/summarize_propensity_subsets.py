from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from slop_sftdiv.cli.teacher_forced_propensity import (
    FeatureAccumulator,
    SUMMARY_COLUMNS,
    _summary_rows,
)
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


OUTPUT_COLUMNS = [
    "stage",
    "reference_subset",
    "reference_subset_field",
    *SUMMARY_COLUMNS,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize scored teacher-forced propensity rows by prompt metadata subset."
    )
    parser.add_argument(
        "--opportunities",
        action="append",
        required=True,
        metavar="STAGE=PATH",
        help="Stage-labeled opportunity CSV from slop-teacher-forced-propensity.",
    )
    parser.add_argument(
        "--metadata-jsonl",
        type=Path,
        required=True,
        help="Annotated Phase 2 prompt JSONL containing the subset field.",
    )
    parser.add_argument("--subset-field", default="reference_subset")
    parser.add_argument("--missing-subset", default="missing")
    parser.add_argument("--normalization-feature", default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-propensity")
    parser.add_argument("--wandb-job-type", default="reference-subset-summary")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _parse_labeled_path(raw_value: str) -> tuple[str, Path]:
    if "=" not in raw_value:
        raise ValueError(f"invalid --opportunities {raw_value!r}; expected STAGE=PATH")
    label, path = raw_value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"invalid --opportunities {raw_value!r}; expected STAGE=PATH")
    return label, Path(path)


def _record_ids(row: dict[str, Any]) -> list[str]:
    ids = [
        row.get("phase2_prompt_id"),
        row.get("prompt_id"),
        row.get("doc_id"),
        row.get("source_record_id"),
        row.get("id"),
    ]
    return [str(item) for item in ids if item not in (None, "")]


def _load_subset_by_record(metadata_jsonl: Path, *, subset_field: str) -> dict[tuple[str, str], str]:
    subset_by_key: dict[tuple[str, str], str] = {}
    with metadata_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {metadata_jsonl}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {metadata_jsonl}:{line_number} is not an object")
            subset = str(row.get(subset_field, ""))
            role = str(row.get("text_role") or "text")
            for record_id in _record_ids(row):
                subset_by_key[(record_id, role)] = subset
    return subset_by_key


def _load_opportunity_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 2 Reference-Subset Propensity Summary",
        "",
        "| Stage | Subset | Feature | Opportunities | Refs | AF | Norm AF |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {stage} | {subset} | {feature} | {opps} | {refs} | {af:.3f} | {naf:.3f} |".format(
                stage=row["stage"],
                subset=row["reference_subset"],
                feature=row["feature"],
                opps=row["opportunities"],
                refs=row["reference_initiations"],
                af=float(row["amplification_factor"]),
                naf=float(row["normalized_amplification_factor"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_summarize_propensity_subsets(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    opportunity_inputs = [_parse_labeled_path(item) for item in args.opportunities]
    started_at = time.perf_counter()
    subset_by_key = _load_subset_by_record(args.metadata_jsonl, subset_field=args.subset_field)
    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage2", "phase2", "teacher-forced", "reference-subsets", *args.wandb_tag],
        config={
            "opportunities": [(stage, str(path)) for stage, path in opportunity_inputs],
            "metadata_jsonl": str(args.metadata_jsonl),
            "subset_field": args.subset_field,
            "missing_subset": args.missing_subset,
            "normalization_feature": args.normalization_feature,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "output": str(args.output),
            "summary_output": str(args.summary_output) if args.summary_output is not None else None,
        },
    )
    try:
        output_rows: list[dict[str, Any]] = []
        total_rows = 0
        missing_rows = 0
        for stage, path in opportunity_inputs:
            accumulators: dict[str, dict[str, FeatureAccumulator]] = defaultdict(
                lambda: defaultdict(FeatureAccumulator)
            )
            rows_by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in _load_opportunity_rows(path):
                total_rows += 1
                record_id = str(row["record_id"])
                role = str(row["role"])
                subset = subset_by_key.get((record_id, role))
                if subset is None:
                    subset = args.missing_subset
                    missing_rows += 1
                feature = str(row["feature"])
                reference_initiates = int(row["reference_initiates"])
                prob_mass = float(row["prob_mass"])
                accumulator = accumulators[subset][feature]
                accumulator.opportunities += 1
                accumulator.reference_initiations += reference_initiates
                accumulator.prob_mass_sum += prob_mass
                rows_by_subset[subset].append(
                    {
                        "source": row["source"],
                        "record_id": record_id,
                        "role": role,
                        "feature": feature,
                        "reference_initiates": reference_initiates,
                        "prob_mass": prob_mass,
                    }
                )
            for subset in sorted(accumulators):
                for summary_row in _summary_rows(
                    accumulators[subset],
                    rows=rows_by_subset[subset],
                    bootstrap_samples=args.bootstrap_samples,
                    bootstrap_seed=args.bootstrap_seed,
                    normalization_feature=args.normalization_feature,
                ):
                    output_rows.append(
                        {
                            "stage": stage,
                            "reference_subset": subset,
                            "reference_subset_field": args.subset_field,
                            **summary_row,
                        }
                    )
        _write_csv(args.output, output_rows, OUTPUT_COLUMNS)
        if args.summary_output is not None:
            _write_markdown(args.summary_output, output_rows)
        wall_seconds = time.perf_counter() - started_at
        run.log(
            {
                "propensity_reference_subsets/input_rows": total_rows,
                "propensity_reference_subsets/missing_rows": missing_rows,
                "propensity_reference_subsets/output_rows": len(output_rows),
                "propensity_reference_subsets/wall_seconds": wall_seconds,
                "propensity_reference_subsets/rows_per_sec": total_rows / wall_seconds
                if wall_seconds > 0
                else 0.0,
                "propensity_reference_subsets/peak_rss_mb": _peak_rss_mb(),
            }
        )
        log_summary_table(run, "propensity_reference_subset_summary", output_rows)
        return output_rows
    finally:
        run.finish()


def main() -> None:
    args = build_parser().parse_args()
    rows = run_summarize_propensity_subsets(args)
    print(f"Wrote {len(rows)} reference-subset summaries to {args.output}")


if __name__ == "__main__":
    main()
