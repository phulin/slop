from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MARKERS = [
    r"<\s*/?\s*think\s*>",
    r"<\s*/?\s*reasoning\s*>",
    r"<\s*analysis\s*>",
    r"<\s*/\s*analysis\s*>",
    r"```thinking\b",
    r"```reasoning\b",
]

OUTPUT_COLUMNS = [
    "generation_cache",
    "records",
    "records_with_generation",
    "records_without_generation",
    "records_with_marker",
    "marker_hits",
    "marker_records_per_1k",
    "marker_hits_per_1k_records",
    "passed",
    "marker_counts_json",
    "example_record_ids_json",
]


@dataclass(frozen=True)
class AuditResult:
    generation_cache: str
    records: int
    records_with_generation: int
    records_without_generation: int
    records_with_marker: int
    marker_hits: int
    marker_records_per_1k: float
    marker_hits_per_1k_records: float
    passed: bool
    marker_counts_json: str
    example_record_ids_json: str

    def as_row(self) -> dict[str, Any]:
        return {
            "generation_cache": self.generation_cache,
            "records": self.records,
            "records_with_generation": self.records_with_generation,
            "records_without_generation": self.records_without_generation,
            "records_with_marker": self.records_with_marker,
            "marker_hits": self.marker_hits,
            "marker_records_per_1k": self.marker_records_per_1k,
            "marker_hits_per_1k_records": self.marker_hits_per_1k_records,
            "passed": self.passed,
            "marker_counts_json": self.marker_counts_json,
            "example_record_ids_json": self.example_record_ids_json,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit generated text caches for explicit thinking/reasoning markup."
    )
    parser.add_argument(
        "--generation-cache",
        action="append",
        type=Path,
        required=True,
        help="Generation JSONL cache to audit. May be repeated.",
    )
    parser.add_argument(
        "--marker-regex",
        action="append",
        default=[],
        help="Additional case-insensitive regex marker to flag.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum flagged record identifiers to keep in the output.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser


def _compile_markers(extra_markers: list[str]) -> list[re.Pattern[str]]:
    patterns = DEFAULT_MARKERS + extra_markers
    return [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]


def _record_id(row: dict[str, Any], line_number: int) -> str:
    for key in ("record_id", "id", "row_index"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return f"line:{line_number}"


def audit_cache(path: Path, *, markers: list[re.Pattern[str]], max_examples: int) -> AuditResult:
    marker_counts: Counter[str] = Counter()
    example_record_ids: list[str] = []
    records = 0
    records_with_generation = 0
    records_with_marker = 0
    marker_hits = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            records += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
            generation = row.get("generation")
            if generation in (None, ""):
                continue
            records_with_generation += 1
            text = str(generation)
            row_hits = 0
            for marker in markers:
                matches = marker.findall(text)
                if matches:
                    marker_counts[marker.pattern] += len(matches)
                    row_hits += len(matches)
            if row_hits:
                records_with_marker += 1
                marker_hits += row_hits
                if len(example_record_ids) < max_examples:
                    example_record_ids.append(_record_id(row, line_number))
    marker_records_per_1k = 1000.0 * records_with_marker / records if records else 0.0
    marker_hits_per_1k_records = 1000.0 * marker_hits / records if records else 0.0
    return AuditResult(
        generation_cache=str(path),
        records=records,
        records_with_generation=records_with_generation,
        records_without_generation=records - records_with_generation,
        records_with_marker=records_with_marker,
        marker_hits=marker_hits,
        marker_records_per_1k=marker_records_per_1k,
        marker_hits_per_1k_records=marker_hits_per_1k_records,
        passed=records_with_marker == 0,
        marker_counts_json=json.dumps(dict(sorted(marker_counts.items())), sort_keys=True),
        example_record_ids_json=json.dumps(example_record_ids),
    )


def _write_csv(path: Path, rows: list[AuditResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_row())


def _write_markdown(path: Path, rows: list[AuditResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# No-Think Generation Markup Audit",
        "",
        "| Generation cache | Records | Non-empty generations | Empty generations | Records with marker | Marker hits | Passed |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| `{path}` | {records} | {with_generation} | {without_generation} | {records_with_marker} | {marker_hits} | {passed} |".format(
                path=row.generation_cache,
                records=row.records,
                with_generation=row.records_with_generation,
                without_generation=row.records_without_generation,
                records_with_marker=row.records_with_marker,
                marker_hits=row.marker_hits,
                passed="yes" if row.passed else "no",
            )
        )
    lines.extend(
        [
            "",
            "This audit flags explicit thinking/reasoning markup only. It does not treat ordinary uses of words such as `think` or `reasoning` as mode failures.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[AuditResult]:
    if args.max_examples < 0:
        raise ValueError("--max-examples must be non-negative")
    markers = _compile_markers(args.marker_regex)
    rows = [
        audit_cache(path, markers=markers, max_examples=args.max_examples)
        for path in args.generation_cache
    ]
    _write_csv(args.output, rows)
    _write_markdown(args.summary_output, rows)
    failures = sum(1 for row in rows if not row.passed)
    print(f"Wrote {len(rows)} no-think generation audits to {args.output}; failures={failures}")
    return rows


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
