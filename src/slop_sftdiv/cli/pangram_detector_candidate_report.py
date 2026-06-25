from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from slop_sftdiv.cli.pangram_detector_ar_search import sequence_quality_penalty


CANDIDATE_FILENAMES = {
    "pangram_detector_ar_search_final_beams.csv",
    "pangram_detector_sample_search_samples.csv",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize AR-beam and sample-search candidate CSVs in one ranked table."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--sort-by", choices=["objective", "score", "quality"], default="objective")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-continuation-chars", type=int, default=220)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def _candidate_paths(paths: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in paths:
        if path.is_file() and path.name in CANDIDATE_FILENAMES:
            found.add(path)
        elif path.is_dir():
            for filename in CANDIDATE_FILENAMES:
                if (path / filename).is_file():
                    found.add(path / filename)
                found.update(path.glob(f"**/{filename}"))
    return sorted(found)


def _float_from_row(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key)
    if value in (None, ""):
        return default
    return float(value)


def _rows_from_candidate_csv(path: Path, *, max_continuation_chars: int) -> list[dict[str, Any]]:
    kind = (
        "sample_search"
        if path.name == "pangram_detector_sample_search_samples.csv"
        else "ar_search"
    )
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            continuation = row.get("continuation") or row.get("generated_continuation") or ""
            quality_penalty = sequence_quality_penalty(continuation)
            display_continuation = continuation.replace("\n", "\\n")
            if max_continuation_chars > 0 and len(display_continuation) > max_continuation_chars:
                display_continuation = display_continuation[: max_continuation_chars - 3] + "..."
            rows.append(
                {
                    "candidate_path": str(path),
                    "run_dir": str(path.parent),
                    "candidate_kind": kind,
                    "candidate_rank_in_file": index,
                    "source_index": row.get("sample_index") or row.get("beam_rank") or index,
                    "editlens_score": _float_from_row(row, "editlens_score"),
                    "objective": _float_from_row(row, "objective"),
                    "mean_lm_log_probability": _float_from_row(row, "mean_lm_log_probability"),
                    "stored_sequence_quality_penalty": _float_from_row(
                        row, "sequence_quality_penalty"
                    ),
                    "computed_sequence_quality_penalty": quality_penalty,
                    "continuation": display_continuation,
                }
            )
    return rows


def summarize_candidates(
    paths: list[Path],
    *,
    sort_by: str,
    top_k: int,
    max_continuation_chars: int,
) -> list[dict[str, Any]]:
    rows = [
        row
        for path in _candidate_paths(paths)
        for row in _rows_from_candidate_csv(path, max_continuation_chars=max_continuation_chars)
    ]
    if sort_by == "score":
        rows.sort(key=lambda row: row["editlens_score"], reverse=True)
    elif sort_by == "quality":
        rows.sort(
            key=lambda row: (
                row["computed_sequence_quality_penalty"],
                -row["editlens_score"],
            )
        )
    else:
        rows.sort(key=lambda row: row["objective"], reverse=True)
    return rows[:top_k] if top_k > 0 else rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("run_dir,kind,score,objective,quality_penalty,mean_lm,continuation")
    for row in rows:
        print(
            f"{row['run_dir']},{row['candidate_kind']},{row['editlens_score']:.4f},"
            f"{row['objective']:.4f},{row['computed_sequence_quality_penalty']:.4f},"
            f"{row['mean_lm_log_probability']:.4f},"
            f"{json.dumps(row['continuation'], ensure_ascii=False)}"
        )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = summarize_candidates(
        args.paths,
        sort_by=args.sort_by,
        top_k=args.top_k,
        max_continuation_chars=args.max_continuation_chars,
    )
    if args.output_csv is not None:
        _write_csv(args.output_csv, rows)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(rows, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    _print_table(rows)


if __name__ == "__main__":
    main()
