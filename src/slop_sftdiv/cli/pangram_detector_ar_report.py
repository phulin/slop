from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from slop_sftdiv.cli.pangram_detector_ar_search import sequence_quality_penalty

SUMMARY_FILENAMES = {
    "pangram_detector_ar_search_summary.json",
    "pangram_detector_sample_search_summary.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize Pangram/EditLens autoregressive-search runs."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="Run directories or parent directories containing pangram_detector_ar_search_summary.json files.",
    )
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-continuation-chars", type=int, default=240)
    return parser


def _summary_paths(paths: list[Path]) -> list[Path]:
    found: set[Path] = set()
    for path in paths:
        if path.is_file() and path.name in SUMMARY_FILENAMES:
            found.add(path)
        elif path.is_dir():
            for filename in SUMMARY_FILENAMES:
                if (path / filename).is_file():
                    found.add(path / filename)
                found.update(path.glob(f"**/{filename}"))
    return sorted(found)


def _row(path: Path, max_continuation_chars: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    initial_score = float(payload["initial_editlens_score"])
    final_score = float(payload["final_editlens_score"])
    continuation = str(payload["optimized_continuation"]).replace("\n", "\\n")
    current_quality_penalty = sequence_quality_penalty(str(payload["optimized_continuation"]))
    if max_continuation_chars > 0 and len(continuation) > max_continuation_chars:
        continuation = continuation[: max_continuation_chars - 3] + "..."
    bucket_probabilities = payload.get("final_bucket_probabilities") or []
    return {
        "summary_path": str(path),
        "run_dir": str(path.parent),
        "experiment": payload.get("experiment"),
        "initial_score": initial_score,
        "final_score": final_score,
        "delta_score": final_score - initial_score,
        "bucket3_probability": float(bucket_probabilities[-1]) if bucket_probabilities else None,
        "mean_lm_log_probability": payload.get("final_mean_lm_log_probability"),
        "lm_logprob_coeff": payload.get("lm_logprob_coeff"),
        "repetition_penalty": payload.get("repetition_penalty"),
        "sequence_quality_penalty_coeff": payload.get("sequence_quality_penalty_coeff"),
        "final_sequence_quality_penalty": payload.get("final_sequence_quality_penalty"),
        "computed_sequence_quality_penalty": current_quality_penalty,
        "beam_size": payload.get("beam_size"),
        "candidate_top_k": payload.get("candidate_top_k"),
        "reject_allcaps_tokens": payload.get("reject_allcaps_tokens"),
        "reject_symbol_tokens": payload.get("reject_symbol_tokens"),
        "reject_camelcase_tokens": payload.get("reject_camelcase_tokens"),
        "reject_code_substrings": payload.get("reject_code_substrings"),
        "reject_bare_subword_tokens": payload.get("reject_bare_subword_tokens"),
        "reject_punctuation_run_tokens": payload.get("reject_punctuation_run_tokens"),
        "prompt": payload.get("prompt"),
        "optimized_continuation": continuation,
    }


def summarize(paths: list[Path], *, max_continuation_chars: int) -> list[dict[str, Any]]:
    return [
        _row(path, max_continuation_chars=max_continuation_chars)
        for path in _summary_paths(paths)
    ]


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
    print("run_dir,experiment,initial,final,delta,bucket3,mean_lm,quality_penalty,continuation")
    for row in rows:
        print(
            f"{row['run_dir']},{row['experiment']},{row['initial_score']:.4f},{row['final_score']:.4f},"
            f"{row['delta_score']:.4f},{row['bucket3_probability']:.4f},"
            f"{float(row['mean_lm_log_probability']):.4f},"
            f"{float(row['computed_sequence_quality_penalty']):.4f},"
            f"{json.dumps(row['optimized_continuation'], ensure_ascii=False)}"
        )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rows = summarize(args.run_dirs, max_continuation_chars=args.max_continuation_chars)
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
