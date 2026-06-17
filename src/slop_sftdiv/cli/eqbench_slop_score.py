from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from slop_sftdiv.features.eqbench_slop import (
    EqbenchSlopScorecard,
    eqbench_prompt_manifest,
    score_eqbench_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score outputs with the EQ-Bench Slop Score.")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help=(
            "JSONL/JSON/text file with generated outputs. Repeat to concatenate sources. "
            "For JSONL/JSON, use --text-field to select a field."
        ),
    )
    parser.add_argument(
        "--text-field",
        action="append",
        default=[],
        help=(
            "Output field to read from JSON records. Defaults to common generation fields: "
            "output, generation, text, response, completion."
        ),
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=300,
        help="Minimum output length, matching EQ-Bench's default >300-char valid-sample filter.",
    )
    parser.add_argument(
        "--contrast-mode",
        choices=["auto", "exact", "portable"],
        default="auto",
        help=(
            "Contrast detector backend. auto uses the wink-pos-tagger exact Stage-1+2 "
            "backend when available and falls back to portable Stage-1 regexes."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON scorecard path.")
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional one-row CSV scorecard path for spreadsheets/joins.",
    )
    parser.add_argument(
        "--write-prompts",
        type=Path,
        default=None,
        help="Optional path to write the vendored 300-prompt EQ-Bench manifest.",
    )
    return parser


def run_eqbench_slop_score(args: argparse.Namespace) -> dict[str, Any]:
    if args.write_prompts is not None:
        args.write_prompts.parent.mkdir(parents=True, exist_ok=True)
        args.write_prompts.write_text(
            json.dumps(eqbench_prompt_manifest(), indent=2) + "\n",
            encoding="utf-8",
        )

    outputs: list[str] = []
    for raw_input in args.input:
        outputs.extend(_load_outputs(Path(raw_input), text_fields=args.text_field))

    scorecard = score_eqbench_outputs(
        outputs,
        min_chars=args.min_chars,
        contrast_mode=args.contrast_mode,
    )
    payload = _scorecard_payload(scorecard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv_output is not None:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload["metrics"].keys()))
            writer.writeheader()
            writer.writerow(payload["metrics"])
    return payload


def _load_outputs(path: Path, *, text_fields: list[str]) -> list[str]:
    if path.suffix == ".jsonl":
        return _load_jsonl_outputs(path, text_fields=text_fields)
    if path.suffix == ".json":
        return _load_json_outputs(path, text_fields=text_fields)
    return [path.read_text(encoding="utf-8")]


def _load_jsonl_outputs(path: Path, *, text_fields: list[str]) -> list[str]:
    outputs = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            text = _record_text(record, text_fields=text_fields)
            if text is not None:
                outputs.append(text)
    return outputs


def _load_json_outputs(path: Path, *, text_fields: list[str]) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [
            text
            for record in payload
            if (text := _record_text(record, text_fields=text_fields)) is not None
        ]
    if isinstance(payload, dict) and "samples" in payload:
        return [
            text
            for record in payload["samples"]
            if (text := _record_text(record, text_fields=text_fields)) is not None
        ]
    if isinstance(payload, dict):
        outputs = []
        for value in payload.values():
            if isinstance(value, dict) and "samples" in value:
                outputs.extend(
                    text
                    for record in value["samples"]
                    if (text := _record_text(record, text_fields=text_fields)) is not None
                )
        if outputs:
            return outputs
        text = _record_text(payload, text_fields=text_fields)
        return [text] if text is not None else []
    if isinstance(payload, str):
        return [payload]
    raise ValueError(f"unsupported JSON payload shape in {path}")


def _record_text(record: Any, *, text_fields: list[str]) -> str | None:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return None
    for field_name in [*text_fields, "output", "generation", "text", "response", "completion"]:
        value = record.get(field_name)
        if isinstance(value, str):
            return value
    return None


def _scorecard_payload(scorecard: EqbenchSlopScorecard) -> dict[str, Any]:
    metrics = scorecard.metrics
    return {
        "metrics": {
            "eqbench_slop_score": metrics.slop_score,
            "slop_list_matches_per_1k_words": metrics.word_matches_per_1k_words,
            "slop_trigram_matches_per_1k_words": metrics.trigram_matches_per_1k_words,
            "not_x_but_y_per_1k_chars": metrics.not_x_but_y_per_1k_chars,
            "word_hit_count": metrics.word_hit_count,
            "trigram_hit_count": metrics.trigram_hit_count,
            "contrast_hit_count": metrics.contrast_hit_count,
            "token_count": metrics.token_count,
            "sample_count": scorecard.sample_count,
            "valid_sample_count": scorecard.valid_sample_count,
            "total_chars": scorecard.total_chars,
            "min_chars": scorecard.min_chars,
            "contrast_mode": scorecard.contrast_mode,
        },
        "slop_word_hits": scorecard.slop_word_hits,
        "slop_trigram_hits": scorecard.slop_trigram_hits,
        "contrast_matches": [
            {"pattern_name": pattern_name, "match_text": match_text}
            for pattern_name, match_text in scorecard.contrast_matches
        ],
    }


def main() -> None:
    args = build_parser().parse_args()
    payload = run_eqbench_slop_score(args)
    print(f"Wrote EQ-Bench scorecard to {args.output}")
    print(f"EQ-Bench Slop Score: {payload['metrics']['eqbench_slop_score']:.3f}")


if __name__ == "__main__":
    main()
