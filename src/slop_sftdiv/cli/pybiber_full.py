from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, cast

from slop_sftdiv.features.pybiber_full import PYBIBER_FEATURES, run_pybiber_full


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract full pybiber features from text records.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="JSONL/JSON/text input. Repeat to concatenate sources.",
    )
    parser.add_argument(
        "--text-field",
        action="append",
        default=[],
        help="Text field for JSON records. Defaults to text, generation, output, response, completion.",
    )
    parser.add_argument(
        "--id-field",
        action="append",
        default=[],
        help="Document id field for JSON records. Defaults to doc_id, record_id, prompt_id, id.",
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--n-process", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True, help="Wide CSV output path.")
    parser.add_argument(
        "--long-output",
        type=Path,
        default=None,
        help="Optional long CSV with one row per document and feature.",
    )
    return parser


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, str]] = []
    for input_index, raw_input in enumerate(args.input):
        records.extend(
            _load_records(
                Path(raw_input),
                text_fields=args.text_field,
                id_fields=args.id_field,
                input_index=input_index,
            )
        )
    records = _ensure_unique_doc_ids(records)
    if args.sample_size is not None:
        records = records[: args.sample_size]
    rows = cast(
        list[dict[str, Any]],
        run_pybiber_full(
            records,
            model=args.model,
            n_process=args.n_process,
            batch_size=args.batch_size,
        ),
    )
    _write_wide_csv(args.output, rows)
    if args.long_output is not None:
        _write_long_csv(args.long_output, rows)
    return rows


def _load_records(
    path: Path,
    *,
    text_fields: list[str],
    id_fields: list[str],
    input_index: int,
) -> list[dict[str, str]]:
    if path.suffix == ".jsonl":
        records = []
        with path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if not line.strip():
                    continue
                payload = json.loads(line)
                record = _record_from_payload(
                    payload,
                    text_fields=text_fields,
                    id_fields=id_fields,
                    fallback_doc_id=f"input{input_index}_{row_index}",
                )
                if record is not None:
                    records.append(record)
        return records
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads = payload if isinstance(payload, list) else [payload]
        return [
            record
            for row_index, item in enumerate(payloads)
            if (
                record := _record_from_payload(
                    item,
                    text_fields=text_fields,
                    id_fields=id_fields,
                    fallback_doc_id=f"input{input_index}_{row_index}",
                )
            )
            is not None
        ]
    return [{"doc_id": f"input{input_index}_0", "text": path.read_text(encoding="utf-8")}]


def _record_from_payload(
    payload: Any,
    *,
    text_fields: list[str],
    id_fields: list[str],
    fallback_doc_id: str,
) -> dict[str, str] | None:
    if isinstance(payload, str):
        return {"doc_id": fallback_doc_id, "text": payload}
    if not isinstance(payload, dict):
        return None
    text = _first_string(
        payload,
        [*text_fields, "text", "generation", "output", "response", "completion"],
    )
    if text is None:
        return None
    doc_id = _first_string(payload, [*id_fields, "doc_id", "record_id", "prompt_id", "id"])
    return {"doc_id": doc_id or fallback_doc_id, "text": text}


def _ensure_unique_doc_ids(records: list[dict[str, str]]) -> list[dict[str, str]]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["doc_id"]] = counts.get(record["doc_id"], 0) + 1
    if all(count == 1 for count in counts.values()):
        return records
    unique_records: list[dict[str, str]] = []
    for row_index, record in enumerate(records):
        if counts[record["doc_id"]] == 1:
            unique_records.append(record)
            continue
        unique_records.append({**record, "doc_id": f"{record['doc_id']}#row{row_index}"})
    return unique_records


def _first_string(payload: dict[str, Any], fields: list[str]) -> str | None:
    for field in fields:
        value = payload.get(field)
        if isinstance(value, str):
            return value
    return None


def _write_wide_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["doc_id", *PYBIBER_FEATURES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_long_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["doc_id", "feature", "value"])
        writer.writeheader()
        for row in rows:
            doc_id = row.get("doc_id")
            for feature in PYBIBER_FEATURES:
                writer.writerow(
                    {
                        "doc_id": doc_id,
                        "feature": feature,
                        "value": row.get(feature, 0.0),
                    }
                )


def main() -> None:
    args = build_parser().parse_args()
    rows = run(args)
    print(f"Wrote {len(rows)} pybiber document rows to {args.output}")


if __name__ == "__main__":
    main()
