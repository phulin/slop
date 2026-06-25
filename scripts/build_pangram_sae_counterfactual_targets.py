from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _compile_replacements(values: list[str]) -> list[tuple[re.Pattern[str], str]]:
    replacements: list[tuple[re.Pattern[str], str]] = []
    for value in values:
        if "=>" not in value:
            raise ValueError(f"invalid replacement {value!r}; expected REGEX=>TEXT")
        pattern, replacement = value.split("=>", 1)
        replacements.append((re.compile(pattern, flags=re.IGNORECASE), replacement))
    return replacements


def _apply_replacements(text: str, replacements: list[tuple[re.Pattern[str], str]]) -> tuple[str, int]:
    total = 0
    out = text
    for pattern, replacement in replacements:
        out, count = pattern.subn(replacement, out)
        total += count
    return out, total


def _shorten(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "chain",
        "doc_id",
        "base_doc_id",
        "turn_id",
        "model",
        "variant",
        "edit_count",
        "text",
        "context",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Pangram SAE Counterfactual Targets",
        "",
        "Each base document has an `original` row and an `edited` row. Interventions use the explicit `text` column, not the token parquet text lookup.",
        "",
    ]
    for row in rows:
        if row["variant"] != "edited":
            continue
        base = next(
            candidate
            for candidate in rows
            if candidate["base_doc_id"] == row["base_doc_id"] and candidate["variant"] == "original"
        )
        lines.extend(
            [
                f"## `{str(row['base_doc_id'])[:10]}` `{row['model']}` edits `{row['edit_count']}`",
                "",
                "Original:",
                "",
                str(base["context"]),
                "",
                "Edited:",
                "",
                str(row["context"]),
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build explicit-text counterfactual targets for Pangram SAE interventions.")
    parser.add_argument("--target-csv", type=Path, required=True, help="Rows containing doc_id values to edit.")
    parser.add_argument("--doc-tokens-parquet", type=Path, default=Path("apps/latent-explorer/public/data/layer19/pangram_llama_sae_doc_tokens.parquet"))
    parser.add_argument("--chain", required=True)
    parser.add_argument("--replacement", action="append", required=True, metavar="REGEX=>TEXT")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--context-chars", type=int, default=420)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replacements = _compile_replacements(args.replacement)
    targets = pd.read_csv(args.target_csv)
    doc_ids = list(dict.fromkeys(str(value) for value in targets["doc_id"].tolist()))
    if args.limit is not None:
        doc_ids = doc_ids[: int(args.limit)]
    docs = pd.read_parquet(
        args.doc_tokens_parquet,
        columns=["doc_id", "turn_id", "model", "text"],
    )
    docs = docs[docs["doc_id"].astype(str).isin(set(doc_ids))]
    docs_by_id = {str(row.doc_id): row for row in docs.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        doc = docs_by_id.get(doc_id)
        if doc is None:
            continue
        text = str(doc.text)
        edited, edit_count = _apply_replacements(text, replacements)
        if edit_count <= 0 or edited == text:
            continue
        base = {
            "chain": args.chain,
            "base_doc_id": doc_id,
            "turn_id": doc.turn_id,
            "model": doc.model,
        }
        rows.append(
            {
                **base,
                "doc_id": f"{doc_id}::original",
                "variant": "original",
                "edit_count": 0,
                "text": text,
                "context": _shorten(text, int(args.context_chars)),
            }
        )
        rows.append(
            {
                **base,
                "doc_id": f"{doc_id}::edited",
                "variant": "edited",
                "edit_count": edit_count,
                "text": edited,
                "context": _shorten(edited, int(args.context_chars)),
            }
        )
    _write_csv(args.output_csv, rows)
    _write_md(args.output_md, rows)
    print(
        {
            "base_docs": len(rows) // 2,
            "rows": len(rows),
            "output_csv": str(args.output_csv),
            "output_md": str(args.output_md),
        }
    )


if __name__ == "__main__":
    main()
