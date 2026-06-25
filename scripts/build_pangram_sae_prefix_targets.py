from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _shorten(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _prefix_words(text: str, words: int) -> str:
    parts = re.findall(r"\S+", text)
    if len(parts) <= words:
        return text
    prefix = " ".join(parts[:words])
    if prefix and prefix[-1] not in ".?!":
        prefix += "."
    return prefix


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "chain",
        "doc_id",
        "base_doc_id",
        "turn_id",
        "model",
        "variant",
        "prefix_words",
        "text",
        "context",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Pangram SAE Prefix Targets",
        "",
        "Each base document has an `original` row and one or more prefix rows. Interventions use the explicit `text` column, not the token parquet text lookup.",
        "",
    ]
    originals = {
        row["base_doc_id"]: row
        for row in rows
        if row["variant"] == "original"
    }
    for row in rows:
        if row["variant"] == "original":
            continue
        base = originals[row["base_doc_id"]]
        lines.extend(
            [
                f"## `{str(row['base_doc_id'])[:10]}` `{row['model']}` `{row['variant']}`",
                "",
                "Original:",
                "",
                str(base["context"]),
                "",
                "Prefix:",
                "",
                str(row["context"]),
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build explicit prefix-truncated targets for Pangram SAE interventions.")
    parser.add_argument("--target-csv", type=Path, required=True, help="Rows containing doc_id values to truncate.")
    parser.add_argument("--doc-tokens-parquet", type=Path, default=Path("apps/latent-explorer/public/data/layer19/pangram_llama_sae_doc_tokens.parquet"))
    parser.add_argument("--chain", required=True)
    parser.add_argument("--prefix-words", action="append", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--context-chars", type=int, default=420)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
                "prefix_words": "",
                "text": text,
                "context": _shorten(text, int(args.context_chars)),
            }
        )
        for prefix_words in args.prefix_words:
            prefix = _prefix_words(text, int(prefix_words))
            if prefix == text:
                continue
            rows.append(
                {
                    **base,
                    "doc_id": f"{doc_id}::prefix_{int(prefix_words)}w",
                    "variant": f"prefix_{int(prefix_words)}w",
                    "prefix_words": int(prefix_words),
                    "text": prefix,
                    "context": _shorten(prefix, int(args.context_chars)),
                }
            )
    _write_csv(args.output_csv, rows)
    _write_md(args.output_md, rows)
    print(
        {
            "base_docs": sum(1 for row in rows if row["variant"] == "original"),
            "rows": len(rows),
            "output_csv": str(args.output_csv),
            "output_md": str(args.output_md),
        }
    )


if __name__ == "__main__":
    main()
