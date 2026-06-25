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


def _load_excluded(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        df = pd.read_csv(path)
        if "doc_id" in df.columns:
            excluded.update(str(value) for value in df["doc_id"].dropna().tolist())
    return excluded


def _pattern_count(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(text))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "chain",
        "node",
        "layer",
        "latent_id",
        "turn_id",
        "doc_id",
        "model",
        "peak_activation",
        "token_index",
        "prompt_contrast_logit_drop",
        "prompt_abs_contrast_logit_change",
        "prompt_group_models",
        "pattern_count",
        "context",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "These are held-out pattern-selected target documents. They are not selected from the latent's top-activation prompt-overlap list.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## `{str(row['doc_id'])[:10]}` `{row['model']}`",
                "",
                f"- turn: `{row['turn_id']}`",
                f"- pattern count: `{row['pattern_count']}`",
                "",
                str(row["context"]),
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select held-out pattern-matched Pangram SAE target documents.")
    parser.add_argument("--doc-tokens-parquet", type=Path, default=Path("apps/latent-explorer/public/data/layer19/pangram_llama_sae_doc_tokens.parquet"))
    parser.add_argument("--pattern", required=True, help="Python regex applied case-insensitively to document text.")
    parser.add_argument("--chain", required=True, help="Label for downstream intervention target, e.g. 'L19:1 -> L24:2'.")
    parser.add_argument("--models", nargs="*", default=[], help="Optional model allowlist.")
    parser.add_argument("--exclude-target-csv", action="append", type=Path, default=[])
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--min-word-count", type=int, default=80)
    parser.add_argument("--max-word-count", type=int, default=520)
    parser.add_argument("--context-chars", type=int, default=420)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--title", default="Held-Out Pattern Targets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pattern = re.compile(args.pattern, flags=re.IGNORECASE)
    excluded = _load_excluded(args.exclude_target_csv)
    df = pd.read_parquet(
        args.doc_tokens_parquet,
        columns=["doc_id", "turn_id", "model", "provider", "role", "category", "prompt", "text", "word_count"],
    )
    df = df[(df["word_count"] >= int(args.min_word_count)) & (df["word_count"] <= int(args.max_word_count))]
    if args.models:
        allowed = set(args.models)
        df = df[df["model"].astype(str).isin(allowed)]
    if excluded:
        df = df[~df["doc_id"].astype(str).isin(excluded)]
    rows: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        text = str(row.text)
        count = _pattern_count(text, pattern)
        if count <= 0:
            continue
        rows.append(
            {
                "chain": args.chain,
                "node": "",
                "layer": "",
                "latent_id": "",
                "turn_id": row.turn_id,
                "doc_id": row.doc_id,
                "model": row.model,
                "peak_activation": "",
                "token_index": "",
                "prompt_contrast_logit_drop": "",
                "prompt_abs_contrast_logit_change": "",
                "prompt_group_models": "",
                "pattern_count": count,
                "context": _shorten(text, int(args.context_chars)),
            }
        )
    rows.sort(key=lambda item: (-int(item["pattern_count"]), str(item["model"]), str(item["doc_id"])))
    rows = rows[: int(args.limit)]
    _write_csv(args.output_csv, rows)
    _write_md(args.output_md, rows, title=str(args.title))
    print(
        {
            "rows": len(rows),
            "excluded_doc_ids": len(excluded),
            "models": args.models,
            "output_csv": str(args.output_csv),
            "output_md": str(args.output_md),
        }
    )


if __name__ == "__main__":
    main()
