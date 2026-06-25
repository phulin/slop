from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def _int(value: Any, default: int = -1) -> int:
    try:
        if value in ("", None):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _shorten(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _token_position(token_indices: list[Any], token_index: int) -> int:
    for index, value in enumerate(token_indices):
        if _int(value) == token_index:
            return index
    return -1


def _peak_window(doc: Any, token_index: int, window_tokens: int) -> tuple[str, str]:
    tokens = list(doc.tokens)
    token_indices = list(doc.token_indices)
    starts = list(doc.token_start_offsets)
    ends = list(doc.token_end_offsets)
    text = str(doc.text)
    position = _token_position(token_indices, token_index)
    if position < 0:
        return "", ""
    left = max(0, position - window_tokens)
    right = min(len(tokens), position + window_tokens + 1)
    char_start = _int(starts[left], 0)
    char_end = _int(ends[right - 1], len(text))
    peak_start = _int(starts[position], char_start)
    peak_end = _int(ends[position], peak_start)
    prefix = text[char_start:peak_start]
    peak = text[peak_start:peak_end]
    suffix = text[peak_end:char_end]
    token_window = "".join(str(token) for token in tokens[left:right])
    return f"{prefix}[[{peak}]]{suffix}", token_window


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "chain",
        "turn_id",
        "doc_id",
        "model",
        "node",
        "layer",
        "latent_id",
        "token_index",
        "peak_activation",
        "prompt_contrast_logit_drop",
        "text_window",
        "token_window",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(path: Path, rows: list[dict[str, Any]], *, max_prompts: int) -> None:
    by_prompt: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row["turn_id"])].append(row)
    lines: list[str] = [
        "# Pangram SAE Peak Windows",
        "",
        "Peak tokens are marked with `[[...]]`. These are already-exported target examples joined to tokenized documents; no model forward pass is used.",
        "",
    ]
    for prompt_index, (turn_id, prompt_rows) in enumerate(by_prompt.items()):
        if prompt_index >= max_prompts:
            break
        model = prompt_rows[0]["model"]
        doc_id = str(prompt_rows[0]["doc_id"])[:10]
        lines.extend([f"## {turn_id}", "", f"Model: `{model}`  Doc: `{doc_id}`", ""])
        for row in prompt_rows:
            lines.extend(
                [
                    "- `{node}` token `{token_index}` activation `{peak_activation}` impact `{prompt_contrast_logit_drop}`".format(**row),
                    "",
                    f"  {row['text_window']}",
                    "",
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export peak-token text windows for Pangram SAE circuit target rows.")
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--doc-tokens-parquet", type=Path, default=Path("apps/latent-explorer/public/data/layer19/pangram_llama_sae_doc_tokens.parquet"))
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--window-tokens", type=int, default=24)
    parser.add_argument("--max-prompts-md", type=int, default=8)
    parser.add_argument("--max-window-chars", type=int, default=420)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = pd.read_csv(args.target_csv)
    if "doc_id" not in targets.columns or targets.empty:
        _write_csv(args.output_csv, [])
        _write_md(args.output_md, [], max_prompts=0)
        print(f"wrote 0 peak-window rows to {args.output_csv}")
        return
    doc_ids = set(str(value) for value in targets["doc_id"].tolist())
    docs = pd.read_parquet(
        args.doc_tokens_parquet,
        columns=["doc_id", "text", "token_indices", "tokens", "token_start_offsets", "token_end_offsets"],
    )
    docs = docs[docs["doc_id"].astype(str).isin(doc_ids)]
    docs_by_id = {str(row.doc_id): row for row in docs.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for row in targets.itertuples(index=False):
        doc = docs_by_id.get(str(row.doc_id))
        if doc is None:
            continue
        text_window, token_window = _peak_window(doc, _int(row.token_index), int(args.window_tokens))
        rows.append(
            {
                "chain": row.chain,
                "turn_id": row.turn_id,
                "doc_id": row.doc_id,
                "model": row.model,
                "node": row.node,
                "layer": row.layer,
                "latent_id": row.latent_id,
                "token_index": row.token_index,
                "peak_activation": row.peak_activation,
                "prompt_contrast_logit_drop": row.prompt_contrast_logit_drop,
                "text_window": _shorten(text_window, int(args.max_window_chars)),
                "token_window": _shorten(token_window, int(args.max_window_chars)),
            }
        )
    _write_csv(args.output_csv, rows)
    _write_md(args.output_md, rows, max_prompts=int(args.max_prompts_md))
    print(f"wrote {len(rows)} peak-window rows to {args.output_csv}")


if __name__ == "__main__":
    main()
