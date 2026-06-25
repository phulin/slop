from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INDEX_TEMPLATE = "apps/latent-explorer/public/data/layer{layer}/pangram_llama_sae_browser_index.json"


def _int(value: Any, default: int = -1) -> int:
    try:
        if value in ("", None):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _token_position(token_indices: list[Any], token_index: int) -> int:
    for index, value in enumerate(token_indices):
        if _int(value) == token_index:
            return index
    return -1


def _peak_token(doc: Any, token_index: int) -> str:
    position = _token_position(list(doc.token_indices), token_index)
    if position < 0:
        return ""
    tokens = list(doc.tokens)
    if position >= len(tokens):
        return ""
    return str(tokens[position]).strip()


def _is_punctuation(token: str) -> bool:
    token = token.strip()
    if not token:
        return False
    return re.fullmatch(r"[^\w]+", token, flags=re.UNICODE) is not None


def _top(counter: Counter[str], limit: int = 8) -> str:
    return "|".join(f"{token}:{count}" for token, count in counter.most_common(limit))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize peak-token distributions for top Pangram SAE browser latents."
    )
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--doc-tokens-parquet", type=Path, default=Path("apps/latent-explorer/public/data/layer19/pangram_llama_sae_doc_tokens.parquet"))
    parser.add_argument("--top-latents", type=int, default=120)
    parser.add_argument("--docs-per-latent", type=int, default=50)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = args.index or Path(DEFAULT_INDEX_TEMPLATE.format(layer=int(args.layer)))
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    latent_rows = sorted(payload.get("latents", []), key=lambda row: _int(row.get("rank"), 1_000_000))[
        : int(args.top_latents)
    ]
    latent_docs = payload.get("latentDocs", {})
    needed_doc_ids: set[str] = set()
    selected_docs: dict[int, list[dict[str, Any]]] = {}
    for row in latent_rows:
        latent_id = _int(row.get("latent_id"))
        docs = list(latent_docs.get(str(latent_id), []))[: int(args.docs_per_latent)]
        selected_docs[latent_id] = docs
        needed_doc_ids.update(str(doc.get("doc_id") or "") for doc in docs if doc.get("doc_id"))
    docs = pd.read_parquet(
        args.doc_tokens_parquet,
        columns=["doc_id", "token_indices", "tokens"],
    )
    docs = docs[docs["doc_id"].astype(str).isin(needed_doc_ids)]
    docs_by_id = {str(row.doc_id): row for row in docs.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    by_latent_row = {_int(row.get("latent_id")): row for row in latent_rows}
    for latent_id, docs_for_latent in selected_docs.items():
        peak_counts: Counter[str] = Counter()
        model_counts: Counter[str] = Counter()
        joined = 0
        missing = 0
        for doc in docs_for_latent:
            doc_id = str(doc.get("doc_id") or "")
            token_doc = docs_by_id.get(doc_id)
            if token_doc is None:
                missing += 1
                continue
            token = _peak_token(token_doc, _int(doc.get("token_index", doc.get("peak_token_index"))))
            if not token:
                missing += 1
                continue
            joined += 1
            peak_counts[token] += 1
            model_counts[str(doc.get("model") or doc.get("source") or "unknown")] += 1
        punctuation = sum(count for token, count in peak_counts.items() if _is_punctuation(token))
        periods = peak_counts.get(".", 0) + peak_counts.get(".”", 0) + peak_counts.get('."', 0)
        words = sum(count for token, count in peak_counts.items() if re.search(r"\w", token, flags=re.UNICODE))
        latent_row = by_latent_row[latent_id]
        rows.append(
            {
                "layer": int(args.layer),
                "latent_id": latent_id,
                "node": f"L{int(args.layer)}:{latent_id}",
                "rank": _int(latent_row.get("rank")),
                "joined_docs": joined,
                "missing_docs": missing,
                "punctuation_fraction": punctuation / max(1, joined),
                "period_fraction": periods / max(1, joined),
                "word_fraction": words / max(1, joined),
                "top_peak_tokens": _top(peak_counts),
                "model_counts": _top(model_counts),
                "max_abs_impact": _float(
                    latent_row.get(
                        "causal_max_abs_target_logit_change",
                        latent_row.get("exact_max_abs_prompt_contrast_logit_change"),
                    )
                ),
                "mean_signed_impact": _float(
                    latent_row.get(
                        "causal_mean_target_logit_drop",
                        latent_row.get("exact_mean_prompt_contrast_logit_drop"),
                    )
                ),
            }
        )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} latent peak-token summaries to {args.output_csv}")


if __name__ == "__main__":
    main()
