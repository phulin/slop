from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INDEX_TEMPLATE = "apps/latent-explorer/public/data/layer{layer}/pangram_llama_sae_browser_index.json"


def _parse_node(raw: str) -> tuple[int, str]:
    raw = raw.strip()
    if not raw.startswith("L") or ":" not in raw:
        raise ValueError(f"invalid node {raw!r}; expected L19:1234")
    layer_text, latent_id = raw.split(":", 1)
    return int(layer_text[1:]), str(int(latent_id))


def _shorten(text: str, limit: int = 260) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _load_index(layer: int) -> dict[str, Any]:
    return json.loads(Path(DEFAULT_INDEX_TEMPLATE.format(layer=layer)).read_text(encoding="utf-8"))


def _node_docs(index: dict[str, Any], latent_id: str) -> list[dict[str, Any]]:
    return list(index.get("latentDocs", {}).get(latent_id, []))


def _doc_by_prompt(docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for doc in docs:
        turn_id = str(doc.get("turn_id") or "")
        if not turn_id:
            continue
        current = best.get(turn_id)
        if current is None or float(doc.get("peak_activation", doc.get("activation", 0.0)) or 0.0) > float(
            current.get("peak_activation", current.get("activation", 0.0)) or 0.0
        ):
            best[turn_id] = doc
    return best


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, chain: str, common_prompts: list[str], rows: list[dict[str, Any]]) -> None:
    by_prompt: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[str(row["turn_id"])].append(row)
    lines = [
        f"# Circuit Target Export: `{chain}`",
        "",
        f"Common prompts across all nodes: `{len(common_prompts)}`",
        "",
        "These are intervention targets, not confirmed circuit evidence. Each prompt group should be tested by ablating or clamping upstream nodes and rescanning downstream SAE activations.",
        "",
    ]
    for turn_id in common_prompts[:20]:
        lines.append(f"## {turn_id}")
        lines.append("")
        for row in by_prompt[turn_id]:
            lines.append(
                "- `{node}` model `{model}` token `{token_index}` activation `{peak_activation}` "
                "impact `{prompt_contrast_logit_drop}`: {context}".format(**row)
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export shared prompt/token targets for a Pangram SAE circuit seed.")
    parser.add_argument("--chain", required=True, help="Chain like 'L16:1 -> L19:2 -> L20:3 -> L24:4'.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes = [_parse_node(part) for part in args.chain.replace(",", " -> ").split("->")]
    by_layer = {layer: _load_index(layer) for layer, _latent_id in nodes}
    docs_by_node = {
        (layer, latent_id): _node_docs(by_layer[layer], latent_id)
        for layer, latent_id in nodes
    }
    prompt_maps = {node: _doc_by_prompt(docs) for node, docs in docs_by_node.items()}
    common = set.intersection(*(set(mapping) for mapping in prompt_maps.values()))

    def prompt_score(turn_id: str) -> float:
        score = 0.0
        for mapping in prompt_maps.values():
            doc = mapping[turn_id]
            score += abs(float(doc.get("prompt_contrast_logit_drop", doc.get("causal_logit_drop", 0.0)) or 0.0))
        return score

    common_prompts = sorted(common, key=prompt_score, reverse=True)[: int(args.max_prompts)]
    rows: list[dict[str, Any]] = []
    for turn_id in common_prompts:
        for layer, latent_id in nodes:
            doc = prompt_maps[(layer, latent_id)][turn_id]
            prompt_group = by_layer[layer].get("promptGroups", {}).get(turn_id, [])
            rows.append(
                {
                    "chain": args.chain,
                    "node": f"L{layer}:{latent_id}",
                    "layer": layer,
                    "latent_id": latent_id,
                    "turn_id": turn_id,
                    "doc_id": doc.get("doc_id", ""),
                    "model": doc.get("model", doc.get("source", "")),
                    "peak_activation": doc.get("peak_activation", doc.get("activation", "")),
                    "token_index": doc.get("token_index", doc.get("peak_token_index", "")),
                    "prompt_contrast_logit_drop": doc.get("prompt_contrast_logit_drop", doc.get("causal_logit_drop", "")),
                    "prompt_abs_contrast_logit_change": doc.get("prompt_abs_contrast_logit_change", doc.get("causal_abs_logit_change", "")),
                    "prompt_group_models": "|".join(
                        str(by_layer[layer].get("documents", {}).get(doc_id, {}).get("model", ""))
                        for doc_id in prompt_group
                    ),
                    "context": _shorten(str(doc.get("context", ""))),
                }
            )
    _write_csv(args.output_csv, rows)
    _write_report(args.output_md, args.chain, common_prompts, rows)
    print(
        json.dumps(
            {
                "chain": args.chain,
                "common_prompts": len(common),
                "exported_prompts": len(common_prompts),
                "rows": len(rows),
                "output_csv": str(args.output_csv),
                "output_md": str(args.output_md),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
