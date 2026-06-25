from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RUNS = {
    16: Path("apps/latent-explorer/public/data/layer16/pangram_llama_sae_browser_index.json"),
    19: Path("apps/latent-explorer/public/data/layer19/pangram_llama_sae_browser_index.json"),
    20: Path("apps/latent-explorer/public/data/layer20/pangram_llama_sae_browser_index.json"),
    24: Path("apps/latent-explorer/public/data/layer24/pangram_llama_sae_browser_index.json"),
}


@dataclass(frozen=True)
class Latent:
    layer: int
    latent_id: int
    rank: int
    prompt_ids: frozenset[str]
    models: frozenset[str]

    @property
    def node(self) -> str:
        return f"L{self.layer}:{self.latent_id}"


def _parse_node(raw: str) -> tuple[int, int]:
    raw = raw.strip()
    if not raw.startswith("L") or ":" not in raw:
        raise ValueError(f"invalid node {raw!r}; expected L19:1234")
    layer, latent = raw.split(":", 1)
    return int(layer[1:]), int(latent)


def _parse_chain(raw: str) -> list[str]:
    nodes = [part.strip() for part in raw.split("->") if part.strip()]
    if len(nodes) < 2:
        raise ValueError("--chain must contain at least two nodes")
    for node in nodes:
        _parse_node(node)
    return nodes


def _parse_run_map(values: list[str]) -> dict[int, Path]:
    out = dict(DEFAULT_RUNS)
    for value in values:
        layer, path = value.split("=", 1)
        out[int(layer)] = Path(path)
    return out


def _int(value: Any, default: int = 1_000_000) -> int:
    try:
        if value in ("", None):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_layer(path: Path, layer: int, *, top_latents: int) -> dict[str, Latent]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(payload.get("latents", []), key=lambda row: _int(row.get("rank")))[:top_latents]
    docs_by_latent = payload.get("latentDocs", {})
    out: dict[str, Latent] = {}
    for row in rows:
        latent_id = _int(row.get("latent_id"))
        docs = docs_by_latent.get(str(latent_id), [])
        prompts = frozenset(str(doc.get("turn_id") or "") for doc in docs if doc.get("turn_id"))
        models = frozenset(str(doc.get("model") or doc.get("source") or "unknown") for doc in docs)
        latent = Latent(
            layer=layer,
            latent_id=latent_id,
            rank=_int(row.get("rank")),
            prompt_ids=prompts,
            models=models,
        )
        out[latent.node] = latent
    return out


def _candidate_score(anchor: Latent, target: Latent, candidate: Latent) -> tuple[float, int, int]:
    shared = len(anchor.prompt_ids & candidate.prompt_ids)
    target_shared = len(anchor.prompt_ids & target.prompt_ids)
    rank_distance = abs(candidate.rank - target.rank)
    # Prefer a control that is similarly ranked but does not share the same high-impact prompts.
    overlap_penalty = shared / max(1, target_shared)
    score = rank_distance + 25.0 * overlap_penalty
    return score, shared, rank_distance


def _pick_controls(
    *,
    layer_latents: dict[int, dict[str, Latent]],
    upstream: Latent,
    downstream: Latent,
    controls_per_type: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    downstream_layer = layer_latents[downstream.layer]
    upstream_layer = layer_latents[upstream.layer]

    downstream_candidates: list[tuple[float, int, int, Latent]] = []
    for candidate in downstream_layer.values():
        if candidate.node == downstream.node:
            continue
        score, shared, rank_distance = _candidate_score(upstream, downstream, candidate)
        downstream_candidates.append((score, shared, rank_distance, candidate))
    downstream_candidates.sort(key=lambda item: item[:3])
    for score, shared, rank_distance, candidate in downstream_candidates[:controls_per_type]:
        rows.append(
            {
                "control_type": "same_upstream_different_downstream",
                "upstream_node": upstream.node,
                "downstream_node": candidate.node,
                "chain_upstream_node": upstream.node,
                "chain_downstream_node": downstream.node,
                "candidate_rank": candidate.rank,
                "chain_rank": downstream.rank,
                "rank_distance": rank_distance,
                "shared_prompts_with_anchor": shared,
                "chain_shared_prompts_with_anchor": len(upstream.prompt_ids & downstream.prompt_ids),
                "selection_score": score,
            }
        )

    upstream_candidates: list[tuple[float, int, int, Latent]] = []
    for candidate in upstream_layer.values():
        if candidate.node == upstream.node:
            continue
        score, shared, rank_distance = _candidate_score(downstream, upstream, candidate)
        upstream_candidates.append((score, shared, rank_distance, candidate))
    upstream_candidates.sort(key=lambda item: item[:3])
    for score, shared, rank_distance, candidate in upstream_candidates[:controls_per_type]:
        rows.append(
            {
                "control_type": "different_upstream_same_downstream",
                "upstream_node": candidate.node,
                "downstream_node": downstream.node,
                "chain_upstream_node": upstream.node,
                "chain_downstream_node": downstream.node,
                "candidate_rank": candidate.rank,
                "chain_rank": upstream.rank,
                "rank_distance": rank_distance,
                "shared_prompts_with_anchor": shared,
                "chain_shared_prompts_with_anchor": len(upstream.prompt_ids & downstream.prompt_ids),
                "selection_score": score,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "target_name",
        "target_csv",
        "control_type",
        "upstream_node",
        "downstream_node",
        "chain_upstream_node",
        "chain_downstream_node",
        "candidate_rank",
        "chain_rank",
        "rank_distance",
        "shared_prompts_with_anchor",
        "chain_shared_prompts_with_anchor",
        "selection_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select low-overlap matched controls for Pangram SAE circuit intervention tests."
    )
    parser.add_argument("--chain", action="append", required=True, help="Chain like 'L16:1 -> L19:2 -> L24:3'.")
    parser.add_argument("--target-name", action="append", required=True)
    parser.add_argument("--target-csv", action="append", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], metavar="LAYER=PATH")
    parser.add_argument("--top-latents-per-layer", type=int, default=160)
    parser.add_argument("--controls-per-type", type=int, default=3)
    parser.add_argument(
        "--skip-layer",
        action="append",
        type=int,
        default=[],
        help="Skip hops touching this layer, useful when the matching SAE checkpoint is unavailable.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (len(args.chain) == len(args.target_name) == len(args.target_csv)):
        raise ValueError("--chain, --target-name, and --target-csv counts must match")
    run_map = _parse_run_map(args.run)
    needed_layers = sorted({_parse_node(node)[0] for chain in args.chain for node in _parse_chain(chain)})
    layer_latents = {
        layer: _load_layer(run_map[layer], layer, top_latents=int(args.top_latents_per_layer))
        for layer in needed_layers
    }
    skip_layers = set(int(layer) for layer in args.skip_layer)
    rows: list[dict[str, Any]] = []
    for chain, target_name, target_csv in zip(args.chain, args.target_name, args.target_csv, strict=True):
        nodes = _parse_chain(chain)
        for left_node, right_node in zip(nodes, nodes[1:], strict=False):
            left_layer, _ = _parse_node(left_node)
            right_layer, _ = _parse_node(right_node)
            if left_layer in skip_layers or right_layer in skip_layers:
                continue
            try:
                upstream = layer_latents[left_layer][left_node]
                downstream = layer_latents[right_layer][right_node]
            except KeyError as exc:
                raise ValueError(
                    f"{exc.args[0]} is missing from the loaded top-{args.top_latents_per_layer} layer index"
                ) from exc
            for row in _pick_controls(
                layer_latents=layer_latents,
                upstream=upstream,
                downstream=downstream,
                controls_per_type=int(args.controls_per_type),
            ):
                row["target_name"] = target_name
                row["target_csv"] = str(target_csv)
                rows.append(row)
    rows.sort(
        key=lambda row: (
            str(row["target_name"]),
            str(row["chain_upstream_node"]),
            str(row["chain_downstream_node"]),
            str(row["control_type"]),
            float(row["selection_score"])
            if not isinstance(row["selection_score"], float) or math.isfinite(row["selection_score"])
            else float("inf"),
        )
    )
    _write_csv(args.output_csv, rows)
    print(f"wrote {len(rows)} control candidates to {args.output_csv}")


if __name__ == "__main__":
    main()
