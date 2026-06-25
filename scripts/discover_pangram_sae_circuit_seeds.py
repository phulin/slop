from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
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
class LatentSummary:
    layer: int
    latent_id: int
    rank: int
    max_abs_impact: float
    mean_abs_impact: float
    mean_signed_impact: float
    linearized_mean_abs: float
    prompt_ids: frozenset[str]
    model_counts: dict[str, int]
    model_mean_abs_impact: dict[str, float]
    model_mean_signed_impact: dict[str, float]
    example_contexts: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"L{self.layer}:{self.latent_id}"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in ("", None):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _shorten(text: str, limit: int = 220) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def load_layer(index_path: Path, layer: int, top_latents: int) -> list[LatentSummary]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    latent_rows = sorted(
        payload.get("latents", []),
        key=lambda row: _int(row.get("rank"), 1_000_000),
    )[:top_latents]
    latent_docs = payload.get("latentDocs", {})
    summaries: list[LatentSummary] = []
    for row in latent_rows:
        latent_id = _int(row.get("latent_id"))
        docs = latent_docs.get(str(latent_id), [])
        prompt_ids: set[str] = set()
        model_counts: Counter[str] = Counter()
        model_abs: defaultdict[str, list[float]] = defaultdict(list)
        model_signed: defaultdict[str, list[float]] = defaultdict(list)
        contexts: list[str] = []
        for doc in docs:
            prompt_id = str(doc.get("turn_id") or "")
            if prompt_id:
                prompt_ids.add(prompt_id)
            model = str(doc.get("model") or doc.get("source") or "unknown")
            signed = _float(
                doc.get("prompt_contrast_logit_drop", doc.get("causal_logit_drop"))
            )
            abs_impact = _float(
                doc.get("prompt_abs_contrast_logit_change", doc.get("causal_abs_logit_change")),
                abs(signed),
            )
            model_counts[model] += 1
            model_abs[model].append(abs_impact)
            model_signed[model].append(signed)
            if len(contexts) < 3:
                contexts.append(_shorten(str(doc.get("context", ""))))
        summaries.append(
            LatentSummary(
                layer=layer,
                latent_id=latent_id,
                rank=_int(row.get("rank")),
                max_abs_impact=_float(
                    row.get(
                        "causal_max_abs_target_logit_change",
                        row.get("exact_max_abs_prompt_contrast_logit_change"),
                    )
                ),
                mean_abs_impact=_float(
                    row.get(
                        "causal_mean_abs_target_logit_change",
                        row.get("exact_mean_abs_prompt_contrast_logit_change"),
                    )
                ),
                mean_signed_impact=_float(
                    row.get(
                        "causal_mean_target_logit_drop",
                        row.get("exact_mean_prompt_contrast_logit_drop"),
                    )
                ),
                linearized_mean_abs=_float(
                    row.get("linearized_mean_abs_estimated_logit_change")
                ),
                prompt_ids=frozenset(prompt_ids),
                model_counts=dict(model_counts),
                model_mean_abs_impact={
                    model: sum(values) / max(1, len(values))
                    for model, values in model_abs.items()
                },
                model_mean_signed_impact={
                    model: sum(values) / max(1, len(values))
                    for model, values in model_signed.items()
                },
                example_contexts=tuple(contexts),
            )
        )
    return summaries


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    dot = sum(left.get(key, 0.0) * right.get(key, 0.0) for key in keys)
    left_norm = math.sqrt(sum(left.get(key, 0.0) ** 2 for key in keys))
    right_norm = math.sqrt(sum(right.get(key, 0.0) ** 2 for key in keys))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def build_edges(
    by_layer: dict[int, list[LatentSummary]],
    *,
    min_shared_prompts: int,
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    adjacent = [(16, 19), (19, 20), (20, 24)]
    for left_layer, right_layer in adjacent:
        for left in by_layer.get(left_layer, []):
            for right in by_layer.get(right_layer, []):
                shared = left.prompt_ids & right.prompt_ids
                if len(shared) < min_shared_prompts:
                    continue
                union = left.prompt_ids | right.prompt_ids
                jaccard = len(shared) / max(1, len(union))
                model_cosine = _cosine(left.model_mean_abs_impact, right.model_mean_abs_impact)
                direction_agreement = _cosine(
                    left.model_mean_signed_impact,
                    right.model_mean_signed_impact,
                )
                edge_score = (
                    len(shared)
                    * (0.5 + jaccard)
                    * max(0.0, model_cosine)
                    * (0.5 + 0.5 * max(-1.0, min(1.0, direction_agreement)))
                )
                edges.append(
                    {
                        "left": left.key,
                        "right": right.key,
                        "left_layer": left.layer,
                        "right_layer": right.layer,
                        "left_latent_id": left.latent_id,
                        "right_latent_id": right.latent_id,
                        "left_rank": left.rank,
                        "right_rank": right.rank,
                        "shared_prompts": len(shared),
                        "jaccard": jaccard,
                        "model_abs_impact_cosine": model_cosine,
                        "model_signed_impact_cosine": direction_agreement,
                        "edge_score": edge_score,
                    }
                )
    edges.sort(key=lambda row: float(row["edge_score"]), reverse=True)
    return edges


def build_chains(edges: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out_by_left: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        out_by_left[str(edge["left"])].append(edge)
    for rows in out_by_left.values():
        rows.sort(key=lambda row: float(row["edge_score"]), reverse=True)

    chains: list[dict[str, Any]] = []
    for e16_19 in [edge for edge in edges if edge["left_layer"] == 16 and edge["right_layer"] == 19]:
        for e19_20 in out_by_left.get(str(e16_19["right"]), [])[:8]:
            if e19_20["right_layer"] != 20:
                continue
            for e20_24 in out_by_left.get(str(e19_20["right"]), [])[:8]:
                if e20_24["right_layer"] != 24:
                    continue
                score = (
                    float(e16_19["edge_score"])
                    * float(e19_20["edge_score"])
                    * float(e20_24["edge_score"])
                ) ** (1 / 3)
                chains.append(
                    {
                        "chain": " -> ".join(
                            [
                                str(e16_19["left"]),
                                str(e16_19["right"]),
                                str(e19_20["right"]),
                                str(e20_24["right"]),
                            ]
                        ),
                        "geomean_edge_score": score,
                        "min_shared_prompts": min(
                            int(e16_19["shared_prompts"]),
                            int(e19_20["shared_prompts"]),
                            int(e20_24["shared_prompts"]),
                        ),
                        "edge_scores": "|".join(
                            f"{float(edge['edge_score']):.3f}"
                            for edge in (e16_19, e19_20, e20_24)
                        ),
                        "shared_prompts": "|".join(
                            str(int(edge["shared_prompts"]))
                            for edge in (e16_19, e19_20, e20_24)
                        ),
                    }
                )
    chains.sort(key=lambda row: float(row["geomean_edge_score"]), reverse=True)
    return chains[:limit]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    by_layer: dict[int, list[LatentSummary]],
    edges: list[dict[str, Any]],
    chains: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Pangram SAE Circuit Seed Report",
        "",
        "This is an initial, low-cost circuit-discovery pass over existing latent explorer exports. "
        "It does not claim mechanistic circuits yet; it identifies cross-layer SAE latent chains "
        "worth deeper intervention tests.",
        "",
        "## Inputs",
        "",
    ]
    for layer, path_in in sorted(args.layer_index):
        lines.append(f"- Layer {layer}: `{path_in}`")
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Keep the top causal/reranked latents per layer.",
            "- For each latent, use its displayed high-impact prompt IDs, model mix, and signed prompt-contrast logit drops.",
            "- Connect adjacent-layer latents when they fire on overlapping high-impact prompts.",
            "- Rank edges by shared prompts, prompt-set Jaccard, model-impact profile cosine, and signed-direction agreement.",
            "- Greedily assemble layer 16 -> 19 -> 20 -> 24 chains from high-scoring adjacent edges.",
            "",
            "## Layer Inventory",
            "",
        ]
    )
    for layer, summaries in sorted(by_layer.items()):
        top = summaries[:5]
        lines.append(f"### Layer {layer}")
        lines.append("")
        for item in top:
            model_counts = ", ".join(f"{k}:{v}" for k, v in sorted(item.model_counts.items()))
            lines.append(
                f"- `{item.key}` rank {item.rank}, max abs impact {item.max_abs_impact:.3f}, "
                f"mean signed impact {item.mean_signed_impact:.3f}, prompts {len(item.prompt_ids)}, "
                f"models {model_counts}."
            )
        lines.append("")
    lines.extend(["## Top Cross-Layer Edges", ""])
    for row in edges[:20]:
        lines.append(
            "- `{left}` -> `{right}`: score {edge_score:.3f}, shared prompts {shared_prompts}, "
            "Jaccard {jaccard:.3f}, model cosine {model_abs_impact_cosine:.3f}, "
            "signed cosine {model_signed_impact_cosine:.3f}.".format(**row)
        )
    lines.extend(["", "## Candidate Chains", ""])
    if chains:
        for row in chains[:20]:
            lines.append(
                "- `{chain}`: geomean score {geomean_edge_score:.3f}, "
                "min shared prompts {min_shared_prompts}, edge scores {edge_scores}.".format(**row)
            )
    else:
        lines.append("- No full 16 -> 19 -> 20 -> 24 chain met the current overlap threshold.")
    lines.extend(
        [
            "",
            "## Immediate Follow-Ups",
            "",
            "1. Open the top chains in the latent explorer and assign semantic labels to each node.",
            "2. For a labeled chain, run activation patching or latent clamping at each layer on the same prompt group.",
            "3. Test whether upstream-latent ablation reduces downstream-latent activation on the same tokens/prompts.",
            "4. Split candidates into likely stylistic circuits, task-format artifacts, and detector bookkeeping directions.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover cross-layer Pangram SAE circuit seeds.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pangram_llama_sae/circuit_discovery"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/experiments/pangram_sae_circuit_seed_report.md"),
    )
    parser.add_argument("--top-latents-per-layer", type=int, default=120)
    parser.add_argument("--min-shared-prompts", type=int, default=3)
    parser.add_argument("--chain-limit", type=int, default=50)
    parser.add_argument(
        "--layer-index",
        action="append",
        nargs=2,
        metavar=("LAYER", "INDEX_JSON"),
        default=None,
        help="Layer number and browser index JSON. Can be repeated.",
    )
    args = parser.parse_args()
    if args.layer_index is None:
        args.layer_index = [(str(layer), str(path)) for layer, path in DEFAULT_RUNS.items()]
    args.layer_index = [(int(layer), Path(path)) for layer, path in args.layer_index]
    return args


def main() -> None:
    args = parse_args()
    by_layer = {
        layer: load_layer(path, layer, top_latents=int(args.top_latents_per_layer))
        for layer, path in args.layer_index
    }
    latent_rows: list[dict[str, Any]] = []
    for layer, summaries in sorted(by_layer.items()):
        for item in summaries:
            latent_rows.append(
                {
                    "layer": item.layer,
                    "latent_id": item.latent_id,
                    "key": item.key,
                    "rank": item.rank,
                    "prompt_count": len(item.prompt_ids),
                    "max_abs_impact": item.max_abs_impact,
                    "mean_abs_impact": item.mean_abs_impact,
                    "mean_signed_impact": item.mean_signed_impact,
                    "linearized_mean_abs": item.linearized_mean_abs,
                    "model_counts": "|".join(f"{k}:{v}" for k, v in sorted(item.model_counts.items())),
                    "example_1": item.example_contexts[0] if len(item.example_contexts) > 0 else "",
                    "example_2": item.example_contexts[1] if len(item.example_contexts) > 1 else "",
                    "example_3": item.example_contexts[2] if len(item.example_contexts) > 2 else "",
                }
            )
    edges = build_edges(by_layer, min_shared_prompts=int(args.min_shared_prompts))
    chains = build_chains(edges, limit=int(args.chain_limit))
    write_csv(args.output_dir / "latent_summaries.csv", latent_rows)
    write_csv(args.output_dir / "cross_layer_edges.csv", edges)
    write_csv(args.output_dir / "candidate_chains.csv", chains)
    write_report(args.report, by_layer, edges, chains, args)
    print(
        json.dumps(
            {
                "layers": {str(layer): len(rows) for layer, rows in by_layer.items()},
                "edges": len(edges),
                "chains": len(chains),
                "output_dir": str(args.output_dir),
                "report": str(args.report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
