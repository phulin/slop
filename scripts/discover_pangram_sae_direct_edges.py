from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from discover_pangram_sae_circuit_seeds import DEFAULT_RUNS, LatentSummary, _cosine, load_layer


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _shared_examples(left: LatentSummary, right: LatentSummary, limit: int = 3) -> str:
    # Contexts are already shortened in the source loader. Use the left side as a
    # compact semantic hint for manual triage.
    return " || ".join(left.example_contexts[:limit] + right.example_contexts[:limit])


def build_direct_edges(
    left_rows: list[LatentSummary],
    right_rows: list[LatentSummary],
    *,
    min_shared_prompts: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left in left_rows:
        for right in right_rows:
            shared = left.prompt_ids & right.prompt_ids
            if len(shared) < min_shared_prompts:
                continue
            union = left.prompt_ids | right.prompt_ids
            jaccard = len(shared) / max(1, len(union))
            model_abs_cosine = _cosine(left.model_mean_abs_impact, right.model_mean_abs_impact)
            model_signed_cosine = _cosine(left.model_mean_signed_impact, right.model_mean_signed_impact)
            edge_score = (
                len(shared)
                * (0.5 + jaccard)
                * max(0.0, model_abs_cosine)
                * (0.5 + 0.5 * max(-1.0, min(1.0, model_signed_cosine)))
            )
            shared_fraction_left = len(shared) / max(1, len(left.prompt_ids))
            shared_fraction_right = len(shared) / max(1, len(right.prompt_ids))
            rows.append(
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
                    "shared_fraction_left": shared_fraction_left,
                    "shared_fraction_right": shared_fraction_right,
                    "jaccard": jaccard,
                    "model_abs_impact_cosine": model_abs_cosine,
                    "model_signed_impact_cosine": model_signed_cosine,
                    "edge_score": edge_score,
                    "left_mean_signed_impact": left.mean_signed_impact,
                    "right_mean_signed_impact": right.mean_signed_impact,
                    "left_max_abs_impact": left.max_abs_impact,
                    "right_max_abs_impact": right.max_abs_impact,
                    "left_models": "|".join(f"{key}:{value}" for key, value in sorted(left.model_counts.items())),
                    "right_models": "|".join(f"{key}:{value}" for key, value in sorted(right.model_counts.items())),
                    "example_contexts": _shared_examples(left, right),
                }
            )
    rows.sort(key=lambda row: float(row["edge_score"]), reverse=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Pangram SAE Direct Edge Seeds",
        "",
        "This ranks direct cross-layer SAE latent pairs from existing latent explorer exports. "
        "It is a seed-selection pass only; direct edges need intervention tests before being treated as circuits.",
        "",
        "## Method",
        "",
        f"- Left layer: `{args.left_layer}`; right layer: `{args.right_layer}`.",
        f"- Top latents per layer: `{args.top_latents_per_layer}`.",
        f"- Minimum shared prompts: `{args.min_shared_prompts}`.",
        "- Score uses shared prompt count, prompt-set Jaccard, model-impact cosine, and signed-effect cosine.",
        "",
        "## Top Edges",
        "",
        "| Rank | Edge | Score | Shared prompts | Jaccard | Signed cosine | Left rank | Right rank | Models |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(rows[: int(args.report_top_edges)], start=1):
        lines.append(
            "| {rank} | `{left} -> {right}` | `{edge_score:.3f}` | {shared_prompts} | `{jaccard:.3f}` | `{model_signed_impact_cosine:.3f}` | {left_rank} | {right_rank} | {left_models} / {right_models} |".format(
                rank=rank,
                **row,
            )
        )
    lines.extend(["", "## Manual Triage Notes", ""])
    for rank, row in enumerate(rows[: int(args.report_top_edges)], start=1):
        lines.extend(
            [
                f"### {rank}. `{row['left']} -> {row['right']}`",
                "",
                f"- Shared prompts: `{row['shared_prompts']}`; edge score: `{float(row['edge_score']):.3f}`.",
                f"- Mean signed impacts: left `{float(row['left_mean_signed_impact']):.3f}`, right `{float(row['right_mean_signed_impact']):.3f}`.",
                f"- Example contexts: {row['example_contexts']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover direct Pangram SAE cross-layer edge seeds.")
    parser.add_argument("--left-layer", type=int, default=19)
    parser.add_argument("--right-layer", type=int, default=24)
    parser.add_argument("--left-index", type=Path, default=None)
    parser.add_argument("--right-index", type=Path, default=None)
    parser.add_argument("--top-latents-per-layer", type=int, default=200)
    parser.add_argument("--min-shared-prompts", type=int, default=8)
    parser.add_argument("--report-top-edges", type=int, default=30)
    parser.add_argument("--output-csv", type=Path, default=Path("artifacts/pangram_llama_sae/circuit_discovery/direct_edges_L19_to_L24.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/experiments/pangram_sae_direct_edges_L19_to_L24.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left_index = args.left_index or DEFAULT_RUNS[int(args.left_layer)]
    right_index = args.right_index or DEFAULT_RUNS[int(args.right_layer)]
    left_rows = load_layer(left_index, int(args.left_layer), int(args.top_latents_per_layer))
    right_rows = load_layer(right_index, int(args.right_layer), int(args.top_latents_per_layer))
    edges = build_direct_edges(
        left_rows,
        right_rows,
        min_shared_prompts=int(args.min_shared_prompts),
    )
    write_csv(args.output_csv, edges)
    write_report(args.output_md, edges, args)
    print(
        json.dumps(
            {
                "left_layer": int(args.left_layer),
                "right_layer": int(args.right_layer),
                "left_latents": len(left_rows),
                "right_latents": len(right_rows),
                "edges": len(edges),
                "output_csv": str(args.output_csv),
                "output_md": str(args.output_md),
                "top_edge": edges[0] if edges else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
