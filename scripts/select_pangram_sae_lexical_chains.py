from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PEAK_TEMPLATE = "artifacts/pangram_llama_sae/circuit_discovery/latent_peak_tokens_layer{layer}.csv"


def _node(layer: int, latent_id: int) -> str:
    return f"L{layer}:{latent_id}"


def _parse_node(raw: str) -> tuple[int, int]:
    layer, latent = raw.split(":", 1)
    return int(layer.removeprefix("L")), int(latent)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_peak_rows(layers: list[int], *, peak_template: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for layer in layers:
        path = Path(peak_template.format(layer=layer))
        df = pd.read_csv(path)
        for row in df.to_dict(orient="records"):
            node = str(row.get("node") or _node(int(row["layer"]), int(row["latent_id"])))
            rows[node] = row
    return rows


def _passes_peak_filter(
    row: dict[str, Any],
    *,
    min_joined_docs: int,
    min_word_fraction: float,
    max_punctuation_fraction: float,
    max_period_fraction: float,
) -> bool:
    return (
        int(row.get("joined_docs", 0)) >= min_joined_docs
        and _float(row.get("word_fraction")) >= min_word_fraction
        and _float(row.get("punctuation_fraction")) <= max_punctuation_fraction
        and _float(row.get("period_fraction")) <= max_period_fraction
    )


def _node_score(row: dict[str, Any]) -> float:
    word = _float(row.get("word_fraction"))
    punctuation = _float(row.get("punctuation_fraction"))
    period = _float(row.get("period_fraction"))
    impact = max(0.0, _float(row.get("max_abs_impact")))
    joined = max(1.0, _float(row.get("joined_docs")))
    return (0.25 + word) * (1.0 - 0.75 * punctuation) * (1.0 - 0.75 * period) * math.log1p(impact) * math.log1p(joined)


def _model_counts(row: dict[str, Any]) -> dict[str, int]:
    raw = str(row.get("model_counts", ""))
    counts: dict[str, int] = {}
    for part in raw.split("|"):
        if ":" not in part:
            continue
        model, count = part.rsplit(":", 1)
        try:
            counts[model] = int(count)
        except ValueError:
            continue
    return counts


def _human_fraction(row: dict[str, Any]) -> float:
    counts = _model_counts(row)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return counts.get("human", 0) / total


def _ai_fraction(row: dict[str, Any]) -> float:
    return 1.0 - _human_fraction(row)


def _chain_rows(edges: pd.DataFrame, peak_rows: dict[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out_by_left: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in edges.to_dict(orient="records"):
        out_by_left[str(row["left"])].append(row)
    for rows in out_by_left.values():
        rows.sort(key=lambda item: _float(item.get("edge_score")), reverse=True)

    chains: list[dict[str, Any]] = []
    for e16_19 in [row for row in edges.to_dict(orient="records") if int(row["left_layer"]) == 16 and int(row["right_layer"]) == 19]:
        for e19_20 in out_by_left.get(str(e16_19["right"]), [])[:40]:
            if int(e19_20["right_layer"]) != 20:
                continue
            for e20_24 in out_by_left.get(str(e19_20["right"]), [])[:40]:
                if int(e20_24["right_layer"]) != 24:
                    continue
                nodes = [
                    str(e16_19["left"]),
                    str(e16_19["right"]),
                    str(e19_20["right"]),
                    str(e20_24["right"]),
                ]
                if any(node not in peak_rows for node in nodes):
                    continue
                edge_geomean = (
                    _float(e16_19["edge_score"])
                    * _float(e19_20["edge_score"])
                    * _float(e20_24["edge_score"])
                ) ** (1 / 3)
                node_scores = [_node_score(peak_rows[node]) for node in nodes]
                human_fractions = [_human_fraction(peak_rows[node]) for node in nodes]
                ai_fractions = [_ai_fraction(peak_rows[node]) for node in nodes]
                lexical_score = edge_geomean * (sum(node_scores) / len(node_scores))
                chains.append(
                    {
                        "chain": " -> ".join(nodes),
                        "lexical_chain_score": lexical_score,
                        "geomean_edge_score": edge_geomean,
                        "min_word_fraction": min(_float(peak_rows[node].get("word_fraction")) for node in nodes),
                        "max_punctuation_fraction": max(_float(peak_rows[node].get("punctuation_fraction")) for node in nodes),
                        "max_period_fraction": max(_float(peak_rows[node].get("period_fraction")) for node in nodes),
                        "mean_ai_fraction": sum(ai_fractions) / len(ai_fractions),
                        "min_ai_fraction": min(ai_fractions),
                        "mean_human_fraction": sum(human_fractions) / len(human_fractions),
                        "max_human_fraction": max(human_fractions),
                        "min_joined_docs": min(int(peak_rows[node].get("joined_docs", 0)) for node in nodes),
                        "max_missing_docs": max(int(peak_rows[node].get("missing_docs", 0)) for node in nodes),
                        "edge_scores": "|".join(f"{_float(edge['edge_score']):.3f}" for edge in [e16_19, e19_20, e20_24]),
                        "shared_prompts": "|".join(str(int(edge["shared_prompts"])) for edge in [e16_19, e19_20, e20_24]),
                        "node_ranks": "|".join(str(int(peak_rows[node].get("rank", 0))) for node in nodes),
                        "node_top_peak_tokens": " || ".join(f"{node} {peak_rows[node].get('top_peak_tokens', '')}" for node in nodes),
                        "node_model_counts": " || ".join(f"{node} {peak_rows[node].get('model_counts', '')}" for node in nodes),
                    }
                )
    chains.sort(key=lambda row: _float(row["lexical_chain_score"]), reverse=True)
    return chains[:limit]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, Any]], *, title: str, max_rows: int) -> None:
    lines = [
        f"# {title}",
        "",
        "This queue filters the broad cross-layer candidate graph by peak-token distributions. It is intended to find lexical/style candidates rather than period or sentence-boundary accumulators.",
        "",
    ]
    for index, row in enumerate(rows[:max_rows], start=1):
        lines.extend(
            [
                f"## {index}. `{row['chain']}`",
                "",
                f"- Lexical score: `{float(row['lexical_chain_score']):.3f}`",
                f"- Edge geomean: `{float(row['geomean_edge_score']):.3f}`; edge scores `{row['edge_scores']}`; shared prompts `{row['shared_prompts']}`",
                f"- Word/punctuation: min word `{float(row['min_word_fraction']):.3f}`, max punctuation `{float(row['max_punctuation_fraction']):.3f}`, max period `{float(row['max_period_fraction']):.3f}`",
                f"- Source balance: mean AI `{float(row['mean_ai_fraction']):.3f}`, min AI `{float(row['min_ai_fraction']):.3f}`, mean human `{float(row['mean_human_fraction']):.3f}`",
                f"- Joined/missing docs: min joined `{int(row['min_joined_docs'])}`, max missing `{int(row['max_missing_docs'])}`",
                f"- Node ranks: `{row['node_ranks']}`",
                f"- Peak tokens: {row['node_top_peak_tokens']}",
                f"- Model counts: {row['node_model_counts']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select lexical/style Pangram SAE chains from cross-layer edges.")
    parser.add_argument("--cross-layer-edges", type=Path, default=Path("artifacts/pangram_llama_sae/circuit_discovery/cross_layer_edges.csv"))
    parser.add_argument("--peak-template", default=DEFAULT_PEAK_TEMPLATE)
    parser.add_argument("--min-joined-docs", type=int, default=8)
    parser.add_argument("--min-word-fraction", type=float, default=0.70)
    parser.add_argument("--max-punctuation-fraction", type=float, default=0.25)
    parser.add_argument("--max-period-fraction", type=float, default=0.25)
    parser.add_argument("--min-mean-ai-fraction", type=float, default=0.0)
    parser.add_argument("--max-mean-human-fraction", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = [16, 19, 20, 24]
    peak_rows_all = _load_peak_rows(layers, peak_template=str(args.peak_template))
    peak_rows = {
        node: row
        for node, row in peak_rows_all.items()
        if _passes_peak_filter(
            row,
            min_joined_docs=int(args.min_joined_docs),
            min_word_fraction=float(args.min_word_fraction),
            max_punctuation_fraction=float(args.max_punctuation_fraction),
            max_period_fraction=float(args.max_period_fraction),
        )
    }
    edges = pd.read_csv(args.cross_layer_edges)
    edges = edges[edges["left"].astype(str).isin(peak_rows) & edges["right"].astype(str).isin(peak_rows)]
    rows = _chain_rows(edges, peak_rows, max(int(args.limit) * 20, int(args.limit)))
    rows = [
        row
        for row in rows
        if float(row["mean_ai_fraction"]) >= float(args.min_mean_ai_fraction)
        and float(row["mean_human_fraction"]) <= float(args.max_mean_human_fraction)
    ][: int(args.limit)]
    _write_csv(args.output_csv, rows)
    _write_report(args.output_md, rows, title="Pangram SAE Lexical/Style Chain Queue", max_rows=min(20, int(args.limit)))
    print(
        {
            "filtered_nodes": len(peak_rows),
            "filtered_edges": len(edges),
            "chains": len(rows),
            "output_csv": str(args.output_csv),
            "output_md": str(args.output_md),
        }
    )


if __name__ == "__main__":
    main()
