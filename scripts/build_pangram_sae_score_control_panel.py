from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd


def _node_label(nodes: list[str]) -> str:
    return "+".join(nodes)


def _add_row(
    rows: list[dict[str, Any]],
    *,
    target_name: str,
    target_csv: str,
    ablation_label: str,
    control_type: str,
    nodes: list[str],
    anchor_nodes: list[str],
    source_control_row: dict[str, Any] | None = None,
) -> None:
    row: dict[str, Any] = {
        "target_name": target_name,
        "target_csv": target_csv,
        "ablation_label": ablation_label,
        "control_type": control_type,
        "nodes": _node_label(nodes),
        "anchor_nodes": _node_label(anchor_nodes),
    }
    if source_control_row:
        for key in [
            "upstream_node",
            "downstream_node",
            "candidate_rank",
            "chain_rank",
            "rank_distance",
            "shared_prompts_with_anchor",
            "selection_score",
        ]:
            row[f"control_{key}"] = source_control_row.get(key, "")
    rows.append(row)


def _build_rows(
    panel: pd.DataFrame,
    *,
    target_name: str,
    target_csv: str | None,
    limit_controls: int,
) -> list[dict[str, Any]]:
    target_panel = panel[panel["target_name"] == target_name]
    if target_panel.empty:
        raise ValueError(f"no rows found for target_name={target_name!r}")
    first = target_panel.iloc[0].to_dict()
    chain_upstream = str(first["chain_upstream_node"])
    chain_downstream = str(first["chain_downstream_node"])
    resolved_target_csv = str(target_csv or first["target_csv"])
    anchor_nodes = [chain_upstream, chain_downstream]
    rows: list[dict[str, Any]] = []
    _add_row(
        rows,
        target_name=target_name,
        target_csv=resolved_target_csv,
        ablation_label="anchor_upstream_only",
        control_type="anchor",
        nodes=[chain_upstream],
        anchor_nodes=anchor_nodes,
    )
    _add_row(
        rows,
        target_name=target_name,
        target_csv=resolved_target_csv,
        ablation_label="anchor_downstream_only",
        control_type="anchor",
        nodes=[chain_downstream],
        anchor_nodes=anchor_nodes,
    )
    _add_row(
        rows,
        target_name=target_name,
        target_csv=resolved_target_csv,
        ablation_label="anchor_joint",
        control_type="anchor",
        nodes=anchor_nodes,
        anchor_nodes=anchor_nodes,
    )

    different_upstream = target_panel[target_panel["control_type"] == "different_upstream_same_downstream"].head(limit_controls)
    for index, control in enumerate(different_upstream.to_dict(orient="records"), start=1):
        control_upstream = str(control["upstream_node"])
        _add_row(
            rows,
            target_name=target_name,
            target_csv=resolved_target_csv,
            ablation_label=f"control_upstream_only_{index}",
            control_type="different_upstream_same_downstream",
            nodes=[control_upstream],
            anchor_nodes=anchor_nodes,
            source_control_row=control,
        )
        _add_row(
            rows,
            target_name=target_name,
            target_csv=resolved_target_csv,
            ablation_label=f"control_upstream_plus_anchor_downstream_{index}",
            control_type="different_upstream_same_downstream",
            nodes=[control_upstream, chain_downstream],
            anchor_nodes=anchor_nodes,
            source_control_row=control,
        )

    different_downstream = target_panel[target_panel["control_type"] == "same_upstream_different_downstream"].head(limit_controls)
    for index, control in enumerate(different_downstream.to_dict(orient="records"), start=1):
        control_downstream = str(control["downstream_node"])
        _add_row(
            rows,
            target_name=target_name,
            target_csv=resolved_target_csv,
            ablation_label=f"control_downstream_only_{index}",
            control_type="same_upstream_different_downstream",
            nodes=[control_downstream],
            anchor_nodes=anchor_nodes,
            source_control_row=control,
        )
        _add_row(
            rows,
            target_name=target_name,
            target_csv=resolved_target_csv,
            ablation_label=f"anchor_upstream_plus_control_downstream_{index}",
            control_type="same_upstream_different_downstream",
            nodes=[chain_upstream, control_downstream],
            anchor_nodes=anchor_nodes,
            source_control_row=control,
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Pangram SAE score-ablation panel from an existing matched-control panel.")
    parser.add_argument("--control-csv", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--target-csv", default=None, help="Optional replacement target CSV, e.g. a held-out target set.")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--limit-controls", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = pd.read_csv(args.control_csv)
    rows = _build_rows(
        panel,
        target_name=str(args.target_name),
        target_csv=args.target_csv,
        limit_controls=int(args.limit_controls),
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "target_name",
        "target_csv",
        "ablation_label",
        "control_type",
        "nodes",
        "anchor_nodes",
    ]
    columns = preferred + [column for column in columns if column not in preferred]
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
