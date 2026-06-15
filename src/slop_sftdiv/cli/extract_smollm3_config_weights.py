from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIGS = (
    "stage3_9T_11T.yaml",
    "long_context_4k_to_32k.yaml",
    "long_context_32k_to_64.yaml",
)


@dataclass(frozen=True)
class SourceRow:
    config_file: str
    data_stage_index: int
    data_stage_name: str
    start_training_step: int
    end_training_step: int
    segment_steps: int
    tokens_per_step: int
    segment_tokens: int
    source_name: str
    source_group: str
    dataset_read_path: str
    dataset_folder: str
    source_weight_raw: float
    source_weight_normalized: float
    source_tokens_in_segment: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract token-weighted SmolLM3 source weights from published Nanotron YAML "
            "training configs."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        type=Path,
        default=[],
        help="Local YAML config path. Repeat for multiple configs.",
    )
    parser.add_argument(
        "--hf-repo",
        default=None,
        help="Optional Hugging Face dataset repo to download configs from.",
    )
    parser.add_argument(
        "--hf-file",
        action="append",
        default=[],
        help="Config filename within --hf-repo. Defaults to the SmolLM3 retained configs.",
    )
    parser.add_argument(
        "--detail-output",
        type=Path,
        default=Path("artifacts/phase3/analysis/smollm3_config_source_weights_detail.csv"),
    )
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        default=Path("artifacts/phase3/analysis/smollm3_config_source_weights_aggregate.csv"),
    )
    parser.add_argument(
        "--group-output",
        type=Path,
        default=Path("artifacts/phase3/analysis/smollm3_config_source_weights_groups.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("artifacts/phase3/analysis/smollm3_config_source_weights_summary.md"),
    )
    return parser


def _source_name(dataset_read_path: str) -> str:
    return dataset_read_path.rstrip("/").split("/")[-1]


def _source_group(source_name: str) -> str:
    name = source_name.lower()
    if "stackexchange" in name:
        return "qa_forum"
    if "stack-edu" in name or any(
        token in name
        for token in (
            "pull-requests",
            "kaggle",
            "jupyter",
            "github-issues",
            "opencode",
            "code_sft",
        )
    ):
        return "code"
    if any(token in name for token in ("math", "gsm", "megamath", "infiwebmath")):
        return "math"
    if any(token in name for token in ("reasoning", "problem-solving", "2students")):
        return "reasoning"
    if "wiki" in name:
        return "wiki"
    if any(token in name for token in ("pes2o", "cosmopedia")):
        return "academic_or_synthetic_web"
    if any(token in name for token in ("fineweb", "dclm", "fw2", "dolmino")):
        return "web"
    return "other"


def _tokens_per_step(config: dict[str, Any]) -> int:
    tokens = config["tokens"]
    parallelism = config["parallelism"]
    return int(
        parallelism["dp"]
        * tokens["batch_accumulation_per_replica"]
        * tokens["micro_batch_size"]
        * tokens["sequence_length"]
    )


def _stage_rows(config_path: Path) -> list[SourceRow]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_steps = int(config["tokens"]["train_steps"])
    tokens_per_step = _tokens_per_step(config)
    stages = sorted(
        enumerate(config["data_stages"]),
        key=lambda item: int(item[1].get("start_training_step", 1)),
    )
    rows: list[SourceRow] = []
    for position, (original_index, stage) in enumerate(stages):
        start = int(stage.get("start_training_step", 1))
        next_start = (
            int(stages[position + 1][1].get("start_training_step", train_steps + 1))
            if position + 1 < len(stages)
            else train_steps + 1
        )
        end = min(train_steps, next_start - 1)
        segment_steps = max(0, end - start + 1)
        if segment_steps == 0:
            continue
        dataset = stage["data"]["dataset"]
        read_paths = list(dataset["dataset_read_path"])
        folders = list(dataset["dataset_folder"])
        raw_weights = [float(weight) for weight in dataset["dataset_weights"]]
        if not (len(read_paths) == len(folders) == len(raw_weights)):
            raise ValueError(
                f"{config_path}: dataset_read_path, dataset_folder, and dataset_weights "
                "must have the same length"
            )
        weight_sum = sum(raw_weights)
        if weight_sum <= 0:
            raise ValueError(f"{config_path}: source weights sum to zero")
        segment_tokens = segment_steps * tokens_per_step
        for read_path, folder, raw_weight in zip(read_paths, folders, raw_weights, strict=True):
            normalized_weight = raw_weight / weight_sum
            source_name = _source_name(str(read_path))
            rows.append(
                SourceRow(
                    config_file=config_path.name,
                    data_stage_index=int(original_index),
                    data_stage_name=str(stage.get("name", f"stage_{original_index}")),
                    start_training_step=start,
                    end_training_step=end,
                    segment_steps=segment_steps,
                    tokens_per_step=tokens_per_step,
                    segment_tokens=segment_tokens,
                    source_name=source_name,
                    source_group=_source_group(source_name),
                    dataset_read_path=str(read_path),
                    dataset_folder=str(folder),
                    source_weight_raw=raw_weight,
                    source_weight_normalized=normalized_weight,
                    source_tokens_in_segment=segment_tokens * normalized_weight,
                )
            )
    return rows


def _resolve_configs(args: argparse.Namespace) -> list[Path]:
    paths = list(args.config)
    if args.hf_repo:
        from huggingface_hub import hf_hub_download

        filenames = args.hf_file or list(DEFAULT_CONFIGS)
        for filename in filenames:
            paths.append(Path(hf_hub_download(args.hf_repo, filename, repo_type="dataset")))
    if not paths:
        raise ValueError("Provide --config or --hf-repo.")
    return paths


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[SourceRow], key: str) -> list[dict[str, Any]]:
    total_tokens = sum(row.source_tokens_in_segment for row in rows)
    bucket_tokens: dict[str, float] = defaultdict(float)
    bucket_segments: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = getattr(row, key)
        bucket_tokens[value] += row.source_tokens_in_segment
        bucket_segments[value].add(f"{row.config_file}:{row.data_stage_name}")
    out = []
    for value, tokens in sorted(bucket_tokens.items(), key=lambda item: (-item[1], item[0])):
        out.append(
            {
                key: value,
                "tokens": f"{tokens:.6f}",
                "share": f"{tokens / total_tokens:.12f}" if total_tokens else "",
                "share_percent": f"{tokens / total_tokens * 100:.6f}" if total_tokens else "",
                "segments": "; ".join(sorted(bucket_segments[value])),
            }
        )
    return out


def _summary(
    rows: list[SourceRow],
    aggregate_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    total_tokens = sum(row.source_tokens_in_segment for row in rows)
    segments: dict[tuple[str, str], SourceRow] = {}
    for row in rows:
        segments.setdefault((row.config_file, row.data_stage_name), row)
    lines = [
        "# SmolLM3 Config Source Weights",
        "",
        "Source: published SmolLM3 Nanotron training configs.",
        "",
        f"Detail rows: {len(rows)}",
        f"Aggregate sources: {len(aggregate_rows)}",
        f"Total config-implied tokens: {total_tokens / 1_000_000_000_000:.6f}T",
        "",
        "## Segments",
        "",
        "| Config | Data stage | Steps | Tokens / step | Segment tokens |",
        "|---|---|---:|---:|---:|",
    ]
    for (config_file, stage_name), row in sorted(segments.items()):
        lines.append(
            "| {config} | {stage} | {steps:,} | {tps:,} | {tokens:.3f}B |".format(
                config=config_file,
                stage=stage_name,
                steps=row.segment_steps,
                tps=row.tokens_per_step,
                tokens=row.segment_tokens / 1_000_000_000,
            )
        )
    lines.extend(["", "## Top Aggregate Sources", ""])
    lines.extend(["| Source | Share | Tokens |", "|---|---:|---:|"])
    for row in aggregate_rows[:20]:
        lines.append(
            f"| `{row['source_name']}` | {float(row['share_percent']):.3f}% | "
            f"{float(row['tokens']) / 1_000_000_000:.3f}B |"
        )
    lines.extend(["", "## Source Groups", ""])
    lines.extend(["| Group | Share | Tokens |", "|---|---:|---:|"])
    for row in group_rows:
        lines.append(
            f"| `{row['source_group']}` | {float(row['share_percent']):.3f}% | "
            f"{float(row['tokens']) / 1_000_000_000:.3f}B |"
        )
    lines.extend(
        [
            "",
            "## Feature-Rate Coverage",
            "",
            "This extractor reports recipe source weights only. Current sampled "
            "feature-rate coverage is assembled by "
            "`slop-assemble-weighted-pretrain-baseline`, which joins these weights "
            "to explicit source maps for local corpus censuses.",
        ]
    )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- These are recipe source weights from config geometry and normalized "
            "`dataset_weights`; they are not feature rates.",
            "- A production weighted pretraining feature baseline still requires "
            "feature rates for all relevant sources or a documented source-group proxy.",
            "- Source groups are heuristic labels added by this extractor for audit "
            "readability; source-level rows are the authoritative extraction.",
            "- Long-context configs are included only when passed to this command. "
            "The 4k pretraining mix is recoverable from `stage3_9T_11T.yaml` alone.",
            "",
            "## Outputs",
            "",
            f"- `{args.detail_output}`",
            f"- `{args.aggregate_output}`",
            f"- `{args.group_output}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> list[SourceRow]:
    rows: list[SourceRow] = []
    for config_path in _resolve_configs(args):
        rows.extend(_stage_rows(config_path))
    detail_rows = [row.__dict__ for row in rows]
    aggregate_rows = _aggregate(rows, "source_name")
    group_rows = _aggregate(rows, "source_group")
    detail_fields = list(SourceRow.__dataclass_fields__)
    _write_csv(args.detail_output, detail_rows, detail_fields)
    _write_csv(
        args.aggregate_output,
        aggregate_rows,
        ["source_name", "tokens", "share", "share_percent", "segments"],
    )
    _write_csv(
        args.group_output,
        group_rows,
        ["source_group", "tokens", "share", "share_percent", "segments"],
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        _summary(rows, aggregate_rows, group_rows, args),
        encoding="utf-8",
    )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
