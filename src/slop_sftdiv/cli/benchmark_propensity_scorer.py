from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from slop_sftdiv.cli.teacher_forced_propensity import (
    _initiator_token_sequences,
    _load_model,
    _sequence_prob_mass,
    _sequence_prob_mass_batch,
    _sequence_prob_mass_batch_cached,
    _sequence_prob_mass_batch_cached_multi,
)
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Phase 2 propensity scoring kernels.")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--feature", default="slop_lexicon")
    parser.add_argument(
        "--multi-feature",
        action="append",
        default=[],
        help="Additional features to score from the same cached prefix batch.",
    )
    parser.add_argument("--batch-size", action="append", type=int, default=[])
    parser.add_argument("--prefix-tokens", type=int, default=64)
    parser.add_argument("--cache-branch-batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-benchmarks")
    parser.add_argument("--wandb-job-type", default="propensity-scorer-benchmark")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _synchronize(device: Any) -> None:
    if getattr(device, "type", None) != "cuda":
        return
    import torch

    torch.cuda.synchronize(device)


def _model_inputs(*, batch_size: int, prefix_tokens: int, tokenizer: Any, device: Any, seed: int) -> dict[str, Any]:
    import torch

    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_ids = torch.randint(
        low=0,
        high=max(2, vocab_size),
        size=(batch_size, prefix_tokens),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "prefix_tokens": [prefix_tokens] * batch_size,
    }


def _time_call(fn: Any, *, repeats: int, warmup: int, device: Any) -> tuple[float, Any]:
    result = None
    for _index in range(warmup):
        result = fn()
    _synchronize(device)
    started_at = time.perf_counter()
    for _index in range(repeats):
        result = fn()
    _synchronize(device)
    elapsed = time.perf_counter() - started_at
    return elapsed / max(1, repeats), result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.prefix_tokens <= 0:
        raise ValueError("prefix_tokens must be positive")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    batch_sizes = args.batch_size or [1, 4, 16]
    if any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("batch_size values must be positive")
    tokenizer, model, device = _load_model(
        model_name=args.model,
        dtype_name=args.dtype,
        device_name=args.device,
        compile_model=args.torch_compile,
    )
    sequences = _initiator_token_sequences(tokenizer, args.feature)
    multi_features = list(dict.fromkeys([args.feature, *args.multi_feature]))
    multi_sequences = {
        feature: _initiator_token_sequences(tokenizer, feature) for feature in multi_features
    }
    rows: list[dict[str, Any]] = []
    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage2", "phase2", "benchmark", *args.wandb_tag],
        config={
            "model": args.model,
            "feature": args.feature,
            "multi_features": multi_features,
            "batch_sizes": batch_sizes,
            "prefix_tokens": args.prefix_tokens,
            "cache_branch_batch_size": args.cache_branch_batch_size,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "dtype": args.dtype,
            "device": str(device),
            "torch_compile": args.torch_compile,
            "initiator_sequences": len(sequences),
            "multi_initiator_sequences": {
                feature: len(feature_sequences)
                for feature, feature_sequences in multi_sequences.items()
            },
        },
    )
    try:
        for batch_size in batch_sizes:
            model_inputs = _model_inputs(
                batch_size=batch_size,
                prefix_tokens=args.prefix_tokens,
                tokenizer=tokenizer,
                device=device,
                seed=args.seed + batch_size,
            )

            def scalar_call() -> list[float]:
                return [
                    _sequence_prob_mass(
                        model,
                        {
                            "input_ids": model_inputs["input_ids"][index : index + 1],
                            "attention_mask": model_inputs["attention_mask"][index : index + 1],
                            "prefix_tokens": 1,
                        },
                        sequences,
                    )
                    for index in range(batch_size)
                ]

            def batched_call() -> list[float]:
                return _sequence_prob_mass_batch(model, model_inputs, sequences)

            def cached_call() -> list[float]:
                return _sequence_prob_mass_batch_cached(
                    model,
                    model_inputs,
                    sequences,
                    cache_branch_batch_size=args.cache_branch_batch_size,
                )

            def cached_multi_call() -> dict[str, list[float]]:
                return _sequence_prob_mass_batch_cached_multi(
                    model,
                    model_inputs,
                    multi_sequences,
                    cache_branch_batch_size=args.cache_branch_batch_size,
                )

            scalar_seconds, scalar_result = _time_call(
                scalar_call,
                repeats=args.repeats,
                warmup=args.warmup,
                device=device,
            )
            batched_seconds, batched_result = _time_call(
                batched_call,
                repeats=args.repeats,
                warmup=args.warmup,
                device=device,
            )
            cached_seconds, cached_result = _time_call(
                cached_call,
                repeats=args.repeats,
                warmup=args.warmup,
                device=device,
            )
            cached_multi_seconds, cached_multi_result = _time_call(
                cached_multi_call,
                repeats=args.repeats,
                warmup=args.warmup,
                device=device,
            )
            max_abs_diff = max(
                abs(left - right) for left, right in zip(scalar_result, batched_result)
            )
            cached_max_abs_diff = max(
                abs(left - right) for left, right in zip(scalar_result, cached_result)
            )
            cached_multi_max_abs_diff = max(
                abs(left - right)
                for left, right in zip(scalar_result, cached_multi_result[args.feature])
            )
            rows.extend(
                [
                    {
                        "batch_size": batch_size,
                        "mode": "scalar",
                        "features_scored": 1,
                        "seconds_per_repeat": scalar_seconds,
                        "opportunities_per_sec": batch_size / scalar_seconds
                        if scalar_seconds > 0
                        else 0.0,
                        "max_abs_diff_vs_scalar": 0.0,
                    },
                    {
                        "batch_size": batch_size,
                        "mode": "batched",
                        "features_scored": 1,
                        "seconds_per_repeat": batched_seconds,
                        "opportunities_per_sec": batch_size / batched_seconds
                        if batched_seconds > 0
                        else 0.0,
                        "max_abs_diff_vs_scalar": max_abs_diff,
                    },
                    {
                        "batch_size": batch_size,
                        "mode": "batched_cached",
                        "features_scored": 1,
                        "seconds_per_repeat": cached_seconds,
                        "opportunities_per_sec": batch_size / cached_seconds
                        if cached_seconds > 0
                        else 0.0,
                        "max_abs_diff_vs_scalar": cached_max_abs_diff,
                    },
                    {
                        "batch_size": batch_size,
                        "mode": "batched_cached_multi_feature",
                        "features_scored": len(multi_features),
                        "seconds_per_repeat": cached_multi_seconds,
                        "opportunities_per_sec": (batch_size * len(multi_features))
                        / cached_multi_seconds
                        if cached_multi_seconds > 0
                        else 0.0,
                        "prefixes_per_sec": batch_size / cached_multi_seconds
                        if cached_multi_seconds > 0
                        else 0.0,
                        "max_abs_diff_vs_scalar": cached_multi_max_abs_diff,
                    },
                ]
            )
        _write_csv(args.output, rows)
        if args.summary_output is not None:
            args.summary_output.parent.mkdir(parents=True, exist_ok=True)
            args.summary_output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log_summary_table(run, "propensity_scorer_benchmark", rows)
        return rows
    finally:
        run.finish()


def main() -> None:
    args = build_parser().parse_args()
    rows = run_benchmark(args)
    print(f"Wrote {len(rows)} benchmark rows to {args.output}")


if __name__ == "__main__":
    main()
