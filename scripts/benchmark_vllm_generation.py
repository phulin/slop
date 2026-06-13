from __future__ import annotations

import argparse
import csv
import json
import os
import random
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from slop_sftdiv.features import extract_tier1_features
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


DEFAULT_FEATURES = {
    "contrastive_negation",
    "rule_of_three_approx",
    "slop_lexicon",
    "stock_openers",
    "stock_closers",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM for Phase 2 free-running generation."
    )
    parser.add_argument("--model", required=True, help="Hugging Face causal LM ID or local path.")
    parser.add_argument("--input", type=Path, required=True, help="Prompt-package JSONL.")
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--temperature", action="append", type=float, default=[])
    parser.add_argument("--top-p", action="append", type=float, default=[])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--completions-per-prompt", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--generations-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-generation-backend")
    parser.add_argument("--wandb-job-type", default="vllm-generation-benchmark")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _read_prompts(path: Path, *, sample_size: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if prompt:
                rows.append(row)
    if sample_size <= 0 or sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def _feature_counts(text: str, selected_features: set[str]) -> dict[str, int]:
    counts = extract_tier1_features(text).counts
    return {feature: counts.get(feature, 0) for feature in sorted(selected_features)}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    selected_features = set(args.feature or sorted(DEFAULT_FEATURES))
    temperatures = args.temperature or [0.0, 0.7]
    top_ps = args.top_p or [0.95]
    prompt_rows = _read_prompts(args.input, sample_size=args.sample_size, seed=args.seed)
    prompts = [row["prompt"] for row in prompt_rows]
    if not prompts:
        raise ValueError(f"No prompt rows found in {args.input}")

    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage2", "phase2", "free-running", "vllm", "benchmark", *args.wandb_tag],
        config={
            "model": args.model,
            "input": str(args.input),
            "sample_size": args.sample_size,
            "seed": args.seed,
            "features": sorted(selected_features),
            "temperatures": temperatures,
            "top_ps": top_ps,
            "max_new_tokens": args.max_new_tokens,
            "completions_per_prompt": args.completions_per_prompt,
            "max_model_len": args.max_model_len,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "backend": "vllm",
            "env_vllm_use_flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
        },
    )

    try:
        started_at = time.perf_counter()
        load_started_at = time.perf_counter()
        llm = LLM(
            model=args.model,
            dtype=args.dtype,
            seed=args.seed,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
        )
        load_seconds = time.perf_counter() - load_started_at

        generation_rows: list[dict[str, Any]] = []
        summary_counts: dict[tuple[float, float], Counter[str]] = defaultdict(Counter)
        summary_repeated_counts: dict[tuple[float, float], Counter[str]] = defaultdict(Counter)
        summary_generations_with_feature: dict[tuple[float, float], Counter[str]] = defaultdict(
            Counter
        )
        summary_repeat_generations: dict[tuple[float, float], Counter[str]] = defaultdict(Counter)
        summary_tokens: Counter[tuple[float, float]] = Counter()
        summary_generations: Counter[tuple[float, float]] = Counter()

        decode_started_at = time.perf_counter()
        for temperature in temperatures:
            for top_p in top_ps:
                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=args.max_new_tokens,
                    n=args.completions_per_prompt,
                    seed=args.seed,
                )
                outputs = llm.generate(prompts, sampling_params)
                for prompt_row, output in zip(prompt_rows, outputs, strict=True):
                    for completion_index, completion in enumerate(output.outputs):
                        generation = completion.text
                        token_ids = getattr(completion, "token_ids", None) or []
                        generated_tokens = len(token_ids)
                        counts = _feature_counts(generation, selected_features)
                        repeated_counts = {
                            feature: max(0, count - 1) for feature, count in counts.items()
                        }
                        key = (temperature, top_p)
                        summary_counts[key].update(counts)
                        summary_repeated_counts[key].update(repeated_counts)
                        for feature, count in counts.items():
                            if count > 0:
                                summary_generations_with_feature[key][feature] += 1
                            if count > 1:
                                summary_repeat_generations[key][feature] += 1
                        summary_tokens[key] += max(1, generated_tokens)
                        summary_generations[key] += 1
                        generation_rows.append(
                            {
                                "source": prompt_row.get("source"),
                                "prompt_id": prompt_row.get("phase2_prompt_id")
                                or prompt_row.get("prompt_id")
                                or prompt_row.get("doc_id"),
                                "source_row_index": prompt_row.get("source_row_index"),
                                "completion_index": completion_index,
                                "temperature": temperature,
                                "top_p": top_p,
                                "seed": args.seed,
                                "generated_tokens": generated_tokens,
                                "generation_chars": len(generation),
                                "features_json": json.dumps(counts, sort_keys=True),
                                "repeated_features_json": json.dumps(
                                    repeated_counts, sort_keys=True
                                ),
                                "feature_count_total": sum(counts.values()),
                                "repeated_feature_count_total": sum(repeated_counts.values()),
                                "finish_reason": getattr(completion, "finish_reason", None),
                                "generation": generation,
                            }
                        )

        decode_seconds = time.perf_counter() - decode_started_at
        wall_seconds = time.perf_counter() - started_at
        _write_jsonl(args.generations_output, generation_rows)

        summary_rows = []
        for (temperature, top_p), counts in sorted(summary_counts.items()):
            key = (temperature, top_p)
            tokens = max(1, summary_tokens[key])
            generations = max(1, summary_generations[key])
            for feature in sorted(selected_features):
                count = counts.get(feature, 0)
                repeated_count = summary_repeated_counts[key].get(feature, 0)
                generations_with_feature = summary_generations_with_feature[key].get(feature, 0)
                repeat_generations = summary_repeat_generations[key].get(feature, 0)
                summary_rows.append(
                    {
                        "backend": "vllm",
                        "model": args.model,
                        "temperature": temperature,
                        "top_p": top_p,
                        "feature": feature,
                        "count": count,
                        "repeated_count": repeated_count,
                        "generations": generations,
                        "generations_with_feature": generations_with_feature,
                        "repeat_generations": repeat_generations,
                        "generated_tokens": tokens,
                        "per_1k_tokens": 1000.0 * count / tokens,
                        "per_generation": count / generations,
                        "repeat_per_generation": repeated_count / generations,
                        "repeat_share_after_first": repeated_count / count if count > 0 else 0.0,
                        "share_generations_with_feature": generations_with_feature / generations,
                        "share_repeat_generations": repeat_generations / generations,
                        "load_seconds": load_seconds,
                        "decode_seconds": decode_seconds,
                        "wall_seconds": wall_seconds,
                        "decode_tokens_per_sec": sum(summary_tokens.values()) / decode_seconds
                        if decode_seconds > 0
                        else 0.0,
                        "wall_tokens_per_sec": sum(summary_tokens.values()) / wall_seconds
                        if wall_seconds > 0
                        else 0.0,
                        "peak_rss_mb": _peak_rss_mb(),
                    }
                )
        _write_csv(
            args.summary_output,
            summary_rows,
            [
                "backend",
                "model",
                "temperature",
                "top_p",
                "feature",
                "count",
                "repeated_count",
                "generations",
                "generations_with_feature",
                "repeat_generations",
                "generated_tokens",
                "per_1k_tokens",
                "per_generation",
                "repeat_per_generation",
                "repeat_share_after_first",
                "share_generations_with_feature",
                "share_repeat_generations",
                "load_seconds",
                "decode_seconds",
                "wall_seconds",
                "decode_tokens_per_sec",
                "wall_tokens_per_sec",
                "peak_rss_mb",
            ],
        )

        total_tokens = sum(row["generated_tokens"] for row in generation_rows)
        run.log(
            {
                "generation/prompts": len(prompts),
                "generation/generations": len(generation_rows),
                "generation/generated_tokens": total_tokens,
                "generation/load_seconds": load_seconds,
                "generation/decode_seconds": decode_seconds,
                "generation/wall_seconds": wall_seconds,
                "generation/decode_tokens_per_sec": total_tokens / decode_seconds
                if decode_seconds > 0
                else 0.0,
                "generation/wall_tokens_per_sec": total_tokens / wall_seconds
                if wall_seconds > 0
                else 0.0,
                "generation/peak_rss_mb": _peak_rss_mb(),
            }
        )
        log_summary_table(run, "generation_feature_rates", summary_rows)
        log_summary_table(
            run,
            "generation_samples_redacted",
            [
                {key: value for key, value in row.items() if key != "generation"}
                for row in generation_rows[:100]
            ],
        )
        return summary_rows
    finally:
        run.finish()


def main() -> None:
    args = build_parser().parse_args()
    summary_rows = run_benchmark(args)
    print(f"Wrote {len(summary_rows)} feature-rate rows to {args.summary_output}")


if __name__ == "__main__":
    main()
