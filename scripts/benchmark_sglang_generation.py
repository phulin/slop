from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import resource
import sys
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
    "stock_openers_closers",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang for Phase 2 free-running generation."
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
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument(
        "--ignore-eos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ask SGLang to ignore EOS and tokenizer additional stop tokens.",
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-level", default="error")
    parser.add_argument("--generations-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-generation-backend")
    parser.add_argument("--wandb-job-type", default="sglang-generation-benchmark")
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
            if row.get("prompt"):
                rows.append(row)
    if sample_size <= 0 or sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def _source_name(path: Path) -> str:
    return path.stem


def _fallback_prompt_token_id(tokenizer: Any) -> int:
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            return int(token_id)
    raise ValueError("tokenizer has no BOS/EOS/PAD token for empty prompt generation")


def _prompt_token_ids(tokenizer: Any, prompt: str, *, max_prompt_tokens: int) -> list[int]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if max_prompt_tokens > 0:
        ids = ids[-max_prompt_tokens:]
    if ids:
        return [int(token_id) for token_id in ids]
    return [_fallback_prompt_token_id(tokenizer)]


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


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _ensure_event_loop() -> None:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def _completion_tokens_from_meta(meta_info: dict[str, Any], text: str) -> int:
    completion_tokens = meta_info.get("completion_tokens")
    if completion_tokens is not None:
        return int(completion_tokens)
    output_token_ids = (
        meta_info.get("output_token_ids")
        or meta_info.get("completion_token_ids")
        or meta_info.get("token_ids")
    )
    if output_token_ids is not None:
        return len(output_token_ids)
    # Last-resort estimate for older SGLang responses that omit token counts.
    return max(1, len(text.split()))


def _finish_reason_from_meta(meta_info: dict[str, Any]) -> str | None:
    finish_reason = meta_info.get("finish_reason") or meta_info.get("finish_reasons")
    if isinstance(finish_reason, list):
        return str(finish_reason[0]) if finish_reason else None
    return str(finish_reason) if finish_reason is not None else None


def _normalize_completion(response: dict[str, Any]) -> list[dict[str, Any]]:
    texts = _as_list(response.get("text", ""))
    meta_values = response.get("meta_info") or {}
    if isinstance(meta_values, list):
        metas = meta_values
    else:
        metas = [meta_values for _ in texts]
    completions = []
    for index, text in enumerate(texts):
        meta_info = metas[index] if index < len(metas) and isinstance(metas[index], dict) else {}
        text = str(text)
        completions.append(
            {
                "text": text,
                "generated_tokens": _completion_tokens_from_meta(meta_info, text),
                "finish_reason": _finish_reason_from_meta(meta_info),
            }
        )
    return completions


def _normalize_responses(responses: Any, *, prompt_count: int) -> list[list[dict[str, Any]]]:
    if isinstance(responses, dict):
        if prompt_count == 1:
            return [_normalize_completion(responses)]
        texts = responses.get("text")
        if isinstance(texts, list) and len(texts) == prompt_count:
            meta_info = responses.get("meta_info")
            return [
                _normalize_completion(
                    {
                        "text": text,
                        "meta_info": meta_info[index]
                        if isinstance(meta_info, list) and index < len(meta_info)
                        else meta_info,
                    }
                )
                for index, text in enumerate(texts)
            ]
    response_list = _as_list(responses)
    if len(response_list) != prompt_count:
        raise ValueError(
            f"SGLang returned {len(response_list)} prompt responses for {prompt_count} prompts"
        )
    return [_normalize_completion(response) for response in response_list]


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    import sglang as sgl
    from transformers import AutoTokenizer

    selected_features = set(args.feature or sorted(DEFAULT_FEATURES))
    temperatures = args.temperature or [0.0, 0.7]
    top_ps = args.top_p or [0.95]
    source_name = _source_name(args.input)
    prompt_rows = _read_prompts(args.input, sample_size=args.sample_size, seed=args.seed)
    raw_prompts = [row["prompt"] for row in prompt_rows]
    if not raw_prompts:
        raise ValueError(f"No prompt rows found in {args.input}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt_token_ids = [
        _prompt_token_ids(tokenizer, prompt, max_prompt_tokens=args.max_prompt_tokens)
        for prompt in raw_prompts
    ]

    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage2", "phase2", "free-running", "sglang", "benchmark", *args.wandb_tag],
        config={
            "model": args.model,
            "input": str(args.input),
            "source": source_name,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "features": sorted(selected_features),
            "temperatures": temperatures,
            "top_ps": top_ps,
            "max_new_tokens": args.max_new_tokens,
            "completions_per_prompt": args.completions_per_prompt,
            "max_prompt_tokens": args.max_prompt_tokens,
            "context_length": args.context_length,
            "dtype": args.dtype,
            "tp_size": args.tp_size,
            "mem_fraction_static": args.mem_fraction_static,
            "ignore_eos": args.ignore_eos,
            "trust_remote_code": args.trust_remote_code,
            "log_level": args.log_level,
            "backend": "sglang",
        },
    )

    engine = None
    try:
        started_at = time.perf_counter()
        load_started_at = time.perf_counter()
        engine = sgl.Engine(
            model_path=args.model,
            tp_size=args.tp_size,
            dtype=args.dtype,
            context_length=args.context_length,
            mem_fraction_static=args.mem_fraction_static,
            trust_remote_code=args.trust_remote_code,
            log_level=args.log_level,
            random_seed=args.seed,
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
                sampling_params = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "n": args.completions_per_prompt,
                    "ignore_eos": args.ignore_eos,
                }
                _ensure_event_loop()
                responses = engine.generate(
                    input_ids=prompt_token_ids,
                    sampling_params=sampling_params,
                )
                normalized_responses = _normalize_responses(
                    responses,
                    prompt_count=len(prompt_rows),
                )
                for prompt_row, input_token_ids, completions in zip(
                    prompt_rows, prompt_token_ids, normalized_responses, strict=True
                ):
                    for completion_index, completion in enumerate(completions):
                        generation = completion["text"]
                        generated_tokens = int(completion["generated_tokens"])
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
                                "backend": "sglang",
                                "model": args.model,
                                "source": source_name,
                                "prompt_id": prompt_row.get("phase2_prompt_id")
                                or prompt_row.get("prompt_id")
                                or prompt_row.get("doc_id"),
                                "source_row_index": prompt_row.get("source_row_index"),
                                "completion_index": completion_index,
                                "temperature": temperature,
                                "top_p": top_p,
                                "seed": args.seed,
                                "prompt_tokens": len(input_token_ids),
                                "generated_tokens": generated_tokens,
                                "generation_chars": len(generation),
                                "features_json": json.dumps(counts, sort_keys=True),
                                "repeated_features_json": json.dumps(
                                    repeated_counts, sort_keys=True
                                ),
                                "feature_count_total": sum(counts.values()),
                                "repeated_feature_count_total": sum(repeated_counts.values()),
                                "finish_reason": completion["finish_reason"],
                                "generation": generation,
                            }
                        )

        decode_seconds = time.perf_counter() - decode_started_at
        wall_seconds = time.perf_counter() - started_at
        _write_jsonl(args.generations_output, generation_rows)

        summary_rows = []
        total_generated_tokens = sum(summary_tokens.values())
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
                        "backend": "sglang",
                        "model": args.model,
                        "source": source_name,
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
                        "decode_tokens_per_sec": total_generated_tokens / decode_seconds
                        if decode_seconds > 0
                        else 0.0,
                        "wall_tokens_per_sec": total_generated_tokens / wall_seconds
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
                "source",
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

        run.log(
            {
                "generation/prompts": len(prompt_token_ids),
                "generation/prompts_seen": len(prompt_rows),
                "generation/prompts_skipped": 0,
                "generation/generations": len(generation_rows),
                "generation/generated_tokens": total_generated_tokens,
                "generation/load_seconds": load_seconds,
                "generation/decode_seconds": decode_seconds,
                "generation/wall_seconds": wall_seconds,
                "generation/decode_tokens_per_sec": total_generated_tokens / decode_seconds
                if decode_seconds > 0
                else 0.0,
                "generation/wall_tokens_per_sec": total_generated_tokens / wall_seconds
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
        try:
            run.finish()
        except Exception as exc:
            print(f"Warning: wandb finish failed: {exc}", file=sys.stderr)
        if engine is not None:
            shutdown = getattr(engine, "shutdown", None)
            if shutdown is not None:
                shutdown()


def main() -> None:
    args = build_parser().parse_args()
    summary_rows = run_benchmark(args)
    print(f"Wrote {len(summary_rows)} feature-rate rows to {args.summary_output}")


if __name__ == "__main__":
    main()
