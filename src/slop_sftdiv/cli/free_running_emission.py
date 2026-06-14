from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from slop_sftdiv.cli.census import _id_fields_with_pair_id, _source_from_input
from slop_sftdiv.features import extract_tier1_features
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


REVISED_CORE_FEATURES = {
    "contrastive_negation",
    "rule_of_three_approx",
    "slop_lexicon",
    "stock_openers",
    "stock_closers",
    "stock_openers_closers",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 2 free-running emission measurement.")
    parser.add_argument("--model", required=True, help="Hugging Face causal LM ID or local path.")
    parser.add_argument("--input", action="append", required=True, help="Local JSONL file or HF dataset id.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--max-scan", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--temperature", action="append", type=float, default=[])
    parser.add_argument("--top-p", action="append", type=float, default=[])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--completions-per-prompt", type=int, default=1)
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=4,
        help="Number of prompt completions to decode per model.generate call.",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--fallback-text-as-prompt", action="store_true")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generations-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase2-generation")
    parser.add_argument("--wandb-job-type", default="free-running-smoke")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _load_components():
    from slop_sftdiv.data import SamplingConfig, iter_corpus_records
    from slop_sftdiv.data.corpora import DEFAULT_ID_FIELDS

    return SamplingConfig, iter_corpus_records, DEFAULT_ID_FIELDS


def _ensure_cuda_library_path() -> None:
    candidates = (
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/targets/x86_64-linux/lib",
    )
    existing = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(":") if item]
    additions = [
        item
        for item in candidates
        if Path(item, "libcuda.so.1").exists() and item not in existing
    ]
    if additions:
        os.environ["LD_LIBRARY_PATH"] = ":".join([*additions, *existing])


def _load_model(*, model_name: str, dtype_name: str, device_name: str, compile_model: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _dtype(dtype_name, torch)
    device = _device(device_name, torch)
    if device.type == "cuda":
        _ensure_cuda_library_path()
        torch.set_float32_matmul_precision("high")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if (
        getattr(tokenizer, "pad_token_id", None) is None
        and getattr(tokenizer, "eos_token_id", None) is not None
    ):
        setattr(tokenizer, "pad_token", getattr(tokenizer, "eos_token"))
    if getattr(tokenizer, "padding_side", None) != "left":
        setattr(tokenizer, "padding_side", "left")
    model_kwargs: dict[str, Any] = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(device)
    model.eval()
    if compile_model:
        model.forward = torch.compile(model.forward)
    return tokenizer, model, device


def _dtype(dtype_name: str, torch_module: Any):
    if dtype_name == "auto":
        return None
    return {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }[dtype_name]


def _device(device_name: str, torch_module: Any):
    if device_name != "auto":
        return torch_module.device(device_name)
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _selected_features(features: list[str]) -> set[str]:
    return set(features or sorted(REVISED_CORE_FEATURES))


def _feature_counts(text: str, selected_features: set[str]) -> dict[str, int]:
    counts = extract_tier1_features(text).counts
    return {feature: counts.get(feature, 0) for feature in sorted(selected_features)}


def _prompt_for_record(record: Any, *, fallback_text_as_prompt: bool) -> str | None:
    if record.prompt:
        return record.prompt
    if fallback_text_as_prompt:
        return record.text
    return None


def _fallback_prompt_token_id(tokenizer: Any) -> int:
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            return int(token_id)
    raise ValueError("tokenizer has no BOS/EOS/PAD token for empty prompt generation")


def _pad_batch(
    tokenized_prompts: list[list[int]],
    *,
    pad_token_id: int,
    device: Any,
) -> tuple[Any, Any]:
    import torch

    width = max(len(ids) for ids in tokenized_prompts)
    rows = []
    masks = []
    for ids in tokenized_prompts:
        pad_width = width - len(ids)
        rows.append([pad_token_id] * pad_width + ids)
        masks.append([0] * pad_width + [1] * len(ids))
    return (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
    )


def _prompt_ids(tokenizer: Any, prompt: str, *, max_prompt_tokens: int) -> list[int]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)[-max_prompt_tokens:]
    if ids:
        return ids
    return [_fallback_prompt_token_id(tokenizer)]


def _generate_batch(
    *,
    tokenizer: Any,
    model: Any,
    device: Any,
    prompts: list[str],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    max_prompt_tokens: int,
    seed: int,
) -> list[tuple[str, int, int]]:
    import torch

    if not prompts:
        return []
    tokenized_prompts = [
        _prompt_ids(tokenizer, prompt, max_prompt_tokens=max_prompt_tokens) for prompt in prompts
    ]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    input_ids, attention_mask = _pad_batch(
        tokenized_prompts,
        pad_token_id=pad_token_id,
        device=device,
    )
    torch.manual_seed(seed)
    if getattr(torch, "cuda").is_available():
        torch.cuda.manual_seed_all(seed)
    do_sample = temperature > 0
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    with torch.inference_mode():
        output = model.generate(**generate_kwargs)
    generated_start = input_ids.shape[-1]
    rows = []
    for index, prompt_ids in enumerate(tokenized_prompts):
        generated_ids = output[index, generated_start:]
        rows.append(
            (
                tokenizer.decode(generated_ids, skip_special_tokens=True),
                len(prompt_ids),
                int(generated_ids.shape[-1]),
            )
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_free_running_emission(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.generation_batch_size <= 0:
        raise ValueError("--generation-batch-size must be positive")
    if args.completions_per_prompt <= 0:
        raise ValueError("--completions-per-prompt must be positive")
    SamplingConfig, iter_corpus_records, DEFAULT_ID_FIELDS = _load_components()
    tokenizer, model, device = _load_model(
        model_name=args.model,
        dtype_name=args.dtype,
        device_name=args.device,
        compile_model=args.torch_compile,
    )
    temperatures = args.temperature or [0.0]
    top_ps = args.top_p or [0.95]
    selected_features = _selected_features(args.feature)
    sampling = SamplingConfig(cap=args.sample_size, seed=args.seed, max_scan=args.max_scan)
    id_fields = _id_fields_with_pair_id(DEFAULT_ID_FIELDS)
    generation_rows: list[dict[str, Any]] = []
    summary_counts: dict[tuple[float, float, str], Counter[str]] = defaultdict(Counter)
    summary_repeated_counts: dict[tuple[float, float, str], Counter[str]] = defaultdict(Counter)
    summary_generations_with_feature: dict[tuple[float, float, str], Counter[str]] = defaultdict(
        Counter
    )
    summary_repeat_generations: dict[tuple[float, float, str], Counter[str]] = defaultdict(Counter)
    summary_tokens: Counter[tuple[float, float, str]] = Counter()
    summary_generations: Counter[tuple[float, float, str]] = Counter()
    prompts_seen = 0
    prompts_skipped = 0
    started_at = time.perf_counter()
    args.generations_output.parent.mkdir(parents=True, exist_ok=True)
    args.generations_output.write_text("", encoding="utf-8")

    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage2", "phase2", "free-running", "smoke", *args.wandb_tag],
        config={
            "model": args.model,
            "inputs": args.input,
            "split": args.split,
            "hf_config": args.hf_config,
            "sample_size": args.sample_size,
            "max_scan": args.max_scan,
            "features": sorted(selected_features),
            "temperatures": temperatures,
            "top_ps": top_ps,
            "max_new_tokens": args.max_new_tokens,
            "completions_per_prompt": args.completions_per_prompt,
            "generation_batch_size": args.generation_batch_size,
            "max_prompt_tokens": args.max_prompt_tokens,
            "fallback_text_as_prompt": args.fallback_text_as_prompt,
            "dtype": args.dtype,
            "device": str(device),
            "torch_compile": args.torch_compile,
        },
    )

    try:
        generation_batch_size = args.generation_batch_size
        for raw_source in args.input:
            source = _source_from_input(raw_source, split=args.split, hf_config=args.hf_config)
            records = iter_corpus_records(source, sampling=sampling, id_fields=id_fields)
            pending: dict[tuple[float, float, int], list[tuple[Any, str]]] = defaultdict(list)

            def flush_batch(batch_key: tuple[float, float, int]) -> None:
                batch = pending[batch_key]
                if not batch:
                    return
                temperature, top_p, completion_index = batch_key
                generated = _generate_batch(
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    prompts=[prompt for _record, prompt in batch],
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=args.max_new_tokens,
                    max_prompt_tokens=args.max_prompt_tokens,
                    seed=args.seed + completion_index,
                )
                for (record, _prompt), (generation, prompt_tokens, generated_tokens) in zip(
                    batch, generated, strict=True
                ):
                    counts = _feature_counts(generation, selected_features)
                    repeated_counts = {
                        feature: max(0, count - 1)
                        for feature, count in counts.items()
                    }
                    key = (temperature, top_p, record.source)
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
                            "source": record.source,
                            "record_id": record.record_id,
                            "source_kind": record.source_kind,
                            "split": record.split,
                            "row_index": record.row_index,
                            "completion_index": completion_index,
                            "temperature": temperature,
                            "top_p": top_p,
                            "seed": args.seed + completion_index,
                            "prompt_tokens": prompt_tokens,
                            "generated_tokens": generated_tokens,
                            "generation_chars": len(generation),
                            "features_json": json.dumps(counts, sort_keys=True),
                            "repeated_features_json": json.dumps(repeated_counts, sort_keys=True),
                            "feature_count_total": sum(counts.values()),
                            "repeated_feature_count_total": sum(repeated_counts.values()),
                            "generation": generation,
                        }
                    )
                _append_jsonl(args.generations_output, generation_rows[-len(batch) :])
                batch.clear()

            for record in tqdm(records, desc=f"free-run:{source.name}", unit="prompt"):
                prompts_seen += 1
                prompt = _prompt_for_record(record, fallback_text_as_prompt=args.fallback_text_as_prompt)
                if not prompt:
                    prompts_skipped += 1
                    continue
                for temperature in temperatures:
                    for top_p in top_ps:
                        for completion_index in range(args.completions_per_prompt):
                            batch_key = (temperature, top_p, completion_index)
                            pending[batch_key].append((record, prompt))
                            if len(pending[batch_key]) >= generation_batch_size:
                                flush_batch(batch_key)
            for batch_key in list(pending):
                flush_batch(batch_key)

        summary_rows = []
        for (temperature, top_p, source), counts in sorted(summary_counts.items()):
            summary_key = (temperature, top_p, source)
            tokens = max(1, summary_tokens[summary_key])
            generations = max(1, summary_generations[summary_key])
            for feature in sorted(selected_features):
                count = counts.get(feature, 0)
                repeated_count = summary_repeated_counts[summary_key].get(feature, 0)
                generations_with_feature = summary_generations_with_feature[summary_key].get(
                    feature, 0
                )
                repeat_generations = summary_repeat_generations[summary_key].get(feature, 0)
                summary_rows.append(
                    {
                        "source": source,
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
                    }
                )
        _write_csv(
            args.summary_output,
            summary_rows,
            [
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
            ],
        )
        wall_seconds = time.perf_counter() - started_at
        run.log(
            {
                "generation/prompts_seen": prompts_seen,
                "generation/prompts_skipped": prompts_skipped,
                "generation/generations": len(generation_rows),
                "generation/generated_tokens": sum(row["generated_tokens"] for row in generation_rows),
                "generation/wall_seconds": wall_seconds,
                "generation/tokens_per_sec": sum(row["generated_tokens"] for row in generation_rows)
                / wall_seconds
                if wall_seconds > 0
                else 0.0,
                "generation/peak_rss_mb": _peak_rss_mb(),
            }
        )
        wandb_rows = [
            {key: value for key, value in row.items() if key != "generation"}
            for row in generation_rows[:100]
        ]
        log_summary_table(run, "generation_feature_rates", summary_rows)
        log_summary_table(run, "generation_samples_redacted", wandb_rows)
        return summary_rows
    finally:
        run.finish()


def main() -> None:
    args = build_parser().parse_args()
    summary_rows = run_free_running_emission(args)
    print(f"Wrote {len(summary_rows)} feature-rate rows to {args.summary_output}")


if __name__ == "__main__":
    main()
