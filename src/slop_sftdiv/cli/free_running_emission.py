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
    parser = argparse.ArgumentParser(description="Run Phase 2 free-running emission smoke.")
    parser.add_argument("--model", required=True, help="Hugging Face causal LM ID or local path.")
    parser.add_argument("--input", action="append", required=True, help="Local JSONL file or HF dataset id.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--max-scan", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--temperature", action="append", type=float, default=[])
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--completions-per-prompt", type=int, default=1)
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


def _generate_one(
    *,
    tokenizer: Any,
    model: Any,
    device: Any,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    max_prompt_tokens: int,
    seed: int,
) -> tuple[str, int, int]:
    import torch

    ids = tokenizer.encode(prompt, add_special_tokens=False)[-max_prompt_tokens:]
    if not ids:
        fallback_id = getattr(tokenizer, "bos_token_id", None)
        if fallback_id is None:
            fallback_id = getattr(tokenizer, "eos_token_id", None)
        if fallback_id is None:
            fallback_id = getattr(tokenizer, "pad_token_id", None)
        if fallback_id is None:
            raise ValueError("tokenizer has no BOS/EOS/PAD token for empty prompt generation")
        ids = [fallback_id]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    torch.manual_seed(seed)
    if getattr(torch, "cuda").is_available():
        torch.cuda.manual_seed_all(seed)
    do_sample = temperature > 0
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    with torch.inference_mode():
        output = model.generate(**generate_kwargs)
    generated_ids = output[0, input_ids.shape[-1] :]
    return (
        tokenizer.decode(generated_ids, skip_special_tokens=True),
        int(input_ids.shape[-1]),
        int(generated_ids.shape[-1]),
    )


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


def run_free_running_emission(args: argparse.Namespace) -> list[dict[str, Any]]:
    SamplingConfig, iter_corpus_records, DEFAULT_ID_FIELDS = _load_components()
    tokenizer, model, device = _load_model(
        model_name=args.model,
        dtype_name=args.dtype,
        device_name=args.device,
        compile_model=args.torch_compile,
    )
    temperatures = args.temperature or [0.0]
    selected_features = _selected_features(args.feature)
    sampling = SamplingConfig(cap=args.sample_size, seed=args.seed, max_scan=args.max_scan)
    id_fields = _id_fields_with_pair_id(DEFAULT_ID_FIELDS)
    generation_rows: list[dict[str, Any]] = []
    summary_counts: dict[tuple[float, str], Counter[str]] = defaultdict(Counter)
    summary_tokens: Counter[tuple[float, str]] = Counter()
    summary_generations: Counter[tuple[float, str]] = Counter()
    prompts_seen = 0
    prompts_skipped = 0
    started_at = time.perf_counter()

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
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "completions_per_prompt": args.completions_per_prompt,
            "max_prompt_tokens": args.max_prompt_tokens,
            "fallback_text_as_prompt": args.fallback_text_as_prompt,
            "dtype": args.dtype,
            "device": str(device),
            "torch_compile": args.torch_compile,
        },
    )

    try:
        for raw_source in args.input:
            source = _source_from_input(raw_source, split=args.split, hf_config=args.hf_config)
            records = iter_corpus_records(source, sampling=sampling, id_fields=id_fields)
            for record in tqdm(records, desc=f"free-run:{source.name}", unit="prompt"):
                prompts_seen += 1
                prompt = _prompt_for_record(record, fallback_text_as_prompt=args.fallback_text_as_prompt)
                if not prompt:
                    prompts_skipped += 1
                    continue
                for temperature in temperatures:
                    for completion_index in range(args.completions_per_prompt):
                        generation, prompt_tokens, generated_tokens = _generate_one(
                            tokenizer=tokenizer,
                            model=model,
                            device=device,
                            prompt=prompt,
                            temperature=temperature,
                            top_p=args.top_p,
                            max_new_tokens=args.max_new_tokens,
                            max_prompt_tokens=args.max_prompt_tokens,
                            seed=args.seed + completion_index,
                        )
                        counts = _feature_counts(generation, selected_features)
                        key = (temperature, record.source)
                        summary_counts[key].update(counts)
                        summary_tokens[key] += max(1, generated_tokens)
                        summary_generations[key] += 1
                        generation_rows.append(
                            {
                                "source": record.source,
                                "record_id": record.record_id,
                                "completion_index": completion_index,
                                "temperature": temperature,
                                "prompt_tokens": prompt_tokens,
                                "generated_tokens": generated_tokens,
                                "generation_chars": len(generation),
                                "features_json": json.dumps(counts, sort_keys=True),
                                "generation": generation,
                            }
                        )

        _write_jsonl(args.generations_output, generation_rows)
        summary_rows = []
        for (temperature, source), counts in sorted(summary_counts.items()):
            tokens = max(1, summary_tokens[(temperature, source)])
            generations = max(1, summary_generations[(temperature, source)])
            for feature in sorted(selected_features):
                count = counts.get(feature, 0)
                summary_rows.append(
                    {
                        "source": source,
                        "temperature": temperature,
                        "feature": feature,
                        "count": count,
                        "generations": generations,
                        "generated_tokens": tokens,
                        "per_1k_tokens": 1000.0 * count / tokens,
                        "per_generation": count / generations,
                    }
                )
        _write_csv(
            args.summary_output,
            summary_rows,
            [
                "source",
                "temperature",
                "feature",
                "count",
                "generations",
                "generated_tokens",
                "per_1k_tokens",
                "per_generation",
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
