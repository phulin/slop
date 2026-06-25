from __future__ import annotations

import argparse
import csv
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from slop_sftdiv.cli.pangram_detector_ar_search import (
    _detector_scores_for_sequences,
    _load_pangram_tokenizer_for_ar,
    _score_text,
    sequence_quality_penalty,
)
from slop_sftdiv.cli.pangram_detector_deepdream import (
    _decode_completion,
    _generate_initial_completion_ids,
    _load_base_lm,
    _load_pangram_detector,
    _read_prompt,
    _special_token_ids,
    _token_ids,
)
from slop_sftdiv.cli.pangram_llama_sae import LLAMA_BASE_ID, PANGRAM_ADAPTER_ID, _device


@dataclass(frozen=True)
class SampleResult:
    sample_index: int
    token_ids: tuple[int, ...]
    text: str
    editlens_score: float
    bucket_probabilities: list[float]
    mean_lm_log_probability: float
    sequence_quality_penalty: float
    objective: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate natural base-Llama continuations after a fixed prefix, then "
            "rerank them by Pangram/EditLens score, LM log probability, and a "
            "full-continuation prose-quality penalty."
        )
    )
    parser.add_argument("--base-model", default=LLAMA_BASE_ID)
    parser.add_argument("--adapter-model", default=PANGRAM_ADAPTER_ID)
    parser.add_argument(
        "--local-model-dir",
        type=Path,
        default=Path("artifacts/models/pangram_editlens_Llama-3.2-3B"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--adapter-revision", default=None)
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--no-hf-token", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--prefix-tokens", type=int, default=50)
    parser.add_argument("--continuation-tokens", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--sample-batch-size", type=int, default=4)
    parser.add_argument("--generate-temperature", type=float, default=0.8)
    parser.add_argument("--generate-top-p", type=float, default=0.95)
    parser.add_argument("--sample-temperature", type=float, default=0.9)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    parser.add_argument("--lm-logprob-coeff", type=float, default=0.0)
    parser.add_argument("--sequence-quality-penalty-coeff", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--allow-special-tokens", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def sample_objective(
    *,
    editlens_score: float,
    mean_lm_log_probability: float,
    sequence_quality_penalty_value: float,
    lm_logprob_coeff: float,
    sequence_quality_penalty_coeff: float,
) -> float:
    return (
        float(editlens_score)
        + float(lm_logprob_coeff) * float(mean_lm_log_probability)
        - float(sequence_quality_penalty_coeff) * float(sequence_quality_penalty_value)
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def _sample_continuation_batch(
    *,
    model: Any,
    tokenizer: Any,
    context_ids: torch.Tensor,
    continuation_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    banned_token_ids: set[int],
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("--sample-batch-size must be positive")
    generation_context = context_ids.repeat(batch_size, 1)
    generated = model.generate(
        input_ids=generation_context,
        max_new_tokens=continuation_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        bad_words_ids=[[token_id] for token_id in sorted(banned_token_ids)] or None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=None if tokenizer.eos_token_id in banned_token_ids else tokenizer.eos_token_id,
    )
    return generated[:, context_ids.shape[1] : context_ids.shape[1] + continuation_tokens].clone()


@torch.inference_mode()
def _mean_lm_log_probabilities(
    *,
    model: Any,
    context_ids: torch.Tensor,
    continuation_ids: torch.Tensor,
    max_length: int,
) -> list[float]:
    if context_ids.ndim != 2 or context_ids.shape[0] != 1:
        raise ValueError("context_ids must have shape (1, sequence_length)")
    if continuation_ids.ndim != 2:
        raise ValueError("continuation_ids must have shape (batch, sequence_length)")
    rows = [
        torch.cat([context_ids[0], continuation], dim=0)[-max_length:]
        for continuation in continuation_ids
    ]
    input_ids = torch.stack(rows, dim=0)
    logits = model(input_ids=input_ids).logits.float()
    continuation_length = continuation_ids.shape[1]
    start = input_ids.shape[1] - continuation_length - 1
    end = input_ids.shape[1] - 1
    if start < 0:
        raise ValueError("--max-length must leave at least one context token before sampled continuation")
    token_log_probs = F.log_softmax(logits[:, start:end, :], dim=-1)
    gathered = token_log_probs.gather(dim=-1, index=continuation_ids[:, :, None]).squeeze(-1)
    return [float(value) for value in gathered.mean(dim=-1).detach().cpu().tolist()]


def _result_row(result: SampleResult) -> dict[str, Any]:
    return {
        "sample_index": result.sample_index,
        "editlens_score": result.editlens_score,
        "objective": result.objective,
        "mean_lm_log_probability": result.mean_lm_log_probability,
        "sequence_quality_penalty": result.sequence_quality_penalty,
        "bucket_probabilities_json": json.dumps(result.bucket_probabilities),
        "continuation": result.text,
        "token_ids_json": json.dumps(list(result.token_ids)),
    }


def _summary_for_best(
    *,
    args: argparse.Namespace,
    prompt: str,
    initial_prefix: str,
    initial_score: float,
    initial_bucket_probabilities: list[float],
    best: SampleResult,
    full_text: str,
) -> dict[str, Any]:
    return {
        "experiment": "pangram_detector_sample_search",
        "base_model": args.base_model,
        "adapter_model": args.adapter_model,
        "local_model_dir": str(args.local_model_dir),
        "prompt": prompt,
        "initial_prefix": initial_prefix,
        "optimized_continuation": best.text,
        "full_text": full_text,
        "initial_editlens_score": initial_score,
        "initial_bucket_probabilities": initial_bucket_probabilities,
        "final_editlens_score": best.editlens_score,
        "final_bucket_probabilities": best.bucket_probabilities,
        "final_mean_lm_log_probability": best.mean_lm_log_probability,
        "final_sequence_quality_penalty": best.sequence_quality_penalty,
        "final_objective": best.objective,
        "prefix_tokens": args.prefix_tokens,
        "continuation_tokens": args.continuation_tokens,
        "num_samples": args.num_samples,
        "sample_batch_size": args.sample_batch_size,
        "sample_temperature": args.sample_temperature,
        "sample_top_p": args.sample_top_p,
        "lm_logprob_coeff": args.lm_logprob_coeff,
        "sequence_quality_penalty_coeff": args.sequence_quality_penalty_coeff,
        "max_length": args.max_length,
        "seed": args.seed,
        "generated_token_ids": list(best.token_ids),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.prefix_tokens <= 0:
        raise ValueError("--prefix-tokens must be positive")
    if args.continuation_tokens <= 0:
        raise ValueError("--continuation-tokens must be positive")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt = _read_prompt(args)
    device = _device(args.device)
    tokenizer = _load_pangram_tokenizer_for_ar(args)
    prompt_ids = _token_ids(tokenizer, prompt, add_special_tokens=True, device=device)
    base_lm = _load_base_lm(args, device=device)
    initial_ids = _generate_initial_completion_ids(
        model=base_lm,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        completion_tokens=args.prefix_tokens,
        temperature=args.generate_temperature,
        top_p=args.generate_top_p,
    )
    context_ids = torch.cat([prompt_ids, initial_ids.unsqueeze(0)], dim=1)
    detector_model, _ = _load_pangram_detector(args, device=device)
    initial_score, initial_bucket_probabilities = _score_text(
        detector_model=detector_model,
        input_ids=context_ids,
        max_length=args.max_length,
    )
    banned_token_ids = set() if args.allow_special_tokens else _special_token_ids(tokenizer)
    results: list[SampleResult] = []
    sample_index = 0
    while sample_index < args.num_samples:
        current_batch_size = min(args.sample_batch_size, args.num_samples - sample_index)
        continuation_batch = _sample_continuation_batch(
            model=base_lm,
            tokenizer=tokenizer,
            context_ids=context_ids,
            continuation_tokens=args.continuation_tokens,
            batch_size=current_batch_size,
            temperature=args.sample_temperature,
            top_p=args.sample_top_p,
            banned_token_ids=banned_token_ids,
        )
        scores, probability_rows = _detector_scores_for_sequences(
            detector_model=detector_model,
            prefix_ids=context_ids,
            sequence_ids=[
                tuple(int(token_id) for token_id in row.detach().cpu().tolist())
                for row in continuation_batch
            ],
            max_length=args.max_length,
        )
        mean_lm_log_probabilities = _mean_lm_log_probabilities(
            model=base_lm,
            context_ids=context_ids,
            continuation_ids=continuation_batch,
            max_length=args.max_length,
        )
        for batch_offset, continuation_ids in enumerate(continuation_batch):
            token_ids = tuple(int(token_id) for token_id in continuation_ids.detach().cpu().tolist())
            text = _decode_completion(tokenizer, list(token_ids))
            quality_penalty = sequence_quality_penalty(text)
            editlens_score = float(scores[batch_offset].detach().cpu())
            mean_lm_log_probability = mean_lm_log_probabilities[batch_offset]
            objective = sample_objective(
                editlens_score=editlens_score,
                mean_lm_log_probability=mean_lm_log_probability,
                sequence_quality_penalty_value=quality_penalty,
                lm_logprob_coeff=args.lm_logprob_coeff,
                sequence_quality_penalty_coeff=args.sequence_quality_penalty_coeff,
            )
            results.append(
                SampleResult(
                    sample_index=sample_index,
                    token_ids=token_ids,
                    text=text,
                    editlens_score=editlens_score,
                    bucket_probabilities=probability_rows[batch_offset],
                    mean_lm_log_probability=mean_lm_log_probability,
                    sequence_quality_penalty=quality_penalty,
                    objective=objective,
                )
            )
            sample_index += 1
        if args.progress_every > 0 and (
            sample_index >= args.num_samples or sample_index % args.progress_every == 0
        ):
            best_so_far = max(results, key=lambda result: result.objective)
            print(
                f"samples={sample_index} best_score={best_so_far.editlens_score:.4f} "
                f"best_obj={best_so_far.objective:.4f} qpen={best_so_far.sequence_quality_penalty:.3f}",
                flush=True,
            )

    rows = [_result_row(result) for result in sorted(results, key=lambda result: result.objective, reverse=True)]
    best = max(results, key=lambda result: result.objective)
    best_context_ids = torch.cat(
        [context_ids, torch.tensor([list(best.token_ids)], dtype=context_ids.dtype, device=context_ids.device)],
        dim=1,
    )
    summary = _summary_for_best(
        args=args,
        prompt=prompt,
        initial_prefix=_decode_completion(tokenizer, initial_ids.detach().cpu().tolist()),
        initial_score=initial_score,
        initial_bucket_probabilities=initial_bucket_probabilities,
        best=best,
        full_text=_decode_completion(tokenizer, best_context_ids[0].detach().cpu().tolist()),
    )
    _write_json(args.output_dir / "pangram_detector_sample_search_summary.json", summary)
    _write_csv(args.output_dir / "pangram_detector_sample_search_samples.csv", rows)
    print(f"Wrote Pangram detector sample-search outputs to {args.output_dir}", flush=True)
    del base_lm, detector_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
