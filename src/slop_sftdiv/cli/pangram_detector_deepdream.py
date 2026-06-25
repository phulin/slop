from __future__ import annotations

import argparse
import csv
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F

from slop_sftdiv.cli.pangram_llama_sae import (
    LLAMA_BASE_ID,
    PANGRAM_ADAPTER_ID,
    PangramScoreHead,
    _adapter_score_labels,
    _device,
    _from_pretrained_kwargs,
    _model_sources,
    _torch_dtype,
    download_pangram_model,
    load_pangram_tokenizer,
)


@dataclass(frozen=True)
class OptimizationResult:
    token_ids: list[int]
    text: str
    target_logit: float
    target_probability: float
    editlens_score: float
    label_logits: list[float]
    label_probabilities: list[float]
    mean_kl: float
    objective: float
    trajectory: list[dict[str, Any]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DeepDream-style relaxed-token optimization against the Pangram/EditLens "
            "Llama AI detector, with an optional static base-LM KL anchor."
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
    parser.add_argument("--download-model", action="store_true")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--initial-completion", default=None)
    parser.add_argument("--completion-tokens", type=int, default=96)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--target-label", type=int, default=1)
    parser.add_argument(
        "--target-objective",
        choices=["score", "logit", "logprob"],
        default="score",
        help=(
            "score maximizes the EditLens continuous score, computed as expected bucket "
            "index divided by n_buckets-1. logit/logprob maximize --target-label directly."
        ),
    )
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--softmax-temperature", type=float, default=1.0)
    parser.add_argument(
        "--kl-anchor",
        choices=["static_base", "none"],
        default="static_base",
        help=(
            "static_base anchors each optimized position to the base LM distribution "
            "under the initial hard continuation prefix."
        ),
    )
    parser.add_argument("--kl-coeff", type=float, default=0.05)
    parser.add_argument("--init-logit-bias", type=float, default=8.0)
    parser.add_argument("--allow-special-tokens", action="store_true")
    parser.add_argument("--generate-temperature", type=float, default=0.8)
    parser.add_argument("--generate-top-p", type=float, default=0.95)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    return str(args.prompt or "").strip()


def _token_ids(
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool,
    device: torch.device,
) -> torch.Tensor:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens, return_tensors="pt")
    return encoded["input_ids"].to(device)


def _fit_completion_ids(
    tokenizer: Any,
    text: str,
    *,
    completion_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    if completion_tokens <= 0:
        raise ValueError("--completion-tokens must be positive")
    ids = _token_ids(tokenizer, text, add_special_tokens=False, device=device)[0]
    if ids.numel() >= completion_tokens:
        return ids[:completion_tokens].clone()
    pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer needs an eos or pad token for completion padding")
    padding = torch.full(
        (completion_tokens - ids.numel(),),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    return torch.cat([ids, padding], dim=0)


def _special_token_ids(tokenizer: Any) -> set[int]:
    raw_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    for attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            raw_ids.add(int(value))
    return raw_ids


def _masked_logits(logits: torch.Tensor, banned_token_ids: set[int]) -> torch.Tensor:
    if not banned_token_ids:
        return logits
    masked = logits.clone()
    valid_ids = [token_id for token_id in banned_token_ids if 0 <= token_id < masked.shape[-1]]
    if valid_ids:
        masked[:, valid_ids] = -1.0e9
    return masked


def _decode_completion(tokenizer: Any, token_ids: list[int]) -> str:
    return str(tokenizer.decode(token_ids, skip_special_tokens=True)).strip()


def _load_base_lm(args: argparse.Namespace, *, device: torch.device) -> Any:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("transformers is required to load the base LM") from exc
    base_source, _adapter_source = _model_sources(args)
    kwargs = _from_pretrained_kwargs(args, revision=args.revision)
    direct_to_cuda = device.type == "cuda"
    if direct_to_cuda:
        kwargs["device_map"] = {"": str(device)}
    model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            base_source,
            torch_dtype=_torch_dtype(args.torch_dtype),
            low_cpu_mem_usage=True,
            **kwargs,
        ),
    )
    if not direct_to_cuda:
        target_dtype = _torch_dtype(args.torch_dtype)
        if isinstance(target_dtype, torch.dtype):
            model.to(device=device, dtype=target_dtype)
        else:
            model.to(device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model


def _load_pangram_detector(args: argparse.Namespace, *, device: torch.device) -> tuple[Any, Any]:
    try:
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:
        raise RuntimeError(
            "Pangram EditLens loading requires the optional model stack: "
            "install this project with the 'models' extra so peft/transformers are available."
        ) from exc

    base_source, adapter_source = _model_sources(args)
    tokenizer = load_pangram_tokenizer(args)
    kwargs = _from_pretrained_kwargs(args, revision=args.revision)
    direct_to_cuda = device.type == "cuda"
    if direct_to_cuda:
        kwargs["device_map"] = {"": str(device)}
    model = cast(
        Any,
        AutoModelForSequenceClassification.from_pretrained(
            base_source,
            torch_dtype=_torch_dtype(args.torch_dtype),
            low_cpu_mem_usage=True,
            **kwargs,
        ),
    )
    score_labels = _adapter_score_labels(adapter_source)
    if score_labels is not None:
        model.config.num_labels = score_labels
        model.config.id2label = {index: f"LABEL_{index}" for index in range(score_labels)}
        model.config.label2id = {label: index for index, label in model.config.id2label.items()}
        model.score = PangramScoreHead(
            hidden_size=int(model.config.hidden_size),
            num_labels=score_labels,
        )
        if direct_to_cuda:
            target_dtype = _torch_dtype(args.torch_dtype)
            if isinstance(target_dtype, torch.dtype):
                model.score.to(device=device, dtype=target_dtype)
            else:
                model.score.to(device)
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    adapter_kwargs = _from_pretrained_kwargs(args, revision=args.adapter_revision)
    if direct_to_cuda:
        adapter_kwargs["device_map"] = {"": str(device)}
    model = PeftModel.from_pretrained(model, adapter_source, **adapter_kwargs)
    if not direct_to_cuda:
        target_dtype = _torch_dtype(args.torch_dtype)
        if isinstance(target_dtype, torch.dtype):
            model.to(device=device, dtype=target_dtype)
        else:
            model.to(device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


@torch.inference_mode()
def _generate_initial_completion_ids(
    *,
    model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    completion_tokens: int,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    generated = model.generate(
        input_ids=prompt_ids,
        max_new_tokens=completion_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return generated[0, prompt_ids.shape[1] : prompt_ids.shape[1] + completion_tokens].clone()


@torch.inference_mode()
def static_base_anchor_log_probs(
    *,
    model: Any,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
) -> torch.Tensor:
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("prompt_ids must have shape (1, prompt_length)")
    if completion_ids.ndim != 1:
        raise ValueError("completion_ids must have shape (completion_length,)")
    if prompt_ids.shape[1] < 1:
        raise ValueError("prompt must tokenize to at least one token")
    input_ids = torch.cat([prompt_ids, completion_ids.unsqueeze(0)], dim=1)
    logits = model(input_ids=input_ids).logits.float()[0]
    start = prompt_ids.shape[1] - 1
    end = start + completion_ids.numel()
    return F.log_softmax(logits[start:end], dim=-1)


def kl_divergence_to_anchor(q: torch.Tensor, anchor_log_probs: torch.Tensor | None) -> torch.Tensor:
    if anchor_log_probs is None:
        return q.new_tensor(0.0)
    if q.shape != anchor_log_probs.shape:
        raise ValueError(
            f"q and anchor_log_probs shape mismatch: {tuple(q.shape)} != {tuple(anchor_log_probs.shape)}"
        )
    return (q * (q.clamp_min(1.0e-12).log() - anchor_log_probs)).sum(dim=-1).mean()


def editlens_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 1:
        raise ValueError("logits must be a 1D tensor")
    if logits.numel() < 2:
        raise ValueError("EditLens score requires at least two buckets")
    probabilities = F.softmax(logits, dim=-1)
    bucket_ids = torch.arange(logits.numel(), dtype=logits.dtype, device=logits.device)
    return (probabilities * bucket_ids).sum() / float(logits.numel() - 1)


def _target_loss(
    logits: torch.Tensor,
    *,
    target_label: int,
    target_objective: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_label < 0 or target_label >= logits.shape[-1]:
        raise ValueError(f"target label {target_label} is out of range for {logits.shape[-1]} labels")
    if target_objective == "score":
        target_score = editlens_score_from_logits(logits)
    elif target_objective == "logit":
        target_score = logits[target_label]
    elif target_objective == "logprob":
        target_score = F.log_softmax(logits, dim=-1)[target_label]
    else:
        raise ValueError(f"unknown target objective: {target_objective}")
    return -target_score, target_score


def optimize_soft_completion(
    *,
    detector_model: Any,
    tokenizer: Any,
    prompt_ids: torch.Tensor,
    initial_completion_ids: torch.Tensor,
    anchor_log_probs: torch.Tensor | None,
    target_label: int,
    target_objective: str,
    steps: int,
    learning_rate: float,
    softmax_temperature: float,
    kl_coeff: float,
    init_logit_bias: float,
    banned_token_ids: set[int],
    log_every: int,
) -> OptimizationResult:
    if steps <= 0:
        raise ValueError("--steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if softmax_temperature <= 0.0:
        raise ValueError("--softmax-temperature must be positive")
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("prompt_ids must have shape (1, prompt_length)")
    if initial_completion_ids.ndim != 1:
        raise ValueError("initial_completion_ids must have shape (completion_length,)")

    device = prompt_ids.device
    embedding_layer = detector_model.get_input_embeddings()
    vocab_size = int(embedding_layer.weight.shape[0])
    completion_tokens = int(initial_completion_ids.numel())
    free_logits = torch.zeros(
        (completion_tokens, vocab_size),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    with torch.no_grad():
        positions = torch.arange(completion_tokens, device=device)
        free_logits[positions, initial_completion_ids.clamp(0, vocab_size - 1)] = init_logit_bias
    optimizer = torch.optim.AdamW([free_logits], lr=learning_rate)
    prompt_embeddings = embedding_layer(prompt_ids).detach()
    attention_mask = torch.ones(
        (1, prompt_ids.shape[1] + completion_tokens),
        dtype=torch.long,
        device=device,
    )
    if anchor_log_probs is not None:
        anchor_log_probs = anchor_log_probs.detach().to(device=device, dtype=torch.float32)

    trajectory: list[dict[str, Any]] = []
    final_logits: torch.Tensor | None = None
    final_kl: torch.Tensor | None = None
    final_objective: torch.Tensor | None = None
    final_q: torch.Tensor | None = None
    for step in range(steps + 1):
        masked = _masked_logits(free_logits, banned_token_ids)
        q = F.softmax(masked / softmax_temperature, dim=-1)
        completion_embeddings = q.to(dtype=embedding_layer.weight.dtype) @ embedding_layer.weight
        inputs_embeds = torch.cat([prompt_embeddings, completion_embeddings.unsqueeze(0)], dim=1)
        outputs = detector_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits[0].float()
        detector_loss, target_score = _target_loss(
            logits,
            target_label=target_label,
            target_objective=target_objective,
        )
        mean_kl = kl_divergence_to_anchor(q, anchor_log_probs)
        loss = detector_loss + float(kl_coeff) * mean_kl
        if step == steps or step % max(1, log_every) == 0:
            probabilities = F.softmax(logits, dim=-1)
            editlens_score = editlens_score_from_logits(logits)
            token_ids = q.argmax(dim=-1).detach().cpu().tolist()
            trajectory.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "target_score": float(target_score.detach().cpu()),
                    "target_logit": float(logits[target_label].detach().cpu()),
                    "target_probability": float(probabilities[target_label].detach().cpu()),
                    "editlens_score": float(editlens_score.detach().cpu()),
                    "label_logits_json": json.dumps(
                        [float(value) for value in logits.detach().cpu().tolist()]
                    ),
                    "label_probabilities_json": json.dumps(
                        [float(value) for value in probabilities.detach().cpu().tolist()]
                    ),
                    "mean_kl": float(mean_kl.detach().cpu()),
                    "decoded_completion": _decode_completion(tokenizer, token_ids),
                }
            )
        final_logits = logits.detach()
        final_kl = mean_kl.detach()
        final_objective = target_score.detach()
        final_q = q.detach()
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    assert final_logits is not None
    assert final_kl is not None
    assert final_objective is not None
    assert final_q is not None
    final_probabilities = F.softmax(final_logits, dim=-1)
    final_editlens_score = editlens_score_from_logits(final_logits)
    final_token_ids = final_q.argmax(dim=-1).detach().cpu().tolist()
    return OptimizationResult(
        token_ids=final_token_ids,
        text=_decode_completion(tokenizer, final_token_ids),
        target_logit=float(final_logits[target_label].detach().cpu()),
        target_probability=float(final_probabilities[target_label].detach().cpu()),
        editlens_score=float(final_editlens_score.detach().cpu()),
        label_logits=[float(value) for value in final_logits.detach().cpu().tolist()],
        label_probabilities=[float(value) for value in final_probabilities.detach().cpu().tolist()],
        mean_kl=float(final_kl.detach().cpu()),
        objective=float(final_objective.detach().cpu()),
        trajectory=trajectory,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.download_model:
        download_pangram_model(args)
    prompt = _read_prompt(args)
    device = _device(args.device)
    tokenizer = load_pangram_tokenizer(args)
    prompt_ids = _token_ids(tokenizer, prompt, add_special_tokens=True, device=device)
    if prompt_ids.shape[1] >= args.max_length:
        raise ValueError("--max-length must leave room for at least one completion token")
    completion_tokens = min(int(args.completion_tokens), int(args.max_length) - int(prompt_ids.shape[1]))

    anchor_log_probs: torch.Tensor | None = None
    initial_completion_ids: torch.Tensor
    if args.initial_completion is None or args.kl_anchor == "static_base":
        base_lm = _load_base_lm(args, device=device)
        if args.initial_completion is None:
            initial_completion_ids = _generate_initial_completion_ids(
                model=base_lm,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                completion_tokens=completion_tokens,
                temperature=args.generate_temperature,
                top_p=args.generate_top_p,
            )
        else:
            initial_completion_ids = _fit_completion_ids(
                tokenizer,
                args.initial_completion,
                completion_tokens=completion_tokens,
                device=device,
            )
        if args.kl_anchor == "static_base":
            anchor_log_probs = static_base_anchor_log_probs(
                model=base_lm,
                prompt_ids=prompt_ids,
                completion_ids=initial_completion_ids,
            )
        del base_lm
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        initial_completion_ids = _fit_completion_ids(
            tokenizer,
            args.initial_completion,
            completion_tokens=completion_tokens,
            device=device,
        )

    detector_model, _ = _load_pangram_detector(args, device=device)
    banned_token_ids = set() if args.allow_special_tokens else _special_token_ids(tokenizer)
    result = optimize_soft_completion(
        detector_model=detector_model,
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        initial_completion_ids=initial_completion_ids,
        anchor_log_probs=anchor_log_probs,
        target_label=args.target_label,
        target_objective=args.target_objective,
        steps=args.steps,
        learning_rate=args.learning_rate,
        softmax_temperature=args.softmax_temperature,
        kl_coeff=args.kl_coeff if args.kl_anchor != "none" else 0.0,
        init_logit_bias=args.init_logit_bias,
        banned_token_ids=banned_token_ids,
        log_every=args.log_every,
    )

    initial_text = _decode_completion(tokenizer, initial_completion_ids.detach().cpu().tolist())
    summary = {
        "experiment": "pangram_detector_deepdream",
        "base_model": args.base_model,
        "adapter_model": args.adapter_model,
        "local_model_dir": str(args.local_model_dir),
        "prompt": prompt,
        "initial_completion": initial_text,
        "final_completion": result.text,
        "target_label": args.target_label,
        "target_objective": args.target_objective,
        "target_logit": result.target_logit,
        "target_probability": result.target_probability,
        "editlens_score": result.editlens_score,
        "label_logits": result.label_logits,
        "label_probabilities": result.label_probabilities,
        "objective": result.objective,
        "mean_kl": result.mean_kl,
        "kl_anchor": args.kl_anchor,
        "kl_coeff": args.kl_coeff if args.kl_anchor != "none" else 0.0,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "softmax_temperature": args.softmax_temperature,
        "completion_tokens": completion_tokens,
        "max_length": args.max_length,
        "seed": args.seed,
        "final_token_ids": result.token_ids,
    }
    _write_json(args.output_dir / "pangram_detector_deepdream_summary.json", summary)
    _write_csv(args.output_dir / "pangram_detector_deepdream_trajectory.csv", result.trajectory)
    print(f"Wrote Pangram detector DeepDream outputs to {args.output_dir}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
