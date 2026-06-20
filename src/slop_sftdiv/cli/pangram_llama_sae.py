from __future__ import annotations

import argparse
import heapq
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from slop_sftdiv.cli.phase4_batchtopk_sae import (
    ActivationCache,
    SAEDoc,
    _activation_mask,
    _activation_document_rows,
    _activation_row_indexes,
    _encode_docs,
    _load_activation_cache,
    _load_docs,
    _save_activation_cache_parquet,
    _save_sae,
    _validate_activation_cache,
    _token_batches,
    _write_csv,
    _write_json,
    train_sae,
)


PANGRAM_ADAPTER_ID = "pangram/editlens_Llama-3.2-3B"
LLAMA_BASE_ID = "meta-llama/Llama-3.2-3B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download Pangram's EditLens Llama-3.2-3B PEFT adapter and train a "
            "BatchTopK SAE on the model's final token-by-token hidden states."
        )
    )
    parser.add_argument("--base-model", default=LLAMA_BASE_ID)
    parser.add_argument("--adapter-model", default=PANGRAM_ADAPTER_ID)
    parser.add_argument(
        "--local-model-dir",
        type=Path,
        default=Path("artifacts/models/pangram_editlens_Llama-3.2-3B"),
        help="Directory where the gated Llama base and Pangram adapter snapshots are stored.",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--adapter-revision", default=None)
    parser.add_argument(
        "--hf-token-env",
        default="HF_TOKEN",
        help="Environment variable containing the Hugging Face token. Falls back to cached HF login.",
    )
    parser.add_argument("--no-hf-token", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Snapshot the base model and adapter, then exit before activation collection.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-jsonl",
        action="append",
        default=[],
        metavar="LABEL=PATH[:FIELD]",
        help="Reference/human JSONL included in SAE training with label 0.",
    )
    parser.add_argument(
        "--generation-jsonl",
        action="append",
        default=[],
        metavar="LABEL=PATH[:FIELD]",
        help="Generation JSONL included in SAE training with label 1.",
    )
    parser.add_argument(
        "--dataset-parquet",
        type=Path,
        default=None,
        help=(
            "Compiled continuation parquet with role/model/text columns. "
            "Rows with role=human_continuation are label 0; llm_continuation rows are label 1."
        ),
    )
    parser.add_argument("--feature-text-mode", choices=["raw", "final_answer"], default="final_answer")
    parser.add_argument("--max-docs-per-source", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--collect-batch-size", type=int, default=2)
    parser.add_argument("--pad-to-max-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-activations", type=int, default=200_000)
    parser.add_argument(
        "--activation-layer",
        type=int,
        default=-1,
        help=(
            "Hidden-state layer to cache. -1 caches the model's final hidden state; "
            "non-negative values cache the output of that Llama decoder layer."
        ),
    )
    parser.add_argument(
        "--activation-cache",
        type=Path,
        default=None,
        help="Optional activation cache path. Existing caches are reused unless --refresh-activation-cache is set.",
    )
    parser.add_argument("--compress-activation-cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--activation-cache-compresslevel",
        type=int,
        default=3,
        help="Zstd compression level for Parquet activation caches.",
    )
    parser.add_argument("--refresh-activation-cache", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=4096)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--init-mode", choices=["kaiming", "tied"], default="kaiming")
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-schedule", choices=["constant", "wsd"], default="constant")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--decay-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--compile-sae", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="default")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--skip-latent-scoring", action="store_true")
    parser.add_argument("--score-top-latents", type=int, default=64)
    parser.add_argument("--score-batch-size", type=int, default=4096)
    parser.add_argument("--max-examples-per-latent", type=int, default=12)
    parser.add_argument("--example-context-tokens", type=int, default=24)
    return parser


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=Path(".env"))


def _hf_token(args: argparse.Namespace) -> str | bool | None:
    if args.no_hf_token:
        return None
    if not os.environ.get(args.hf_token_env, "").strip():
        _load_dotenv_if_available()
    token = os.environ.get(args.hf_token_env, "").strip()
    return token if token else True


def download_pangram_model(args: argparse.Namespace) -> tuple[Path, Path]:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download Pangram/Llama snapshots") from exc

    base_dir = args.local_model_dir / "base"
    adapter_dir = args.local_model_dir / "adapter"
    base_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    token = _hf_token(args)
    cache_dir = str(args.cache_dir) if args.cache_dir is not None else None
    local_files_only = bool(args.local_files_only)
    print(f"Downloading base model {args.base_model} to {base_dir}", flush=True)
    try:
        snapshot_download(
            repo_id=args.base_model,
            revision=args.revision,
            local_dir=base_dir,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
    except LocalTokenNotFoundError as exc:
        raise RuntimeError(
            f"{args.base_model} requires Hugging Face authentication. "
            f"Set {args.hf_token_env} to a token with accepted Llama-3.2 access, "
            "or run `hf auth login`, then retry."
        ) from exc
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"Could not download gated base model {args.base_model}. "
            f"Confirm the token in {args.hf_token_env} has accepted the model terms."
        ) from exc
    print(f"Downloading adapter {args.adapter_model} to {adapter_dir}", flush=True)
    try:
        snapshot_download(
            repo_id=args.adapter_model,
            revision=args.adapter_revision,
            local_dir=adapter_dir,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            token=token,
        )
    except HfHubHTTPError as exc:
        raise RuntimeError(f"Could not download Pangram adapter {args.adapter_model}.") from exc
    return base_dir, adapter_dir


def _torch_dtype(raw: str) -> torch.dtype | str:
    if raw == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[raw]


def _device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _snapshot_dir_has_file(path: Path, file_name: str) -> bool:
    return path.is_dir() and (path / file_name).is_file()


def _model_sources(args: argparse.Namespace) -> tuple[str | Path, str | Path]:
    base_dir = args.local_model_dir / "base"
    adapter_dir = args.local_model_dir / "adapter"
    base_source: str | Path = (
        base_dir if _snapshot_dir_has_file(base_dir, "config.json") else args.base_model
    )
    adapter_source: str | Path = (
        adapter_dir if _snapshot_dir_has_file(adapter_dir, "adapter_config.json") else args.adapter_model
    )
    return base_source, adapter_source


def _from_pretrained_kwargs(args: argparse.Namespace, *, revision: str | None) -> dict[str, Any]:
    return {
        "cache_dir": str(args.cache_dir) if args.cache_dir is not None else None,
        "local_files_only": bool(args.local_files_only),
        "revision": revision,
        "token": _hf_token(args),
    }


class PangramScoreHead(nn.Module):
    def __init__(self, *, hidden_size: int, num_labels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, num_labels, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(hidden_states))


def _adapter_score_labels(adapter_source: str | Path) -> int | None:
    if not isinstance(adapter_source, Path):
        return None
    adapter_weights = adapter_source / "adapter_model.safetensors"
    if not adapter_weights.is_file():
        return None
    try:
        from safetensors.torch import safe_open
    except ImportError:
        return None
    with safe_open(adapter_weights, framework="pt", device="cpu") as handle:
        key = "base_model.model.score.linear.weight"
        if key not in handle.keys():
            return None
        return int(handle.get_tensor(key).shape[0])


def load_pangram_tokenizer(args: argparse.Namespace) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to load the Pangram tokenizer") from exc
    base_source, _adapter_source = _model_sources(args)
    tokenizer = cast(
        Any,
        AutoTokenizer.from_pretrained(
            base_source,
            **_from_pretrained_kwargs(args, revision=args.revision),
        ),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_pangram_model(args: argparse.Namespace, *, device: torch.device) -> tuple[Any, Any]:
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
    model = AutoModelForSequenceClassification.from_pretrained(
        base_source,
        torch_dtype=_torch_dtype(args.torch_dtype),
        low_cpu_mem_usage=True,
        **_from_pretrained_kwargs(args, revision=args.revision),
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
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(
        model,
        adapter_source,
        **_from_pretrained_kwargs(args, revision=args.adapter_revision),
    )
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
def collect_last_token_activation_cache(
    *,
    model: Any,
    tokenizer: Any,
    docs: list[SAEDoc],
    max_length: int,
    batch_size: int,
    max_activations: int,
    device: torch.device,
    progress_every: int,
    activation_layer: int = -1,
    pad_to_max_length: bool = False,
) -> ActivationCache:
    if max_activations <= 0:
        raise ValueError("--max-activations must be positive")
    chunks: list[torch.Tensor] = []
    doc_index_chunks: list[torch.Tensor] = []
    token_index_chunks: list[torch.Tensor] = []
    collected = 0
    batches = _token_batches(docs, batch_size)
    for batch_index, batch_docs in enumerate(batches, start=1):
        batch_start_doc_index = (batch_index - 1) * batch_size
        encoded = _encode_docs(
            tokenizer,
            batch_docs,
            max_length=max_length,
            device=device,
            pad_to_max_length=pad_to_max_length,
        )
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Pangram/Llama forward pass did not return hidden_states")
        hidden_index = -1 if activation_layer < 0 else activation_layer + 1
        if not -len(hidden_states) <= hidden_index < len(hidden_states):
            raise ValueError(
                f"--activation-layer {activation_layer} is out of range for "
                f"{len(hidden_states) - 1} decoder layers"
            )
        hidden = hidden_states[hidden_index]
        token_mask = _activation_mask(encoded)
        vectors = hidden[token_mask].detach().to(dtype=torch.float16).cpu()
        doc_indexes, token_indexes = _activation_row_indexes(
            token_mask=token_mask,
            batch_start_doc_index=batch_start_doc_index,
        )
        remaining = max_activations - collected
        if remaining <= 0:
            break
        if vectors.shape[0] > remaining:
            vectors = vectors[:remaining]
            doc_indexes = doc_indexes[:remaining]
            token_indexes = token_indexes[:remaining]
        chunks.append(vectors)
        doc_index_chunks.append(doc_indexes)
        token_index_chunks.append(token_indexes)
        collected += int(vectors.shape[0])
        if progress_every > 0 and (batch_index == 1 or batch_index % progress_every == 0):
            layer_name = "final-layer" if activation_layer < 0 else f"layer-{activation_layer}"
            print(
                f"Collected {collected:,}/{max_activations:,} {layer_name} Llama activation vectors "
                f"from {min(batch_index * batch_size, len(docs)):,}/{len(docs):,} docs",
                flush=True,
            )
        if collected >= max_activations:
            break
    if not chunks:
        raise RuntimeError("no activation vectors collected")
    return _validate_activation_cache(
        ActivationCache(
            activations=torch.cat(chunks, dim=0),
            document_rows=_activation_document_rows(docs),
            doc_indices=torch.cat(doc_index_chunks, dim=0),
            token_indices=torch.cat(token_index_chunks, dim=0),
        )
    )


@torch.inference_mode()
def collect_last_token_activations(
    *,
    model: Any,
    tokenizer: Any,
    docs: list[SAEDoc],
    max_length: int,
    batch_size: int,
    max_activations: int,
    device: torch.device,
    progress_every: int,
    activation_layer: int = -1,
    pad_to_max_length: bool = False,
) -> torch.Tensor:
    return collect_last_token_activation_cache(
        model=model,
        tokenizer=tokenizer,
        docs=docs,
        max_length=max_length,
        batch_size=batch_size,
        max_activations=max_activations,
        activation_layer=activation_layer,
        device=device,
        progress_every=progress_every,
        pad_to_max_length=pad_to_max_length,
    ).activations


def _save_pangram_activation_cache(
    path: Path,
    *,
    activation_cache: ActivationCache,
    args: argparse.Namespace,
    docs: list[SAEDoc],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    activation_cache = _validate_activation_cache(activation_cache)
    metadata = {
        "cache_version": 2,
        "source": "pangram_llama_last_hidden_state",
        "activation_layer": args.activation_layer,
        "base_model": args.base_model,
        "adapter_model": args.adapter_model,
        "local_model_dir": str(args.local_model_dir),
        "dataset_parquet": str(args.dataset_parquet) if args.dataset_parquet else "",
        "reference_jsonl": list(args.reference_jsonl),
        "generation_jsonl": list(args.generation_jsonl),
        "max_length": args.max_length,
        "pad_to_max_length": bool(args.pad_to_max_length),
        "max_activations": args.max_activations,
        "documents": len(docs),
    }
    if path.suffix == ".parquet":
        _save_activation_cache_parquet(
            path,
            activation_cache=activation_cache,
            metadata=metadata,
            compression_level=args.activation_cache_compresslevel,
        )
    else:
        payload = {
            "activations": activation_cache.activations.half(),
            **metadata,
            "activation_documents": activation_cache.document_rows,
            "activation_doc_indices": activation_cache.doc_indices,
            "activation_token_indices": activation_cache.token_indices,
        }
        torch.save(payload, path)


def _source_key_for_document(row: dict[str, Any]) -> str:
    label = int(row.get("label", -1))
    if label == 0:
        return "human"
    return str(row.get("source", "") or "ai")


def _decode_cached_context(
    *,
    tokenizer: Any,
    doc: SAEDoc,
    token_index: int,
    max_length: int,
    context_tokens: int,
) -> tuple[str, str]:
    encoded = tokenizer(
        doc.text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    ids = list(encoded.get("input_ids", []))
    if not ids:
        return "", ""
    index = max(0, min(int(token_index), len(ids) - 1))
    token = tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens(ids[index])]).strip()
    left = max(0, index - context_tokens)
    right = min(len(ids), index + context_tokens + 1)
    context = tokenizer.decode(ids[left:right], skip_special_tokens=True)
    return token, " ".join(str(context).split())


def _push_latent_example(
    heaps: dict[int, list[tuple[float, int, dict[str, Any]]]],
    *,
    latent_id: int,
    activation: float,
    serial: int,
    row: dict[str, Any],
    max_examples: int,
) -> None:
    heap = heaps[latent_id]
    item = (activation, serial, row)
    if len(heap) < max_examples:
        heapq.heappush(heap, item)
        return
    if activation > heap[0][0]:
        heapq.heapreplace(heap, item)


@torch.inference_mode()
def score_pangram_latents_from_cache(
    *,
    sae: Any,
    activation_cache: ActivationCache,
    docs: list[SAEDoc],
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rank Pangram SAE latents by AI-vs-human source enrichment.

    This uses dense ReLU encoder activations rather than BatchTopK-masked activations
    so offline scoring is stable and does not depend on arbitrary scan chunk size.
    """

    activation_cache = _validate_activation_cache(activation_cache)
    if not activation_cache.has_provenance:
        raise ValueError("latent scoring requires an activation cache with document/token provenance")
    if len(docs) < len(activation_cache.document_rows):
        raise ValueError(
            "latent scoring requires the loaded docs to cover all activation cache documents"
        )

    sae.eval()
    latent_dim = int(sae.latent_dim)
    source_mass: dict[str, torch.Tensor] = defaultdict(
        lambda: torch.zeros(latent_dim, dtype=torch.float64)
    )
    source_token_totals: Counter[str] = Counter()
    latent_max = torch.zeros(latent_dim, dtype=torch.float32)
    chunk_size = max(1, int(args.score_batch_size))
    activations = activation_cache.activations
    doc_indices = activation_cache.doc_indices
    token_indices = activation_cache.token_indices
    document_rows = activation_cache.document_rows

    for start in range(0, int(activations.shape[0]), chunk_size):
        end = min(int(activations.shape[0]), start + chunk_size)
        batch = activations[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        dense_codes = sae.encode_dense(batch).detach().cpu()
        latent_max = torch.maximum(latent_max, dense_codes.max(dim=0).values.float())
        batch_doc_indices = doc_indices[start:end].cpu()
        batch_token_indices = token_indices[start:end].cpu()
        for source in {_source_key_for_document(document_rows[int(index)]) for index in batch_doc_indices}:
            source_indexes = [
                offset
                for offset, doc_index in enumerate(batch_doc_indices.tolist())
                if _source_key_for_document(document_rows[int(doc_index)]) == source
            ]
            if not source_indexes:
                continue
            source_tensor_indexes = torch.tensor(source_indexes, dtype=torch.long)
            source_mass[source] += dense_codes[source_tensor_indexes].sum(dim=0, dtype=torch.float64)
            source_token_totals[source] += len(source_indexes)

        if args.progress_every > 0 and (start == 0 or (start // chunk_size) % args.progress_every == 0):
            print(
                f"Scored Pangram SAE activations for {end:,}/{int(activations.shape[0]):,} vectors",
                flush=True,
            )

    ai_sources = [source for source in source_mass if source != "human"]
    eps = 1e-6
    rows: list[dict[str, Any]] = []
    human_tokens = max(1, int(source_token_totals["human"]))
    ai_tokens = max(1, sum(int(source_token_totals[source]) for source in ai_sources))
    for latent_id in range(latent_dim):
        human_mass = float(source_mass["human"][latent_id]) if "human" in source_mass else 0.0
        ai_mass = sum(float(source_mass[source][latent_id]) for source in ai_sources)
        total_mass = human_mass + ai_mass
        if total_mass <= 0.0:
            continue
        ai_rate = ai_mass / ai_tokens
        human_rate = human_mass / human_tokens
        log_odds = math.log((ai_rate + eps) / (human_rate + eps))
        rows.append(
            {
                "latent_id": latent_id,
                "ai_human_log_odds": log_odds,
                "ai_mass": ai_mass,
                "human_mass": human_mass,
                "ai_mass_share": ai_mass / max(eps, total_mass),
                "total_mass": total_mass,
                "max_activation": float(latent_max[latent_id]),
                "source_token_totals": "|".join(
                    f"{source}:{source_token_totals[source]}"
                    for source in sorted(source_token_totals)
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["ai_human_log_odds"]),
            float(row["ai_mass_share"]),
            float(row["total_mass"]),
        ),
        reverse=True,
    )
    output_rows = rows[: int(args.score_top_latents)]
    for rank, row in enumerate(output_rows, start=1):
        row["rank"] = rank

    output_ids = [int(row["latent_id"]) for row in output_rows]
    output_id_set = set(output_ids)
    example_heaps: dict[int, list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    if output_ids:
        selected = torch.tensor(output_ids, dtype=torch.long)
        serial = 0
        max_examples = int(args.max_examples_per_latent)
        candidate_examples = max(max_examples, max_examples * 20)
        for start in range(0, int(activations.shape[0]), chunk_size):
            end = min(int(activations.shape[0]), start + chunk_size)
            batch = activations[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
            selected_codes = sae.encode_dense(batch).detach().cpu()[:, selected]
            batch_doc_indices = doc_indices[start:end].cpu()
            batch_token_indices = token_indices[start:end].cpu()
            keep = min(max_examples, int(selected_codes.shape[0]))
            top_values, top_indexes = torch.topk(selected_codes, k=keep, dim=0)
            for selected_offset, latent_id in enumerate(output_ids):
                for value, row_index in zip(
                    top_values[:, selected_offset].tolist(),
                    top_indexes[:, selected_offset].tolist(),
                ):
                    if value <= 0:
                        continue
                    doc_index = int(batch_doc_indices[int(row_index)])
                    token_index = int(batch_token_indices[int(row_index)])
                    doc = docs[doc_index]
                    serial += 1
                    _push_latent_example(
                        example_heaps,
                        latent_id=latent_id,
                        activation=float(value),
                        serial=serial,
                        row={
                            "latent_id": latent_id,
                            "activation": float(value),
                            "source": _source_key_for_document(document_rows[doc_index]),
                            "doc_id": doc.doc_id,
                            "label": int(doc.label),
                            "token_index": token_index,
                        },
                        max_examples=candidate_examples,
                    )
    example_rows: list[dict[str, Any]] = []
    for row in output_rows:
        latent_id = int(row["latent_id"])
        seen_doc_ids: set[str] = set()
        example_rank = 0
        for _activation, _serial, example in sorted(
            example_heaps.get(latent_id, []),
            key=lambda item: item[0],
            reverse=True,
        ):
            if latent_id not in output_id_set:
                continue
            doc_id = str(example["doc_id"])
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            example_rank += 1
            doc = next(doc for doc in docs if doc.doc_id == example["doc_id"])
            token, context = _decode_cached_context(
                tokenizer=tokenizer,
                doc=doc,
                token_index=int(example["token_index"]),
                max_length=args.max_length,
                context_tokens=args.example_context_tokens,
            )
            example_rows.append(
                {
                    "rank": int(row["rank"]),
                    "example_rank": example_rank,
                    **example,
                    "token": token,
                    "context": context,
                }
            )
            if example_rank >= max_examples:
                break
    return output_rows, example_rows


def _summary(
    args: argparse.Namespace,
    *,
    docs: list[SAEDoc],
    device: torch.device,
    cache_path: Path,
    cache_reused: bool,
    activation_cache_has_provenance: bool,
    activations: torch.Tensor,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    train_summary: dict[str, Any],
) -> dict[str, Any]:
    source_counts = Counter(doc.source for doc in docs)
    label_counts = Counter(str(doc.label) for doc in docs)
    return {
        "base_model": args.base_model,
        "adapter_model": args.adapter_model,
        "local_model_dir": str(args.local_model_dir),
        "activation_source": "pangram_llama_last_hidden_state",
        "activation_layer": args.activation_layer,
        "device": str(device),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "documents": len(docs),
        "source_counts": dict(sorted(source_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "activation_cache": str(cache_path),
        "activation_cache_reused": cache_reused,
        "activation_cache_has_provenance": activation_cache_has_provenance,
        "activation_vectors": int(activations.shape[0]),
        "input_dim": int(activations.shape[1]),
        "pad_to_max_length": bool(args.pad_to_max_length),
        "latent_dim": args.latent_dim,
        "k": args.k,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_schedule": args.lr_schedule,
        "warmup_ratio": args.warmup_ratio,
        "decay_ratio": args.decay_ratio,
        "min_lr_ratio": args.min_lr_ratio,
        "warmup_steps": train_summary.get("warmup_steps", 0),
        "stable_steps": train_summary.get("stable_steps", 0),
        "decay_steps": train_summary.get("decay_steps", 0),
        "eval_fraction": args.eval_fraction,
        "final_train_mse": float(train_rows[-1]["mse"]) if train_rows else None,
        "best_eval_mse": min((float(row["eval_mse"]) for row in eval_rows), default=None),
        "compile_sae": bool(args.compile_sae),
        "compile_sae_fallback": bool(train_summary["compile_sae_fallback"]),
        "compile_mode": args.compile_mode,
        "torch_dtype": args.torch_dtype,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.local_files_only:
        download_pangram_model(args)
    if args.download_only:
        payload = {
            "base_model": args.base_model,
            "adapter_model": args.adapter_model,
            "local_model_dir": str(args.local_model_dir),
            "downloaded": True,
        }
        _write_json(args.output_dir / "pangram_llama_sae_download.json", payload)
        return payload

    docs = _load_docs(args)
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    cache_name = (
        "pangram_llama_sae_activation_cache.parquet"
        if args.compress_activation_cache
        else "pangram_llama_sae_activation_cache.pt"
    )
    cache_path = args.activation_cache or args.output_dir / cache_name
    cache_reused = cache_path.exists() and not args.refresh_activation_cache
    tokenizer: Any | None = None
    if cache_reused:
        print(f"Loading activation cache from {cache_path}", flush=True)
        activation_cache = _load_activation_cache(cache_path)
        activations = activation_cache.activations
    else:
        model, tokenizer = load_pangram_model(args, device=device)
        activation_cache = collect_last_token_activation_cache(
            model=model,
            tokenizer=tokenizer,
            docs=docs,
            max_length=args.max_length,
            batch_size=args.collect_batch_size,
            max_activations=args.max_activations,
            activation_layer=args.activation_layer,
            device=device,
            progress_every=args.progress_every,
            pad_to_max_length=args.pad_to_max_length,
        )
        activations = activation_cache.activations
        _save_pangram_activation_cache(
            cache_path,
            activation_cache=activation_cache,
            args=args,
            docs=docs,
        )

    sae, train_rows, eval_rows, train_summary = train_sae(
        activations=activations,
        args=args,
        device=device,
    )
    summary = _summary(
        args,
        docs=docs,
        device=device,
        cache_path=cache_path,
        cache_reused=cache_reused,
        activation_cache_has_provenance=activation_cache.has_provenance,
        activations=activations,
        train_rows=train_rows,
        eval_rows=eval_rows,
        train_summary=train_summary,
    )
    _save_sae(args.output_dir / "pangram_llama_batchtopk_sae.pt", sae, summary)
    _write_csv(args.output_dir / "pangram_llama_sae_train_log.csv", train_rows)
    _write_csv(args.output_dir / "pangram_llama_sae_eval_log.csv", eval_rows)
    if not args.skip_latent_scoring:
        if tokenizer is None:
            tokenizer = load_pangram_tokenizer(args)
        latent_rows, example_rows = score_pangram_latents_from_cache(
            sae=sae,
            activation_cache=activation_cache,
            docs=docs,
            tokenizer=tokenizer,
            args=args,
            device=device,
        )
        _write_csv(args.output_dir / "pangram_llama_sae_latents.csv", latent_rows)
        _write_csv(args.output_dir / "pangram_llama_sae_examples.csv", example_rows)
        summary["latent_scoring"] = {
            "score_top_latents": int(args.score_top_latents),
            "max_examples_per_latent": int(args.max_examples_per_latent),
            "latents_written": len(latent_rows),
            "examples_written": len(example_rows),
            "scoring_method": "dense_relu_source_enrichment_from_activation_cache",
        }
    _write_json(args.output_dir / "pangram_llama_sae_summary.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    try:
        summary = run(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(
        "Wrote Pangram Llama SAE outputs to "
        f"{args.output_dir} ({summary.get('activation_vectors', 0)} activations)"
    )


if __name__ == "__main__":
    main()
