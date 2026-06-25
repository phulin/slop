from __future__ import annotations

import argparse
import csv
import heapq
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from slop_sftdiv.generation_text import (
    FeatureTextMode,
    feature_text_for_mode,
    strip_chat_role_markers,
)
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


@dataclass(frozen=True)
class SAEDoc:
    source: str
    doc_id: str
    text: str
    label: int


@dataclass(frozen=True)
class ActivationCache:
    activations: torch.Tensor
    document_rows: list[dict[str, Any]]
    doc_indices: torch.Tensor
    token_indices: torch.Tensor

    @property
    def has_provenance(self) -> bool:
        return (
            bool(self.document_rows)
            and self.doc_indices.numel() == self.activations.shape[0]
            and self.token_indices.numel() == self.activations.shape[0]
        )


class BatchTopKSAE(nn.Module):
    """Sparse autoencoder with a BatchTopK activation bottleneck.

    `k` is interpreted as the average number of active latents per activation
    vector. For a token batch of N vectors, the largest N*k activations across
    the full batch-latent matrix survive and every other activation is zeroed.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        latent_dim: int,
        k: int,
        decoder_bias: bool = True,
        init_mode: str = "kaiming",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if k <= 0:
            raise ValueError("k must be positive")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.k = k
        self.init_mode = init_mode
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=decoder_bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_mode not in {"kaiming", "tied"}:
            raise ValueError(f"unknown SAE init_mode {self.init_mode!r}")
        if self.init_mode == "tied":
            nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)
            self.normalize_decoder_weights()
            with torch.no_grad():
                self.encoder.weight.copy_(self.decoder.weight.T)
        else:
            nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
            nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        if self.decoder.bias is not None:
            nn.init.zeros_(self.decoder.bias)
        self.normalize_decoder_weights()

    def encode_dense(self, activations: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(activations))

    def batch_topk(self, dense_codes: torch.Tensor) -> torch.Tensor:
        if dense_codes.ndim != 2:
            raise ValueError("BatchTopKSAE expects a 2D activation matrix")
        keep = min(dense_codes.numel(), max(1, dense_codes.shape[0] * self.k))
        if keep >= dense_codes.numel():
            return dense_codes
        flat_codes = dense_codes.flatten()
        top_values, top_indexes = torch.topk(flat_codes, keep)
        sparse_flat = torch.zeros_like(flat_codes)
        sparse_flat.scatter_(0, top_indexes, top_values)
        return sparse_flat.reshape_as(dense_codes)

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        return self.batch_topk(self.encode_dense(activations))

    def encode_with_dense(self, activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dense_codes = self.encode_dense(activations)
        return self.batch_topk(dense_codes), dense_codes

    def decode(self, sparse_codes: torch.Tensor) -> torch.Tensor:
        return self.decoder(sparse_codes)

    def forward(self, activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sparse_codes = self.encode(activations)
        return self.decode(sparse_codes), sparse_codes

    @torch.no_grad()
    def normalize_decoder_weights(self) -> None:
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-6)
        self.decoder.weight.div_(norms)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and score a BatchTopK SAE on the fine-tuned Phase 4 ModernBERT "
            "detector's penultimate-layer activations."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/phase4/modernbert_detector_combined_v2_clean/checkpoint"),
        help="Fine-tuned ModernBERT sequence-classifier checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--load-sae",
        type=Path,
        default=None,
        help="Load an existing phase4_batchtopk_sae.pt and score it instead of training a new SAE.",
    )
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
        help="Generation JSONL included in SAE training and LLM-logit scoring with label 1.",
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
    parser.add_argument("--collect-batch-size", type=int, default=8)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--pad-to-max-length", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-activations", type=int, default=200_000)
    parser.add_argument(
        "--activation-cache",
        type=Path,
        default=None,
        help="Optional shared activation cache path. Existing caches are reused unless --refresh-activation-cache is set.",
    )
    parser.add_argument("--refresh-activation-cache", action="store_true")
    parser.add_argument(
        "--activation-cache-train-chunks",
        type=int,
        default=4,
        help=(
            "For reused Parquet activation caches, train by loading this many row-group "
            "chunks one at a time and shuffling within each chunk. Set to 1 for the old "
            "full-cache tensor load."
        ),
    )
    parser.add_argument("--latent-dim", type=int, default=4096)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument(
        "--matryoshka-prefixes",
        default="",
        help=(
            "Comma-separated latent prefix sizes for Matryoshka SAE training, "
            "for example 256,512,1024,2048,4096. Empty disables nested-prefix loss."
        ),
    )
    parser.add_argument(
        "--matryoshka-loss-weights",
        default="",
        help=(
            "Optional comma-separated weights for --matryoshka-prefixes. "
            "Defaults to equal weights."
        ),
    )
    parser.add_argument(
        "--init-mode",
        choices=["kaiming", "tied"],
        default="kaiming",
        help="SAE initialization. tied initializes encoder rows from normalized decoder columns.",
    )
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "muon-adamw"],
        default="adamw",
        help="SAE optimizer. muon-adamw uses torch.optim.Muon for 2D weights and AdamW for biases.",
    )
    parser.add_argument(
        "--muon-learning-rate",
        type=float,
        default=None,
        help="Muon LR for 2D SAE weights. Defaults to --learning-rate when omitted.",
    )
    parser.add_argument(
        "--adamw-learning-rate",
        type=float,
        default=None,
        help="AdamW LR for non-2D SAE params. Defaults to --learning-rate when omitted.",
    )
    parser.add_argument("--muon-weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "wsd", "warmup_linear_decay"],
        default="constant",
        help=(
            "SAE learning-rate schedule. WSD is warmup/stable/decay; "
            "warmup_linear_decay linearly decays from 1.0 to --min-lr-ratio after warmup."
        ),
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--decay-ratio", type=float, default=0.1)
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--auxk-coefficient", type=float, default=0.0)
    parser.add_argument("--auxk", type=int, default=256)
    parser.add_argument(
        "--auxk-dead-steps",
        type=int,
        default=500,
        help="A latent is AuxK-eligible after this many optimizer steps without activation.",
    )
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--compile-sae", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-detector", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-mode", choices=["default", "reduce-overhead", "max-autotune"], default="default")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-latent-scoring", action="store_true")
    parser.add_argument(
        "--target-class",
        action="append",
        default=[],
        help=(
            "Detector class name/id to treat as the AI target for latent ablation scoring. "
            "Defaults to all non-human labels when label metadata is present."
        ),
    )
    parser.add_argument("--score-top-latents", type=int, default=128)
    parser.add_argument("--max-examples-per-latent", type=int, default=8)
    parser.add_argument("--example-context-tokens", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="phase4-sae")
    parser.add_argument("--wandb-job-type", default="train-batchtopk-sae")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    return parser


def _parse_spec(raw: str, *, default_field: str) -> tuple[str, Path, str]:
    if "=" not in raw:
        raise ValueError(f"expected LABEL=PATH[:FIELD], got {raw!r}")
    label, rest = raw.split("=", 1)
    path = Path(rest)
    if path.exists():
        return label, path, default_field
    if ":" in rest:
        path_text, field = rest.rsplit(":", 1)
        return label, Path(path_text), field
    return label, Path(rest), default_field


def _load_jsonl_docs(
    specs: list[str],
    *,
    label: int,
    default_field: str,
    max_docs_per_source: int,
    feature_text_mode: str,
) -> list[SAEDoc]:
    docs: list[SAEDoc] = []
    for raw in specs:
        source, path, field = _parse_spec(raw, default_field=default_field)
        loaded = 0
        with path.open(encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if loaded >= max_docs_per_source:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                text = str(row.get(field, "") or "")
                if label == 1:
                    text = strip_chat_role_markers(
                        feature_text_for_mode(text, cast(FeatureTextMode, feature_text_mode))
                    )
                text = text.strip()
                if not text:
                    continue
                docs.append(
                    SAEDoc(
                        source=source,
                        doc_id=str(
                            row.get("record_id")
                            or row.get("doc_id")
                            or row.get("phase2_prompt_id")
                            or row_index
                        ),
                        text=text,
                        label=label,
                    )
                )
                loaded += 1
    return docs


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _load_parquet_docs(
    path: Path,
    *,
    max_docs_per_source: int,
    seed: int,
) -> list[SAEDoc]:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    required = {"role", "model", "text"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"dataset parquet is missing columns: {missing}")
    frame = frame[frame["role"].isin(["human_continuation", "llm_continuation"])].copy()
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame["model"] = frame["model"].fillna("").astype(str).str.strip()
    if "record_id" not in frame.columns:
        frame["record_id"] = [str(index) for index in range(len(frame))]
    docs: list[SAEDoc] = []
    rng = random.Random(seed)
    for source, source_frame in frame.groupby("model", sort=True):
        rows = list(source_frame.itertuples(index=False))
        rng.shuffle(rows)
        loaded = 0
        for row in rows:
            if loaded >= max_docs_per_source:
                break
            text = _string_or_empty(getattr(row, "text"))
            role = _string_or_empty(getattr(row, "role"))
            if not text:
                continue
            label = 0 if role == "human_continuation" else 1
            docs.append(
                SAEDoc(
                    source=_string_or_empty(source),
                    doc_id=_string_or_empty(getattr(row, "record_id", loaded)),
                    text=text,
                    label=label,
                )
            )
            loaded += 1
    return docs


def _load_docs(args: argparse.Namespace) -> list[SAEDoc]:
    docs = [
        *_load_jsonl_docs(
            args.reference_jsonl,
            label=0,
            default_field="text",
            max_docs_per_source=args.max_docs_per_source,
            feature_text_mode=args.feature_text_mode,
        ),
        *_load_jsonl_docs(
            args.generation_jsonl,
            label=1,
            default_field="generation",
            max_docs_per_source=args.max_docs_per_source,
            feature_text_mode=args.feature_text_mode,
        ),
    ]
    if args.dataset_parquet is not None:
        docs.extend(
            _load_parquet_docs(
                args.dataset_parquet,
                max_docs_per_source=args.max_docs_per_source,
                seed=args.seed,
            )
        )
    if not docs:
        raise ValueError("provide at least one --reference-jsonl, --generation-jsonl, or --dataset-parquet input")
    random.Random(args.seed).shuffle(docs)
    return docs


def _load_detector(
    checkpoint: Path,
    *,
    device: torch.device,
) -> tuple[Any, Any]:
    tokenizer = cast(Any, AutoTokenizer.from_pretrained(checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.eval()
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("expected a ModernBERT sequence classifier with model.layers")
    if len(model.model.layers) < 1:
        raise ValueError("ModernBERT model has no encoder layers")
    return model, tokenizer


def _token_batches(items: list[SAEDoc], batch_size: int) -> list[list[SAEDoc]]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _encode_docs(
    tokenizer: Any,
    docs: list[SAEDoc],
    *,
    max_length: int,
    device: torch.device,
    pad_to_max_length: bool = False,
) -> dict[str, Any]:
    encoded = tokenizer(
        [doc.text for doc in docs],
        padding="max_length" if pad_to_max_length else True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }


def _activation_mask(encoded: dict[str, Any]) -> torch.Tensor:
    mask = encoded["attention_mask"].bool()
    special = encoded.get("special_tokens_mask")
    if isinstance(special, torch.Tensor):
        mask = mask & ~special.bool()
    return mask


def _activation_document_rows(docs: list[SAEDoc]) -> list[dict[str, Any]]:
    return [
        {
            "document_index": index,
            "source": doc.source,
            "doc_id": doc.doc_id,
            "label": int(doc.label),
        }
        for index, doc in enumerate(docs)
    ]


def _activation_row_indexes(
    *,
    token_mask: torch.Tensor,
    batch_start_doc_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = token_mask.detach().cpu().bool()
    if mask.ndim != 2:
        raise ValueError(f"activation mask must be 2D, got shape {tuple(mask.shape)}")
    batch_docs, doc_token_count = mask.shape
    flat_doc_positions = (
        torch.arange(batch_docs, dtype=torch.long)
        .unsqueeze(1)
        .expand(-1, doc_token_count)
        .reshape(-1)
    )
    flat_token_positions = (
        torch.arange(doc_token_count, dtype=torch.long)
        .unsqueeze(0)
        .expand(batch_docs, -1)
        .reshape(-1)
    )
    flat_mask = mask.reshape(-1)
    return flat_doc_positions[flat_mask] + batch_start_doc_index, flat_token_positions[flat_mask]


def _validate_activation_cache(cache: ActivationCache, *, path: Path | None = None) -> ActivationCache:
    where = f" {path}" if path is not None else ""
    if cache.activations.ndim != 2:
        raise ValueError(
            f"activation cache{where} must be a 2D tensor, got shape {tuple(cache.activations.shape)}"
        )
    if cache.doc_indices.numel() == 0 and cache.token_indices.numel() == 0 and not cache.document_rows:
        return ActivationCache(
            activations=cache.activations.contiguous(),
            document_rows=[],
            doc_indices=torch.empty(0, dtype=torch.long),
            token_indices=torch.empty(0, dtype=torch.long),
        )
    if cache.doc_indices.ndim != 1 or cache.token_indices.ndim != 1:
        raise ValueError(f"activation cache{where} provenance indexes must be 1D tensors")
    if cache.doc_indices.numel() != cache.activations.shape[0]:
        raise ValueError(
            f"activation cache{where} has {cache.activations.shape[0]} activation rows "
            f"but {cache.doc_indices.numel()} doc indexes"
        )
    if cache.token_indices.numel() != cache.activations.shape[0]:
        raise ValueError(
            f"activation cache{where} has {cache.activations.shape[0]} activation rows "
            f"but {cache.token_indices.numel()} token indexes"
        )
    if cache.document_rows:
        max_doc_index = int(cache.doc_indices.max().item()) if cache.doc_indices.numel() else -1
        if max_doc_index >= len(cache.document_rows):
            raise ValueError(
                f"activation cache{where} references document index {max_doc_index} "
                f"but only stores {len(cache.document_rows)} documents"
            )
    return ActivationCache(
        activations=cache.activations.contiguous(),
        document_rows=list(cache.document_rows),
        doc_indices=cache.doc_indices.to(dtype=torch.long).contiguous(),
        token_indices=cache.token_indices.to(dtype=torch.long).contiguous(),
    )


@torch.inference_mode()
def collect_penultimate_activation_cache(
    *,
    model: Any,
    tokenizer: Any,
    docs: list[SAEDoc],
    max_length: int,
    batch_size: int,
    max_activations: int,
    device: torch.device,
    progress_every: int,
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
        )
        hidden = outputs.hidden_states[-2]
        token_mask = _activation_mask(encoded)
        vectors = hidden[token_mask].detach().float().cpu()
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
            print(
                f"Collected {collected:,}/{max_activations:,} activation vectors "
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
def collect_penultimate_activations(
    *,
    model: Any,
    tokenizer: Any,
    docs: list[SAEDoc],
    max_length: int,
    batch_size: int,
    max_activations: int,
    device: torch.device,
    progress_every: int,
    pad_to_max_length: bool = False,
) -> torch.Tensor:
    return collect_penultimate_activation_cache(
        model=model,
        tokenizer=tokenizer,
        docs=docs,
        max_length=max_length,
        batch_size=batch_size,
        max_activations=max_activations,
        device=device,
        progress_every=progress_every,
        pad_to_max_length=pad_to_max_length,
    ).activations


class _ActivationSubsetDataset(Dataset):
    def __init__(self, activations: torch.Tensor, indexes: torch.Tensor) -> None:
        self.activations = activations
        self.indexes = indexes.to(dtype=torch.long).contiguous()

    def __len__(self) -> int:
        return int(self.indexes.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        return (self.activations[int(self.indexes[index])],)


def _activation_dataset(
    activations: torch.Tensor,
    indexes: torch.Tensor,
) -> Dataset:
    if indexes.numel() == activations.shape[0] and torch.equal(
        indexes,
        torch.arange(indexes.numel(), dtype=indexes.dtype),
    ):
        return TensorDataset(activations)
    return _ActivationSubsetDataset(activations, indexes)


@dataclass(frozen=True)
class ParquetActivationChunk:
    index: int
    row_groups: list[int]
    row_count: int


class ParquetActivationChunkSource:
    def __init__(self, path: Path, *, chunks: int, batch_size: int = 8192) -> None:
        if chunks <= 0:
            raise ValueError("--activation-cache-train-chunks must be positive")
        self.path = path
        self.batch_size = int(batch_size)
        self.parquet_file = pq.ParquetFile(path)
        if "activation" not in self.parquet_file.schema_arrow.names:
            raise ValueError(f"activation cache {path} is missing an activation column")
        activation_type = self.parquet_file.schema_arrow.field("activation").type
        if not isinstance(activation_type, pa.FixedSizeListType):
            raise ValueError(f"activation cache {path} must use a fixed-size-list activation column")
        self.hidden_dim = int(activation_type.list_size)
        self.row_count = int(self.parquet_file.metadata.num_rows)
        self.row_group_rows = [
            int(self.parquet_file.metadata.row_group(index).num_rows)
            for index in range(self.parquet_file.num_row_groups)
        ]
        self.chunks = self._split_row_groups(max(1, int(chunks)))

    def _split_row_groups(self, chunks: int) -> list[ParquetActivationChunk]:
        target_rows = max(1, (self.row_count + chunks - 1) // chunks)
        split_chunks: list[ParquetActivationChunk] = []
        current_groups: list[int] = []
        current_rows = 0
        for row_group, rows in enumerate(self.row_group_rows):
            if current_groups and len(split_chunks) + 1 < chunks and current_rows + rows > target_rows:
                split_chunks.append(
                    ParquetActivationChunk(
                        index=len(split_chunks),
                        row_groups=current_groups,
                        row_count=current_rows,
                    )
                )
                current_groups = []
                current_rows = 0
            current_groups.append(row_group)
            current_rows += rows
        if current_groups:
            split_chunks.append(
                ParquetActivationChunk(
                    index=len(split_chunks),
                    row_groups=current_groups,
                    row_count=current_rows,
                )
            )
        return split_chunks

    def load_chunk(self, chunk: ParquetActivationChunk) -> torch.Tensor:
        activations = torch.empty((chunk.row_count, self.hidden_dim), dtype=torch.float16)
        offset = 0
        for batch in self.parquet_file.iter_batches(
            columns=["activation"],
            row_groups=chunk.row_groups,
            batch_size=self.batch_size,
        ):
            activation_column = batch.column("activation")
            if not isinstance(activation_column, pa.FixedSizeListArray):
                activation_column = activation_column.combine_chunks()
            values = activation_column.values.to_numpy(zero_copy_only=False)
            next_offset = offset + len(activation_column)
            activations[offset:next_offset].copy_(
                torch.from_numpy(values.reshape(len(activation_column), self.hidden_dim).copy())
            )
            offset = next_offset
        if offset != chunk.row_count:
            raise ValueError(
                f"loaded {offset} activation rows from chunk {chunk.index}, expected {chunk.row_count}"
            )
        return activations


def _split_activation_indexes(
    activation_count: int,
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError("--eval-fraction must be >= 0 and < 1")
    all_indexes = torch.arange(activation_count, dtype=torch.long)
    if eval_fraction == 0.0 or activation_count < 2:
        return all_indexes, all_indexes[:0]
    generator = torch.Generator()
    generator.manual_seed(seed)
    permutation = torch.randperm(activation_count, generator=generator)
    eval_count = max(1, int(activation_count * eval_fraction))
    eval_indexes = permutation[:eval_count]
    train_indexes = permutation[eval_count:]
    return train_indexes.contiguous(), eval_indexes.contiguous()


def _compile_module(module: nn.Module, *, enabled: bool, mode: str, device: torch.device) -> nn.Module:
    if not enabled or device.type != "cuda":
        return module
    if not hasattr(torch, "compile"):
        return module
    print(f"Compiling {module.__class__.__name__} with torch.compile(mode={mode})", flush=True)
    return cast(nn.Module, torch.compile(module, mode=mode))


def _reset_torch_compile() -> None:
    dynamo = getattr(torch, "_dynamo", None)
    reset = getattr(dynamo, "reset", None)
    if callable(reset):
        reset()


@torch.inference_mode()
def _evaluate_sae_mse(
    *,
    sae_model: nn.Module,
    activations: torch.Tensor,
    indexes: torch.Tensor,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> float | None:
    if activations.numel() == 0 or indexes.numel() == 0:
        return None
    loader = DataLoader(
        _activation_dataset(activations, indexes),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    total_sse = 0.0
    total_values = 0
    was_training = sae_model.training
    sae_model.eval()
    for (batch_cpu,) in loader:
        batch = batch_cpu.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
        reconstruction, _sparse_codes = sae_model(batch)
        residual = reconstruction.float() - batch
        total_sse += float(residual.pow(2).sum().detach().cpu())
        total_values += int(residual.numel())
    if was_training:
        sae_model.train()
    return total_sse / max(1, total_values)


def _wsd_lr_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    decay_steps: int,
    min_lr_ratio: float,
) -> float:
    if step <= 0:
        return 0.0 if warmup_steps > 0 else 1.0
    if warmup_steps > 0 and step <= warmup_steps:
        return step / warmup_steps
    decay_start = max(warmup_steps, total_steps - decay_steps)
    if decay_steps > 0 and step > decay_start:
        remaining = max(0, total_steps - step)
        decay_progress = remaining / decay_steps
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay_progress
    return 1.0


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    total_steps: int,
) -> tuple[torch.optim.lr_scheduler.LambdaLR | None, dict[str, Any]]:
    if args.warmup_ratio < 0.0 or args.decay_ratio < 0.0:
        raise ValueError("--warmup-ratio and --decay-ratio must be non-negative")
    if args.lr_schedule == "wsd" and args.warmup_ratio + args.decay_ratio > 1.0:
        raise ValueError("--warmup-ratio + --decay-ratio must be <= 1")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be between 0 and 1")
    if args.lr_schedule == "constant":
        return None, {
            "lr_schedule": "constant",
            "warmup_steps": 0,
            "stable_steps": total_steps,
            "decay_steps": 0,
            "min_lr_ratio": 1.0,
        }
    if args.lr_schedule == "warmup_linear_decay":
        warmup_steps = int(total_steps * args.warmup_ratio)
        decay_steps = max(0, total_steps - warmup_steps)

        def lr_lambda(step: int) -> float:
            if step <= 0:
                return 0.0 if warmup_steps > 0 else 1.0
            if warmup_steps > 0 and step <= warmup_steps:
                return step / warmup_steps
            if decay_steps <= 0:
                return args.min_lr_ratio
            decay_progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
            return 1.0 - (1.0 - args.min_lr_ratio) * decay_progress

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler, {
            "lr_schedule": "warmup_linear_decay",
            "warmup_steps": warmup_steps,
            "stable_steps": 0,
            "decay_steps": decay_steps,
            "min_lr_ratio": args.min_lr_ratio,
        }
    warmup_steps = int(total_steps * args.warmup_ratio)
    decay_steps = int(total_steps * args.decay_ratio)
    stable_steps = max(0, total_steps - warmup_steps - decay_steps)

    def lr_lambda(step: int) -> float:
        return _wsd_lr_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler, {
        "lr_schedule": "wsd",
        "warmup_steps": warmup_steps,
        "stable_steps": stable_steps,
        "decay_steps": decay_steps,
        "min_lr_ratio": args.min_lr_ratio,
    }


def _build_sae_optimizers(
    sae: BatchTopKSAE,
    *,
    args: argparse.Namespace,
) -> tuple[list[torch.optim.Optimizer], dict[str, Any]]:
    optimizer_name = getattr(args, "optimizer", "adamw")
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            sae.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        return [optimizer], {
            "optimizer": "adamw",
            "learning_rate": args.learning_rate,
            "adamw_learning_rate": args.learning_rate,
            "muon_learning_rate": None,
        }
    if optimizer_name != "muon-adamw":
        raise ValueError(f"unknown SAE optimizer {optimizer_name!r}")
    muon_cls = getattr(torch.optim, "Muon", None)
    if muon_cls is None:
        raise RuntimeError("torch.optim.Muon is unavailable in this PyTorch build")
    muon_lr = args.muon_learning_rate if args.muon_learning_rate is not None else args.learning_rate
    adamw_lr = args.adamw_learning_rate if args.adamw_learning_rate is not None else args.learning_rate
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    for _name, parameter in sae.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 2:
            muon_params.append(parameter)
        else:
            adamw_params.append(parameter)
    optimizers: list[torch.optim.Optimizer] = []
    if muon_params:
        optimizers.append(
            muon_cls(
                muon_params,
                lr=muon_lr,
                weight_decay=args.muon_weight_decay,
                momentum=args.muon_momentum,
                ns_steps=args.muon_ns_steps,
            )
        )
    if adamw_params:
        optimizers.append(
            torch.optim.AdamW(
                adamw_params,
                lr=adamw_lr,
                weight_decay=args.weight_decay,
            )
        )
    if not optimizers:
        raise ValueError("SAE has no trainable parameters")
    return optimizers, {
        "optimizer": "muon-adamw",
        "learning_rate": None,
        "muon_learning_rate": muon_lr,
        "adamw_learning_rate": adamw_lr,
        "muon_weight_decay": args.muon_weight_decay,
        "muon_momentum": args.muon_momentum,
        "muon_ns_steps": args.muon_ns_steps,
        "muon_parameter_count": sum(parameter.numel() for parameter in muon_params),
        "adamw_parameter_count": sum(parameter.numel() for parameter in adamw_params),
    }


def _auxk_loss(
    *,
    sae: BatchTopKSAE,
    dense_codes: torch.Tensor,
    residual: torch.Tensor,
    dead_latents: torch.Tensor,
    auxk: int,
) -> tuple[torch.Tensor, int]:
    dead_indexes = torch.nonzero(dead_latents, as_tuple=False).flatten()
    if auxk <= 0 or dead_indexes.numel() == 0:
        return dense_codes.new_zeros(()), int(dead_indexes.numel())
    dead_codes = dense_codes[:, dead_indexes]
    keep = min(dead_codes.numel(), max(1, dense_codes.shape[0] * min(auxk, dead_indexes.numel())))
    if dead_codes.numel() == 0 or keep <= 0:
        return dense_codes.new_zeros(()), int(dead_indexes.numel())
    top_values, top_indexes = torch.topk(dead_codes.flatten(), keep)
    aux_flat = torch.zeros_like(dead_codes.flatten())
    aux_flat.scatter_(0, top_indexes, top_values)
    aux_dead_codes = aux_flat.reshape_as(dead_codes)
    aux_codes = torch.zeros_like(dense_codes)
    aux_codes[:, dead_indexes] = aux_dead_codes
    aux_reconstruction = F.linear(aux_codes, sae.decoder.weight)
    return F.mse_loss(aux_reconstruction, residual.detach()), int(dead_indexes.numel())


def _parse_int_list(raw: str) -> list[int]:
    if not str(raw or "").strip():
        return []
    values: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def _parse_float_list(raw: str) -> list[float]:
    if not str(raw or "").strip():
        return []
    values: list[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return values


def _matryoshka_config(args: argparse.Namespace) -> tuple[list[int], list[float]]:
    prefixes = _parse_int_list(getattr(args, "matryoshka_prefixes", ""))
    if not prefixes:
        return [], []
    latent_dim = int(args.latent_dim)
    prefixes = sorted(set(prefixes))
    if prefixes[0] <= 0:
        raise ValueError("--matryoshka-prefixes must be positive")
    if prefixes[-1] > latent_dim:
        raise ValueError("--matryoshka-prefixes cannot exceed --latent-dim")
    if prefixes[-1] != latent_dim:
        prefixes.append(latent_dim)
    weights = _parse_float_list(getattr(args, "matryoshka_loss_weights", ""))
    if weights and len(weights) != len(prefixes):
        raise ValueError("--matryoshka-loss-weights must match --matryoshka-prefixes length after appending latent_dim")
    if not weights:
        weights = [1.0 / len(prefixes)] * len(prefixes)
    else:
        total = sum(weights)
        if total <= 0.0:
            raise ValueError("--matryoshka-loss-weights must sum to a positive value")
        weights = [weight / total for weight in weights]
    return prefixes, weights


def _matryoshka_reconstruction_loss(
    *,
    sae: BatchTopKSAE,
    batch: torch.Tensor,
    prefixes: list[int],
    weights: list[float],
) -> tuple[torch.Tensor, dict[str, float]]:
    dense_codes = sae.encode_dense(batch)
    sparse_codes = sae.batch_topk(dense_codes)
    losses: list[torch.Tensor] = []
    metrics: dict[str, float] = {}
    for prefix, weight in zip(prefixes, weights, strict=True):
        prefix_codes = sparse_codes[:, :prefix]
        reconstruction = F.linear(
            prefix_codes,
            sae.decoder.weight[:, :prefix],
            sae.decoder.bias,
        )
        prefix_mse = F.mse_loss(reconstruction, batch)
        losses.append(float(weight) * prefix_mse)
        metrics[f"matryoshka_mse_{prefix}"] = float(prefix_mse.detach().cpu())
    return torch.stack(losses).sum(), metrics


def train_sae(
    *,
    activations: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    wandb_run: Any | None = None,
) -> tuple[BatchTopKSAE, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_indexes, eval_indexes = _split_activation_indexes(
        int(activations.shape[0]),
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    sae = BatchTopKSAE(
        input_dim=int(activations.shape[1]),
        latent_dim=args.latent_dim,
        k=args.k,
        init_mode=getattr(args, "init_mode", "kaiming"),
    ).to(device)
    sae_model = _compile_module(
        sae,
        enabled=bool(getattr(args, "compile_sae", False)),
        mode=getattr(args, "compile_mode", "default"),
        device=device,
    )
    sae_compile_fallback = False
    optimizers, optimizer_summary = _build_sae_optimizers(sae, args=args)
    dataset = _activation_dataset(activations, train_indexes)
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    total_steps = max(1, len(loader) * args.epochs)
    scheduler_pairs: list[tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR | None]] = []
    scheduler_summary: dict[str, Any] = {}
    for optimizer_index, optimizer in enumerate(optimizers):
        scheduler, one_scheduler_summary = _build_lr_scheduler(
            optimizer,
            args=args,
            total_steps=total_steps,
        )
        scheduler_pairs.append((optimizer, scheduler))
        if optimizer_index == 0:
            scheduler_summary = one_scheduler_summary
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    start = time.time()
    step = 0
    last_active_step = torch.zeros(args.latent_dim, dtype=torch.long, device=device)
    auxk_coefficient = float(getattr(args, "auxk_coefficient", 0.0))
    auxk = int(getattr(args, "auxk", 0))
    auxk_dead_steps = max(1, int(getattr(args, "auxk_dead_steps", 500)))
    matryoshka_prefixes, matryoshka_weights = _matryoshka_config(args)
    sae_model.train()
    for epoch in range(1, args.epochs + 1):
        for (batch_cpu,) in loader:
            step += 1
            batch = batch_cpu.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            try:
                reconstruction, sparse_codes = sae_model(batch)
            except Exception as exc:
                if sae_model is sae:
                    raise
                print(
                    f"torch.compile SAE forward failed ({type(exc).__name__}: {exc}); falling back to eager SAE",
                    flush=True,
                )
                _reset_torch_compile()
                sae_compile_fallback = True
                sae_model = sae
                sae_model.train()
                reconstruction, sparse_codes = sae_model(batch)
            residual = reconstruction - batch
            mse = residual.pow(2).mean()
            matryoshka_metrics: dict[str, float] = {}
            reconstruction_loss = mse
            if matryoshka_prefixes:
                reconstruction_loss, matryoshka_metrics = _matryoshka_reconstruction_loss(
                    sae=sae,
                    batch=batch,
                    prefixes=matryoshka_prefixes,
                    weights=matryoshka_weights,
                )
            active_latents = (sparse_codes.detach() > 0).any(dim=0)
            last_active_step[active_latents] = step
            dead_latents = (step - last_active_step) >= auxk_dead_steps
            aux_loss = batch.new_zeros(())
            dead_latent_count = int(dead_latents.sum().detach().cpu())
            if auxk_coefficient > 0.0:
                dense_codes = sae.encode_dense(batch)
                aux_loss, dead_latent_count = _auxk_loss(
                    sae=sae,
                    dense_codes=dense_codes,
                    residual=residual,
                    dead_latents=dead_latents,
                    auxk=auxk,
                )
            loss = reconstruction_loss + auxk_coefficient * aux_loss
            loss.backward()
            for optimizer, scheduler in scheduler_pairs:
                optimizer.step()
                if scheduler is None:
                    continue
                scheduler.step()
            sae.normalize_decoder_weights()
            if step == 1 or step % args.progress_every == 0:
                active_per_vector = (sparse_codes > 0).float().sum(dim=1).mean()
                current_lrs = [
                    float(optimizer.param_groups[0]["lr"])
                    for optimizer, _scheduler in scheduler_pairs
                    if optimizer.param_groups
                ]
                current_lr = current_lrs[0] if current_lrs else 0.0
                row = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "mse": float(mse.detach().cpu()),
                    "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
                    "auxk_loss": float(aux_loss.detach().cpu()),
                    "auxk_coefficient": auxk_coefficient,
                    "learning_rate": current_lr,
                    "learning_rates": "|".join(f"{lr:.8g}" for lr in current_lrs),
                    "active_latents_per_vector": float(active_per_vector.detach().cpu()),
                    "unique_active_latents": int(active_latents.sum().detach().cpu()),
                    "dead_latents": dead_latent_count,
                    "elapsed_seconds": time.time() - start,
                }
                row.update(matryoshka_metrics)
                train_rows.append(row)
                if wandb_run is not None:
                    wandb_payload = {
                            "train/loss": row["loss"],
                            "train/mse": row["mse"],
                            "train/reconstruction_loss": row["reconstruction_loss"],
                            "train/auxk_loss": row["auxk_loss"],
                            "train/learning_rate": current_lr,
                            "train/dead_latents": dead_latent_count,
                            "train/unique_active_latents": row["unique_active_latents"],
                            "train/active_latents_per_vector": row[
                                "active_latents_per_vector"
                            ],
                            "train/epoch": epoch,
                            "train/elapsed_seconds": row["elapsed_seconds"],
                        }
                    for key, value in matryoshka_metrics.items():
                        wandb_payload[f"train/{key}"] = value
                    wandb_run.log(wandb_payload, step=step)
                print(
                    f"SAE step {step}: loss={row['loss']:.6f} "
                    f"mse={row['mse']:.6f} recon={row['reconstruction_loss']:.6f} "
                    f"auxk={row['auxk_loss']:.6f} "
                    f"active/vector={row['active_latents_per_vector']:.2f} "
                    f"unique={row['unique_active_latents']} dead={row['dead_latents']} "
                    f"lr={current_lr:.3g}",
                    flush=True,
                )
        eval_mse = _evaluate_sae_mse(
            sae_model=sae_model,
            activations=activations,
            indexes=eval_indexes,
            batch_size=args.train_batch_size,
            device=device,
            num_workers=args.num_workers,
        )
        if eval_mse is not None:
            row = {
                "step": step,
                "epoch": epoch,
                "eval_mse": eval_mse,
                "elapsed_seconds": time.time() - start,
            }
            eval_rows.append(row)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/mse": eval_mse,
                        "eval/epoch": epoch,
                        "eval/elapsed_seconds": row["elapsed_seconds"],
                    },
                    step=step,
                )
            print(f"SAE epoch {epoch}: eval_mse={eval_mse:.6f}", flush=True)
    train_summary = {
        "compile_sae_requested": bool(getattr(args, "compile_sae", False)),
        "compile_sae_fallback": sae_compile_fallback,
        "auxk_coefficient": auxk_coefficient,
        "auxk": auxk,
        "auxk_dead_steps": auxk_dead_steps,
        "matryoshka_prefixes": matryoshka_prefixes,
        "matryoshka_loss_weights": matryoshka_weights,
        **optimizer_summary,
        **scheduler_summary,
    }
    return sae, train_rows, eval_rows, train_summary


def train_sae_from_parquet_chunks(
    *,
    source: ParquetActivationChunkSource,
    args: argparse.Namespace,
    device: torch.device,
    wandb_run: Any | None = None,
) -> tuple[BatchTopKSAE, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sae = BatchTopKSAE(
        input_dim=source.hidden_dim,
        latent_dim=args.latent_dim,
        k=args.k,
        init_mode=getattr(args, "init_mode", "kaiming"),
    ).to(device)
    sae_model = _compile_module(
        sae,
        enabled=bool(getattr(args, "compile_sae", False)),
        mode=getattr(args, "compile_mode", "default"),
        device=device,
    )
    sae_compile_fallback = False
    optimizers, optimizer_summary = _build_sae_optimizers(sae, args=args)
    steps_per_epoch = 0
    for chunk in source.chunks:
        train_rows_in_chunk = chunk.row_count
        if args.eval_fraction > 0.0 and chunk.row_count >= 2:
            train_rows_in_chunk -= max(1, int(chunk.row_count * args.eval_fraction))
        steps_per_epoch += max(1, (max(1, train_rows_in_chunk) + args.train_batch_size - 1) // args.train_batch_size)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler_pairs: list[tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR | None]] = []
    scheduler_summary: dict[str, Any] = {}
    for optimizer_index, optimizer in enumerate(optimizers):
        scheduler, one_scheduler_summary = _build_lr_scheduler(
            optimizer,
            args=args,
            total_steps=total_steps,
        )
        scheduler_pairs.append((optimizer, scheduler))
        if optimizer_index == 0:
            scheduler_summary = one_scheduler_summary
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    start = time.time()
    step = 0
    last_active_step = torch.zeros(args.latent_dim, dtype=torch.long, device=device)
    auxk_coefficient = float(getattr(args, "auxk_coefficient", 0.0))
    auxk = int(getattr(args, "auxk", 0))
    auxk_dead_steps = max(1, int(getattr(args, "auxk_dead_steps", 500)))
    matryoshka_prefixes, matryoshka_weights = _matryoshka_config(args)
    sae_model.train()
    for epoch in range(1, args.epochs + 1):
        chunk_order = list(source.chunks)
        random.Random(args.seed + epoch).shuffle(chunk_order)
        epoch_eval_sse = 0.0
        epoch_eval_values = 0
        for chunk_position, chunk in enumerate(chunk_order, start=1):
            print(
                f"Loading activation cache chunk {chunk_position}/{len(chunk_order)} "
                f"(chunk={chunk.index}, rows={chunk.row_count}, row_groups={len(chunk.row_groups)})",
                flush=True,
            )
            chunk_activations = source.load_chunk(chunk)
            train_indexes, eval_indexes = _split_activation_indexes(
                int(chunk_activations.shape[0]),
                eval_fraction=args.eval_fraction,
                seed=args.seed + epoch * 1_000_003 + chunk.index,
            )
            loader = DataLoader(
                _activation_dataset(chunk_activations, train_indexes),
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            for (batch_cpu,) in loader:
                step += 1
                batch = batch_cpu.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                try:
                    reconstruction, sparse_codes = sae_model(batch)
                except Exception as exc:
                    if sae_model is sae:
                        raise
                    print(
                        f"torch.compile SAE forward failed ({type(exc).__name__}: {exc}); falling back to eager SAE",
                        flush=True,
                    )
                    _reset_torch_compile()
                    sae_compile_fallback = True
                    sae_model = sae
                    sae_model.train()
                    reconstruction, sparse_codes = sae_model(batch)
                residual = reconstruction - batch
                mse = residual.pow(2).mean()
                matryoshka_metrics: dict[str, float] = {}
                reconstruction_loss = mse
                if matryoshka_prefixes:
                    reconstruction_loss, matryoshka_metrics = _matryoshka_reconstruction_loss(
                        sae=sae,
                        batch=batch,
                        prefixes=matryoshka_prefixes,
                        weights=matryoshka_weights,
                    )
                active_latents = (sparse_codes.detach() > 0).any(dim=0)
                last_active_step[active_latents] = step
                dead_latents = (step - last_active_step) >= auxk_dead_steps
                aux_loss = batch.new_zeros(())
                dead_latent_count = int(dead_latents.sum().detach().cpu())
                if auxk_coefficient > 0.0:
                    dense_codes = sae.encode_dense(batch)
                    aux_loss, dead_latent_count = _auxk_loss(
                        sae=sae,
                        dense_codes=dense_codes,
                        residual=residual,
                        dead_latents=dead_latents,
                        auxk=auxk,
                    )
                loss = reconstruction_loss + auxk_coefficient * aux_loss
                loss.backward()
                for optimizer, scheduler in scheduler_pairs:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                sae.normalize_decoder_weights()
                if step == 1 or step % args.progress_every == 0:
                    active_per_vector = (sparse_codes > 0).float().sum(dim=1).mean()
                    current_lrs = [
                        float(optimizer.param_groups[0]["lr"])
                        for optimizer, _scheduler in scheduler_pairs
                        if optimizer.param_groups
                    ]
                    current_lr = current_lrs[0] if current_lrs else 0.0
                    row = {
                        "step": step,
                        "epoch": epoch,
                        "chunk": chunk.index,
                        "loss": float(loss.detach().cpu()),
                        "mse": float(mse.detach().cpu()),
                        "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
                        "auxk_loss": float(aux_loss.detach().cpu()),
                        "auxk_coefficient": auxk_coefficient,
                        "learning_rate": current_lr,
                        "learning_rates": "|".join(f"{lr:.8g}" for lr in current_lrs),
                        "active_latents_per_vector": float(active_per_vector.detach().cpu()),
                        "unique_active_latents": int(active_latents.sum().detach().cpu()),
                        "dead_latents": dead_latent_count,
                        "elapsed_seconds": time.time() - start,
                    }
                    row.update(matryoshka_metrics)
                    train_rows.append(row)
                    if wandb_run is not None:
                        wandb_payload = {
                            "train/loss": row["loss"],
                            "train/mse": row["mse"],
                            "train/reconstruction_loss": row["reconstruction_loss"],
                            "train/auxk_loss": row["auxk_loss"],
                            "train/learning_rate": current_lr,
                            "train/dead_latents": dead_latent_count,
                            "train/unique_active_latents": row["unique_active_latents"],
                            "train/active_latents_per_vector": row["active_latents_per_vector"],
                            "train/epoch": epoch,
                            "train/chunk": chunk.index,
                            "train/elapsed_seconds": row["elapsed_seconds"],
                        }
                        for key, value in matryoshka_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        wandb_run.log(wandb_payload, step=step)
                    print(
                        f"SAE step {step}: loss={row['loss']:.6f} "
                        f"mse={row['mse']:.6f} recon={row['reconstruction_loss']:.6f} "
                        f"auxk={row['auxk_loss']:.6f} "
                        f"active/vector={row['active_latents_per_vector']:.2f} "
                        f"unique={row['unique_active_latents']} dead={row['dead_latents']} "
                        f"lr={current_lr:.3g}",
                        flush=True,
                    )
            if eval_indexes.numel() > 0:
                chunk_eval_mse = _evaluate_sae_mse(
                    sae_model=sae_model,
                    activations=chunk_activations,
                    indexes=eval_indexes,
                    batch_size=args.train_batch_size,
                    device=device,
                    num_workers=args.num_workers,
                )
                if chunk_eval_mse is not None:
                    epoch_eval_sse += chunk_eval_mse * int(eval_indexes.numel()) * source.hidden_dim
                    epoch_eval_values += int(eval_indexes.numel()) * source.hidden_dim
            del chunk_activations
        if epoch_eval_values > 0:
            eval_mse = epoch_eval_sse / epoch_eval_values
            row = {
                "step": step,
                "epoch": epoch,
                "eval_mse": eval_mse,
                "elapsed_seconds": time.time() - start,
            }
            eval_rows.append(row)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/mse": eval_mse,
                        "eval/epoch": epoch,
                        "eval/elapsed_seconds": row["elapsed_seconds"],
                    },
                    step=step,
                )
            print(f"SAE epoch {epoch}: eval_mse={eval_mse:.6f}", flush=True)
    train_summary = {
        "compile_sae_requested": bool(getattr(args, "compile_sae", False)),
        "compile_sae_fallback": sae_compile_fallback,
        "auxk_coefficient": auxk_coefficient,
        "auxk": auxk,
        "auxk_dead_steps": auxk_dead_steps,
        "matryoshka_prefixes": matryoshka_prefixes,
        "matryoshka_loss_weights": matryoshka_weights,
        "activation_cache_train_chunks": len(source.chunks),
        "activation_cache_chunk_rows": [chunk.row_count for chunk in source.chunks],
        **optimizer_summary,
        **scheduler_summary,
    }
    return sae, train_rows, eval_rows, train_summary


def _last_layer(model: Any) -> nn.Module:
    return cast(nn.Module, model.model.layers[-1])


@torch.inference_mode()
def _logits_with_sae(
    *,
    model: Any,
    encoded: dict[str, Any],
    sae: BatchTopKSAE,
    ablate_latent: int | None = None,
    record_codes: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    captured_codes: torch.Tensor | None = None

    def hook(
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        nonlocal captured_codes
        hidden = cast(
            torch.Tensor,
            kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0],
        )
        flat = hidden.reshape(-1, hidden.shape[-1]).float()
        sparse_codes = sae.encode(flat)
        if record_codes:
            captured_codes = sparse_codes.detach()
        if ablate_latent is not None:
            sparse_codes = sparse_codes.clone()
            sparse_codes[:, ablate_latent] = 0.0
        replacement = sae.decode(sparse_codes).reshape_as(hidden).to(hidden.dtype)
        if "hidden_states" in kwargs:
            kwargs = {**kwargs, "hidden_states": replacement}
            return args, kwargs
        return (replacement, *args[1:]), kwargs

    handle = _last_layer(model).register_forward_pre_hook(hook, with_kwargs=True)
    try:
        outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
    finally:
        handle.remove()
    return outputs.logits.detach(), captured_codes


def _token_strings(tokenizer: Any, encoded: dict[str, Any], batch_index: int) -> list[str]:
    token_ids = encoded["input_ids"][batch_index].detach().cpu().tolist()
    return list(tokenizer.convert_ids_to_tokens(token_ids))


def _context_for_token(
    tokenizer: Any,
    encoded: dict[str, Any],
    *,
    batch_index: int,
    token_index: int,
    context_tokens: int,
) -> str:
    tokens = _token_strings(tokenizer, encoded, batch_index)
    start = max(0, token_index - context_tokens)
    end = min(len(tokens), token_index + context_tokens + 1)
    return " ".join(tokenizer.convert_tokens_to_string(tokens[start:end]).split())


def _push_example(
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


def _id_to_label(model: Any) -> dict[int, str]:
    raw = getattr(getattr(model, "config", None), "id2label", {}) or {}
    labels: dict[int, str] = {}
    for key, value in dict(raw).items():
        try:
            label_id = int(key)
        except (TypeError, ValueError):
            continue
        labels[label_id] = str(value)
    return labels


def _target_class_ids(model: Any, raw_targets: list[str]) -> list[int]:
    labels = _id_to_label(model)
    label_to_id = {label.lower(): label_id for label_id, label in labels.items()}
    if raw_targets:
        targets: list[int] = []
        for raw in raw_targets:
            raw_text = str(raw).strip()
            if raw_text == "":
                continue
            if raw_text.isdigit():
                targets.append(int(raw_text))
            else:
                lowered = raw_text.lower()
                if lowered not in label_to_id:
                    raise ValueError(f"unknown --target-class {raw_text!r}; known labels are {labels}")
                targets.append(label_to_id[lowered])
        if targets:
            return sorted(set(targets))
    non_human = [
        label_id
        for label_id, label in labels.items()
        if label.strip().lower() not in {"human", "label_0"}
    ]
    if non_human:
        return sorted(non_human)
    return [1]


def _target_scores(logits: torch.Tensor, target_ids: list[int]) -> torch.Tensor:
    valid_ids = [target_id for target_id in target_ids if 0 <= target_id < logits.shape[-1]]
    if not valid_ids:
        raise ValueError(f"target class ids {target_ids} are out of range for logits with shape {tuple(logits.shape)}")
    return logits[:, valid_ids].float().mean(dim=-1)


@torch.inference_mode()
def score_latents(
    *,
    model: Any,
    tokenizer: Any,
    sae: BatchTopKSAE,
    docs: list[SAEDoc],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sae.eval()
    candidate_docs = [doc for doc in docs if doc.label == 1] or docs
    target_ids = _target_class_ids(model, args.target_class)
    id_to_label = _id_to_label(model)
    target_labels = [id_to_label.get(target_id, str(target_id)) for target_id in target_ids]
    latent_sum = torch.zeros(sae.latent_dim, dtype=torch.float64)
    latent_active_docs: Counter[int] = Counter()
    latent_max = torch.zeros(sae.latent_dim, dtype=torch.float32)
    example_heaps: dict[int, list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    serial = 0

    batches = _token_batches(candidate_docs, args.score_batch_size)
    for batch_index, batch_docs in enumerate(batches, start=1):
        encoded = _encode_docs(
            tokenizer,
            batch_docs,
            max_length=args.max_length,
            device=device,
            pad_to_max_length=args.pad_to_max_length,
        )
        _, sparse_codes = _logits_with_sae(
            model=model,
            encoded=encoded,
            sae=sae,
            record_codes=True,
        )
        if sparse_codes is None:
            raise RuntimeError("SAE scoring hook did not capture latent codes")
        token_mask = _activation_mask(encoded).reshape(-1)
        codes = sparse_codes[token_mask].detach().cpu()
        if codes.numel() == 0:
            continue
        latent_sum += codes.sum(dim=0, dtype=torch.float64)
        latent_max = torch.maximum(latent_max, codes.max(dim=0).values.float())
        doc_token_counts = encoded["input_ids"].shape[1]
        flat_doc_indexes = (
            torch.arange(len(batch_docs)).unsqueeze(1).expand(-1, doc_token_counts).reshape(-1)
        )
        flat_token_indexes = (
            torch.arange(doc_token_counts).unsqueeze(0).expand(len(batch_docs), -1).reshape(-1)
        )
        kept_doc_indexes = flat_doc_indexes[token_mask.cpu()]
        kept_token_indexes = flat_token_indexes[token_mask.cpu()]
        for doc_index in range(len(batch_docs)):
            doc_codes = codes[kept_doc_indexes == doc_index]
            if doc_codes.numel() == 0:
                continue
            for latent_id in torch.nonzero(doc_codes.sum(dim=0) > 0, as_tuple=False).flatten().tolist():
                latent_active_docs[int(latent_id)] += 1
        top_values, top_indexes = torch.topk(
            codes.flatten(),
            k=min(codes.numel(), max(1, args.score_top_latents * args.max_examples_per_latent)),
        )
        for value, flat_index in zip(top_values.tolist(), top_indexes.tolist()):
            if value <= 0:
                continue
            token_row = flat_index // sae.latent_dim
            latent_id = flat_index % sae.latent_dim
            doc_index = int(kept_doc_indexes[token_row])
            token_index = int(kept_token_indexes[token_row])
            doc = batch_docs[doc_index]
            serial += 1
            _push_example(
                example_heaps,
                latent_id=latent_id,
                activation=float(value),
                serial=serial,
                row={
                    "latent_id": latent_id,
                    "activation": float(value),
                    "source": doc.source,
                    "doc_id": doc.doc_id,
                    "label": doc.label,
                    "token_index": token_index,
                    "token": tokenizer.convert_tokens_to_string(
                        [tokenizer.convert_ids_to_tokens(int(encoded["input_ids"][doc_index, token_index]))]
                    ).strip(),
                    "context": _context_for_token(
                        tokenizer,
                        encoded,
                        batch_index=doc_index,
                        token_index=token_index,
                        context_tokens=args.example_context_tokens,
                    ),
                },
                max_examples=args.max_examples_per_latent,
            )
        if args.progress_every > 0 and (batch_index == 1 or batch_index % args.progress_every == 0):
            print(
                f"Scanned SAE activations for {min(batch_index * args.score_batch_size, len(candidate_docs)):,}/"
                f"{len(candidate_docs):,} scoring docs",
                flush=True,
            )

    candidate_latents = [
        int(index)
        for index in torch.topk(
            latent_sum.float(),
            k=min(args.score_top_latents, sae.latent_dim),
        ).indices.tolist()
        if latent_sum[index] > 0
    ]

    ablation_effects: dict[int, list[float]] = {latent_id: [] for latent_id in candidate_latents}
    baseline_logit_sums: dict[int, float] = defaultdict(float)
    for batch_index, batch_docs in enumerate(batches, start=1):
        encoded = _encode_docs(
            tokenizer,
            batch_docs,
            max_length=args.max_length,
            device=device,
            pad_to_max_length=args.pad_to_max_length,
        )
        baseline_logits, _ = _logits_with_sae(model=model, encoded=encoded, sae=sae)
        baseline_target_scores = _target_scores(baseline_logits, target_ids)
        for latent_id in candidate_latents:
            ablated_logits, _ = _logits_with_sae(
                model=model,
                encoded=encoded,
                sae=sae,
                ablate_latent=latent_id,
            )
            effects = (
                baseline_target_scores - _target_scores(ablated_logits, target_ids)
            ).detach().cpu().tolist()
            ablation_effects[latent_id].extend(float(value) for value in effects)
            baseline_logit_sums[latent_id] += float(baseline_target_scores.detach().sum().cpu())
        if args.progress_every > 0 and (batch_index == 1 or batch_index % args.progress_every == 0):
            print(
                f"Ablated {len(candidate_latents)} candidate latents on "
                f"{min(batch_index * args.score_batch_size, len(candidate_docs)):,}/{len(candidate_docs):,} docs",
                flush=True,
            )

    latent_rows = []
    for latent_id in candidate_latents:
        effects = ablation_effects[latent_id]
        mean_effect = sum(effects) / max(1, len(effects))
        positive_count = sum(1 for value in effects if value > 0)
        latent_rows.append(
            {
                "latent_id": latent_id,
                "mean_llm_logit_drop_when_ablated": mean_effect,
                "positive_effect_rate": positive_count / max(1, len(effects)),
                "scored_docs": len(effects),
                "activation_sum": float(latent_sum[latent_id]),
                "max_activation": float(latent_max[latent_id]),
                "active_doc_count": int(latent_active_docs[latent_id]),
                "target_class_ids": "|".join(str(target_id) for target_id in target_ids),
                "target_class_labels": "|".join(target_labels),
            }
        )
    latent_rows.sort(
        key=lambda row: (
            float(row["mean_llm_logit_drop_when_ablated"]),
            float(row["activation_sum"]),
        ),
        reverse=True,
    )

    example_rows: list[dict[str, Any]] = []
    for latent_id, heap in example_heaps.items():
        if latent_id not in candidate_latents:
            continue
        for rank, (_activation, _serial, row) in enumerate(
            sorted(heap, key=lambda item: item[0], reverse=True),
            start=1,
        ):
            example_rows.append({**row, "example_rank": rank})
    example_rows.sort(key=lambda row: (int(row["latent_id"]), int(row["example_rank"])))
    return latent_rows, example_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "latent_id",
        "mean_llm_logit_drop_when_ablated",
        "positive_effect_rate",
        "activation_sum",
        "max_activation",
        "active_doc_count",
        "source",
        "doc_id",
        "activation",
        "context",
    ]
    columns = [column for column in preferred if column in columns] + [
        column for column in columns if column not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_sae(path: Path, sae: BatchTopKSAE, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": sae.state_dict(),
            "config": {
            "input_dim": sae.input_dim,
            "latent_dim": sae.latent_dim,
            "k": sae.k,
            "init_mode": sae.init_mode,
        },
            "summary": summary,
        },
        path,
    )


def _load_sae(path: Path, *, device: torch.device) -> tuple[BatchTopKSAE, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"SAE checkpoint {path} must contain a dictionary payload")
    config = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError(f"SAE checkpoint {path} must contain config and state_dict entries")
    sae = BatchTopKSAE(
        input_dim=int(config["input_dim"]),
        latent_dim=int(config["latent_dim"]),
        k=int(config["k"]),
        init_mode=str(config.get("init_mode", "kaiming")),
    )
    sae.load_state_dict(state_dict)
    sae.to(device)
    sae.eval()
    summary = payload.get("summary")
    return sae, summary if isinstance(summary, dict) else {}


def _activation_cache_sidecar(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}.{suffix}")


def _save_activation_cache_parquet(
    path: Path,
    *,
    activation_cache: ActivationCache,
    metadata: dict[str, Any],
    compression_level: int = 3,
    row_group_size: int = 8192,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    activation_cache = _validate_activation_cache(activation_cache)
    activations = activation_cache.activations.detach().cpu().contiguous().to(dtype=torch.float16)
    row_count, hidden_dim = activations.shape
    doc_indices = activation_cache.doc_indices.detach().cpu().to(dtype=torch.int64).numpy()
    token_indices = activation_cache.token_indices.detach().cpu().to(dtype=torch.int64).numpy()
    activation_array = activations.numpy()
    cache_metadata = {
        **metadata,
        "cache_version": 3,
        "format": "parquet_fixed_size_list_halffloat",
        "compression": "zstd",
        "compression_level": int(compression_level),
        "activation_rows": int(row_count),
        "activation_dim": int(hidden_dim),
        "document_rows": int(len(activation_cache.document_rows)),
    }
    schema = pa.schema(
        [
            ("activation", pa.list_(pa.float16(), list_size=hidden_dim)),
            ("doc_index", pa.int64()),
            ("token_index", pa.int64()),
        ],
        metadata={b"slop_activation_cache": json.dumps(cache_metadata).encode("utf-8")},
    )
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(
            path,
            schema,
            compression="zstd",
            compression_level=max(1, min(22, int(compression_level))),
            use_dictionary=False,
        )
        for start in range(0, row_count, row_group_size):
            end = min(start + row_group_size, row_count)
            flat_values = pa.array(
                activation_array[start:end].reshape(-1),
                type=pa.float16(),
            )
            table = pa.Table.from_arrays(
                [
                    pa.FixedSizeListArray.from_arrays(flat_values, hidden_dim),
                    pa.array(doc_indices[start:end], type=pa.int64()),
                    pa.array(token_indices[start:end], type=pa.int64()),
                ],
                schema=schema,
            )
            writer.write_table(table, row_group_size=row_group_size)
    finally:
        if writer is not None:
            writer.close()

    documents_path = _activation_cache_sidecar(path, "documents.parquet")
    pd.DataFrame(activation_cache.document_rows).to_parquet(
        documents_path,
        compression="zstd",
        index=False,
    )
    _write_json(_activation_cache_sidecar(path, "metadata.json"), cache_metadata)


class ActivationCacheParquetStream:
    def __init__(
        self,
        path: Path,
        *,
        hidden_dim: int,
        metadata: dict[str, Any],
        compression_level: int = 3,
        row_group_size: int = 8192,
    ) -> None:
        self.path = path
        self.hidden_dim = int(hidden_dim)
        self.metadata = {
            **metadata,
            "cache_version": 3,
            "format": "parquet_fixed_size_list_halffloat",
            "compression": "zstd",
            "compression_level": int(compression_level),
            "activation_dim": int(hidden_dim),
        }
        self.row_group_size = int(row_group_size)
        self.row_count = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self.schema = pa.schema(
            [
                ("activation", pa.list_(pa.float16(), list_size=self.hidden_dim)),
                ("doc_index", pa.int64()),
                ("token_index", pa.int64()),
            ],
            metadata={b"slop_activation_cache": json.dumps(self.metadata).encode("utf-8")},
        )
        self.writer = pq.ParquetWriter(
            path,
            self.schema,
            compression="zstd",
            compression_level=max(1, min(22, int(compression_level))),
            use_dictionary=False,
        )

    def write(
        self,
        *,
        activations: torch.Tensor,
        doc_indices: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> None:
        activations = activations.detach().cpu().contiguous().to(dtype=torch.float16)
        if activations.ndim != 2 or int(activations.shape[1]) != self.hidden_dim:
            raise ValueError(
                f"activation chunk must have shape [N, {self.hidden_dim}], got {tuple(activations.shape)}"
            )
        if doc_indices.numel() != activations.shape[0] or token_indices.numel() != activations.shape[0]:
            raise ValueError("provenance chunk lengths must match activation rows")
        activation_array = activations.numpy()
        flat_values = pa.array(activation_array.reshape(-1), type=pa.float16())
        table = pa.Table.from_arrays(
            [
                pa.FixedSizeListArray.from_arrays(flat_values, self.hidden_dim),
                pa.array(doc_indices.detach().cpu().to(dtype=torch.int64).numpy(), type=pa.int64()),
                pa.array(token_indices.detach().cpu().to(dtype=torch.int64).numpy(), type=pa.int64()),
            ],
            schema=self.schema,
        )
        self.writer.write_table(table, row_group_size=self.row_group_size)
        self.row_count += int(activations.shape[0])

    def close(self, *, document_rows: list[dict[str, Any]]) -> None:
        self.writer.close()
        final_metadata = {
            **self.metadata,
            "activation_rows": int(self.row_count),
            "document_rows": int(len(document_rows)),
        }
        pd.DataFrame(document_rows).to_parquet(
            _activation_cache_sidecar(self.path, "documents.parquet"),
            compression="zstd",
            index=False,
        )
        _write_json(_activation_cache_sidecar(self.path, "metadata.json"), final_metadata)


def _load_activation_cache_parquet(path: Path) -> ActivationCache:
    parquet_file = pq.ParquetFile(path)
    if not parquet_file.schema_arrow.names or "activation" not in parquet_file.schema_arrow.names:
        raise ValueError(f"activation cache {path} is missing an activation column")
    activation_type = parquet_file.schema_arrow.field("activation").type
    if not isinstance(activation_type, pa.FixedSizeListType):
        raise ValueError(f"activation cache {path} must use a fixed-size-list activation column")
    hidden_dim = int(activation_type.list_size)
    row_count = int(parquet_file.metadata.num_rows)
    activations = torch.empty((row_count, hidden_dim), dtype=torch.float16)
    doc_indices = torch.empty(row_count, dtype=torch.long)
    token_indices = torch.empty(row_count, dtype=torch.long)
    offset = 0
    for batch in parquet_file.iter_batches(
        columns=["activation", "doc_index", "token_index"],
        batch_size=8192,
    ):
        activation_column = batch.column("activation")
        if not isinstance(activation_column, pa.FixedSizeListArray):
            activation_column = activation_column.combine_chunks()
        values = activation_column.values.to_numpy(zero_copy_only=False)
        next_offset = offset + len(activation_column)
        activations[offset:next_offset].copy_(
            torch.from_numpy(values.reshape(len(activation_column), hidden_dim).copy())
        )
        doc_indices[offset:next_offset].copy_(
            torch.from_numpy(batch.column("doc_index").to_numpy(zero_copy_only=False).copy()).long()
        )
        token_indices[offset:next_offset].copy_(
            torch.from_numpy(batch.column("token_index").to_numpy(zero_copy_only=False).copy()).long()
        )
        offset = next_offset
    documents_path = _activation_cache_sidecar(path, "documents.parquet")
    document_rows = (
        pd.read_parquet(documents_path).to_dict(orient="records")
        if documents_path.is_file()
        else []
    )
    if row_count == 0:
        raise ValueError(f"activation cache {path} does not contain activation rows")
    return _validate_activation_cache(
        ActivationCache(
            activations=activations,
            document_rows=[dict(row) for row in document_rows],
            doc_indices=doc_indices,
            token_indices=token_indices,
        ),
        path=path,
    )


def _load_activation_cache(path: Path) -> ActivationCache:
    if path.suffix == ".parquet":
        return _load_activation_cache_parquet(path)
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        cache = ActivationCache(
            activations=payload,
            document_rows=[],
            doc_indices=torch.empty(0, dtype=torch.long),
            token_indices=torch.empty(0, dtype=torch.long),
        )
    elif isinstance(payload, dict) and isinstance(payload.get("activations"), torch.Tensor):
        raw_documents = payload.get("activation_documents")
        document_rows = (
            [dict(row) for row in raw_documents if isinstance(row, dict)]
            if isinstance(raw_documents, list)
            else []
        )
        raw_doc_indices = payload.get("activation_doc_indices")
        raw_token_indices = payload.get("activation_token_indices")
        doc_indices = (
            raw_doc_indices
            if isinstance(raw_doc_indices, torch.Tensor)
            else torch.empty(0, dtype=torch.long)
        )
        token_indices = (
            raw_token_indices
            if isinstance(raw_token_indices, torch.Tensor)
            else torch.empty(0, dtype=torch.long)
        )
        cache = ActivationCache(
            activations=payload["activations"],
            document_rows=document_rows,
            doc_indices=doc_indices,
            token_indices=token_indices,
        )
    else:
        raise ValueError(f"activation cache {path} does not contain an activation tensor")
    return _validate_activation_cache(cache, path=path)


def _save_activation_cache(
    path: Path,
    *,
    activation_cache: ActivationCache,
    args: argparse.Namespace,
    docs: list[SAEDoc],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    activation_cache = _validate_activation_cache(activation_cache)
    torch.save(
        {
            "activations": activation_cache.activations.half(),
            "cache_version": 2,
            "source": "modernbert_penultimate_hidden_states",
            "checkpoint": str(args.checkpoint),
            "dataset_parquet": str(args.dataset_parquet) if args.dataset_parquet else "",
            "reference_jsonl": list(args.reference_jsonl),
            "generation_jsonl": list(args.generation_jsonl),
            "max_length": args.max_length,
            "pad_to_max_length": bool(args.pad_to_max_length),
            "max_activations": args.max_activations,
            "documents": len(docs),
            "activation_documents": activation_cache.document_rows,
            "activation_doc_indices": activation_cache.doc_indices,
            "activation_token_indices": activation_cache.token_indices,
        },
        path,
    )


def _wandb_config(
    args: argparse.Namespace,
    *,
    docs: list[SAEDoc],
    cache_path: Path,
    activation_vectors: int | None = None,
    input_dim: int | None = None,
) -> dict[str, Any]:
    source_counts = Counter(doc.source for doc in docs)
    label_counts = Counter(str(doc.label) for doc in docs)
    return {
        "checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "load_sae": str(args.load_sae) if args.load_sae else "",
        "dataset_parquet": str(args.dataset_parquet) if args.dataset_parquet else "",
        "reference_jsonl": list(args.reference_jsonl),
        "generation_jsonl": list(args.generation_jsonl),
        "feature_text_mode": args.feature_text_mode,
        "documents": len(docs),
        "source_counts": dict(sorted(source_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "max_docs_per_source": args.max_docs_per_source,
        "max_length": args.max_length,
        "collect_batch_size": args.collect_batch_size,
        "score_batch_size": args.score_batch_size,
        "pad_to_max_length": bool(args.pad_to_max_length),
        "max_activations": args.max_activations,
        "activation_cache": str(cache_path),
        "activation_cache_train_chunks": getattr(args, "activation_cache_train_chunks", 1),
        "activation_vectors": activation_vectors,
        "input_dim": input_dim,
        "latent_dim": args.latent_dim,
        "k": args.k,
        "init_mode": getattr(args, "init_mode", "kaiming"),
        "train_batch_size": args.train_batch_size,
        "epochs": args.epochs,
        "optimizer": getattr(args, "optimizer", "adamw"),
        "learning_rate": args.learning_rate,
        "muon_learning_rate": getattr(args, "muon_learning_rate", None),
        "adamw_learning_rate": getattr(args, "adamw_learning_rate", None),
        "muon_weight_decay": getattr(args, "muon_weight_decay", None),
        "muon_momentum": getattr(args, "muon_momentum", None),
        "muon_ns_steps": getattr(args, "muon_ns_steps", None),
        "weight_decay": args.weight_decay,
        "lr_schedule": args.lr_schedule,
        "warmup_ratio": args.warmup_ratio,
        "decay_ratio": args.decay_ratio,
        "min_lr_ratio": args.min_lr_ratio,
        "auxk_coefficient": getattr(args, "auxk_coefficient", 0.0),
        "auxk": getattr(args, "auxk", 0),
        "auxk_dead_steps": getattr(args, "auxk_dead_steps", 0),
        "eval_fraction": args.eval_fraction,
        "compile_sae": bool(args.compile_sae),
        "compile_detector": bool(args.compile_detector),
        "compile_mode": args.compile_mode,
        "num_workers": args.num_workers,
        "skip_latent_scoring": bool(args.skip_latent_scoring),
        "target_class": list(args.target_class),
        "score_top_latents": args.score_top_latents,
        "max_examples_per_latent": args.max_examples_per_latent,
        "seed": args.seed,
    }


def _init_sae_wandb(
    args: argparse.Namespace,
    *,
    docs: list[SAEDoc],
    cache_path: Path,
    activation_vectors: int | None = None,
    input_dim: int | None = None,
):
    return init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage4", "phase4", "sae", "batchtopk", *args.wandb_tag],
        config=_wandb_config(
            args,
            docs=docs,
            cache_path=cache_path,
            activation_vectors=activation_vectors,
            input_dim=input_dim,
        ),
    )


def _log_sae_wandb_outputs(
    wandb_run: Any,
    *,
    summary: dict[str, Any],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    latent_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
) -> None:
    wandb_run.log(
        {
            "summary/best_eval_mse": summary.get("best_eval_mse"),
            "summary/final_train_mse": summary.get("final_train_mse"),
            "summary/activation_vectors": summary.get("activation_vectors"),
            "summary/ranked_latents": summary.get("ranked_latents"),
            "summary/example_rows": summary.get("example_rows"),
        }
    )
    log_summary_table(wandb_run, "phase4_batchtopk_sae_train_log", train_rows)
    log_summary_table(wandb_run, "phase4_batchtopk_sae_eval_log", eval_rows)
    log_summary_table(wandb_run, "phase4_batchtopk_sae_latents", latent_rows)
    log_summary_table(wandb_run, "phase4_batchtopk_sae_examples", example_rows)
    log_summary_table(wandb_run, "phase4_batchtopk_sae_summary", [summary])


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    docs = _load_docs(args)
    cache_path = args.activation_cache or output_dir / "phase4_batchtopk_sae_activation_cache.pt"
    model: Any | None = None
    tokenizer: Any | None = None
    chunk_source: ParquetActivationChunkSource | None = None

    if args.load_sae is not None:
        sae, loaded_summary = _load_sae(args.load_sae, device=device)
        wandb_run = _init_sae_wandb(
            args,
            docs=docs,
            cache_path=Path(str(loaded_summary.get("activation_cache", cache_path))),
            activation_vectors=loaded_summary.get("activation_vectors"),
            input_dim=sae.input_dim,
        )
        try:
            if args.skip_latent_scoring:
                latent_rows: list[dict[str, Any]] = []
                example_rows: list[dict[str, Any]] = []
            else:
                model, tokenizer = _load_detector(args.checkpoint, device=device)
                latent_rows, example_rows = score_latents(
                    model=model,
                    tokenizer=tokenizer,
                    sae=sae,
                    docs=docs,
                    args=args,
                    device=device,
                )
            source_counts = Counter(doc.source for doc in docs)
            label_counts = Counter(str(doc.label) for doc in docs)
            summary = {
                "checkpoint": str(args.checkpoint),
                "device": str(device),
                "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                "documents": len(docs),
                "source_counts": dict(sorted(source_counts.items())),
                "label_counts": dict(sorted(label_counts.items())),
                "activation_cache": str(loaded_summary.get("activation_cache", cache_path)),
                "activation_cache_reused": True,
                "activation_vectors": loaded_summary.get("activation_vectors"),
                "input_dim": sae.input_dim,
                "pad_to_max_length": loaded_summary.get(
                    "pad_to_max_length", bool(args.pad_to_max_length)
                ),
                "latent_dim": sae.latent_dim,
                "k": sae.k,
                "init_mode": loaded_summary.get("init_mode", sae.init_mode),
                "epochs": loaded_summary.get("epochs"),
                "optimizer": loaded_summary.get("optimizer"),
                "learning_rate": loaded_summary.get("learning_rate"),
                "muon_learning_rate": loaded_summary.get("muon_learning_rate"),
                "adamw_learning_rate": loaded_summary.get("adamw_learning_rate"),
                "muon_weight_decay": loaded_summary.get("muon_weight_decay"),
                "muon_momentum": loaded_summary.get("muon_momentum"),
                "muon_ns_steps": loaded_summary.get("muon_ns_steps"),
                "weight_decay": loaded_summary.get("weight_decay"),
                "lr_schedule": loaded_summary.get("lr_schedule"),
                "warmup_ratio": loaded_summary.get("warmup_ratio"),
                "decay_ratio": loaded_summary.get("decay_ratio"),
                "min_lr_ratio": loaded_summary.get("min_lr_ratio"),
                "warmup_steps": loaded_summary.get("warmup_steps", 0),
                "stable_steps": loaded_summary.get("stable_steps", 0),
                "decay_steps": loaded_summary.get("decay_steps", 0),
                "auxk_coefficient": loaded_summary.get("auxk_coefficient"),
                "auxk": loaded_summary.get("auxk"),
                "auxk_dead_steps": loaded_summary.get("auxk_dead_steps"),
                "eval_fraction": loaded_summary.get("eval_fraction"),
                "final_train_mse": loaded_summary.get("final_train_mse"),
                "best_eval_mse": loaded_summary.get("best_eval_mse"),
                "compile_sae": loaded_summary.get("compile_sae"),
                "compile_detector": bool(args.compile_detector),
                "compile_sae_fallback": loaded_summary.get("compile_sae_fallback"),
                "compile_detector_fallback": False,
                "compile_mode": loaded_summary.get("compile_mode", args.compile_mode),
                "skip_latent_scoring": bool(args.skip_latent_scoring),
                "score_top_latents": args.score_top_latents,
                "ranked_latents": len(latent_rows),
                "example_rows": len(example_rows),
                "loaded_sae": str(args.load_sae),
                "loaded_sae_summary": loaded_summary,
            }
            _save_sae(output_dir / "phase4_batchtopk_sae.pt", sae, summary)
            _write_csv(output_dir / "phase4_batchtopk_sae_train_log.csv", [])
            _write_csv(output_dir / "phase4_batchtopk_sae_eval_log.csv", [])
            _write_csv(output_dir / "phase4_batchtopk_sae_latents.csv", latent_rows)
            _write_csv(output_dir / "phase4_batchtopk_sae_examples.csv", example_rows)
            _write_json(output_dir / "phase4_batchtopk_sae_summary.json", summary)
            _log_sae_wandb_outputs(
                wandb_run,
                summary=summary,
                train_rows=[],
                eval_rows=[],
                latent_rows=latent_rows,
                example_rows=example_rows,
            )
            return summary
        finally:
            wandb_run.finish()

    cache_reused = cache_path.exists() and not args.refresh_activation_cache
    if cache_reused:
        activation_cache_has_provenance = _activation_cache_sidecar(cache_path, "documents.parquet").is_file()
        if cache_path.suffix == ".parquet" and int(args.activation_cache_train_chunks) > 1:
            chunk_source = ParquetActivationChunkSource(
                cache_path,
                chunks=int(args.activation_cache_train_chunks),
            )
            print(
                f"Training from Parquet activation cache {cache_path} in "
                f"{len(chunk_source.chunks)} chunks "
                f"({chunk_source.row_count} activations, dim={chunk_source.hidden_dim})",
                flush=True,
            )
            activation_cache = None
            activations = None
            activation_vectors = chunk_source.row_count
            input_dim = chunk_source.hidden_dim
        else:
            print(f"Loading activation cache from {cache_path}", flush=True)
            activation_cache = _load_activation_cache(cache_path)
            activations = activation_cache.activations
            activation_vectors = int(activations.shape[0])
            input_dim = int(activations.shape[1])
            activation_cache_has_provenance = activation_cache.has_provenance
        detector_compile_fallback = False
    else:
        model, tokenizer = _load_detector(args.checkpoint, device=device)
        collection_model = _compile_module(
            model,
            enabled=bool(args.compile_detector),
            mode=args.compile_mode,
            device=device,
        )
        detector_compile_fallback = False
        try:
            activation_cache = collect_penultimate_activation_cache(
                model=collection_model,
                tokenizer=tokenizer,
                docs=docs,
                max_length=args.max_length,
                batch_size=args.collect_batch_size,
                max_activations=args.max_activations,
                device=device,
                progress_every=args.progress_every,
                pad_to_max_length=args.pad_to_max_length,
            )
        except Exception as exc:
            if collection_model is model:
                raise
            print(
                f"torch.compile detector collection failed ({type(exc).__name__}: {exc}); "
                "falling back to eager detector collection",
                flush=True,
            )
            _reset_torch_compile()
            detector_compile_fallback = True
            activation_cache = collect_penultimate_activation_cache(
                model=model,
                tokenizer=tokenizer,
                docs=docs,
                max_length=args.max_length,
                batch_size=args.collect_batch_size,
                max_activations=args.max_activations,
                device=device,
                progress_every=args.progress_every,
                pad_to_max_length=args.pad_to_max_length,
            )
        activations = activation_cache.activations
        activation_vectors = int(activations.shape[0])
        input_dim = int(activations.shape[1])
        activation_cache_has_provenance = activation_cache.has_provenance
        _save_activation_cache(cache_path, activation_cache=activation_cache, args=args, docs=docs)
    wandb_run = _init_sae_wandb(
        args,
        docs=docs,
        cache_path=cache_path,
        activation_vectors=activation_vectors,
        input_dim=input_dim,
    )
    try:
        if chunk_source is not None:
            sae, train_rows, eval_rows, train_summary = train_sae_from_parquet_chunks(
                source=chunk_source,
                args=args,
                device=device,
                wandb_run=wandb_run,
            )
        else:
            if activations is None:
                raise ValueError("activation tensor was not loaded")
            sae, train_rows, eval_rows, train_summary = train_sae(
                activations=activations,
                args=args,
                device=device,
                wandb_run=wandb_run,
            )
        if args.skip_latent_scoring:
            latent_rows: list[dict[str, Any]] = []
            example_rows: list[dict[str, Any]] = []
        else:
            if model is None or tokenizer is None:
                model, tokenizer = _load_detector(args.checkpoint, device=device)
            latent_rows, example_rows = score_latents(
                model=model,
                tokenizer=tokenizer,
                sae=sae,
                docs=docs,
                args=args,
                device=device,
            )
        source_counts = Counter(doc.source for doc in docs)
        label_counts = Counter(str(doc.label) for doc in docs)
        best_eval_mse = min((float(row["eval_mse"]) for row in eval_rows), default=None)
        final_train_mse = float(train_rows[-1]["mse"]) if train_rows else None
        summary = {
            "checkpoint": str(args.checkpoint),
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "documents": len(docs),
            "source_counts": dict(sorted(source_counts.items())),
            "label_counts": dict(sorted(label_counts.items())),
                "activation_cache": str(cache_path),
                "activation_cache_reused": cache_reused,
                "activation_cache_has_provenance": activation_cache_has_provenance,
                "activation_cache_train_chunks": train_summary.get(
                    "activation_cache_train_chunks",
                    1,
                ),
                "activation_cache_chunk_rows": train_summary.get("activation_cache_chunk_rows", []),
                "activation_vectors": activation_vectors,
                "input_dim": input_dim,
            "pad_to_max_length": bool(args.pad_to_max_length),
            "latent_dim": args.latent_dim,
            "k": args.k,
            "init_mode": getattr(args, "init_mode", "kaiming"),
            "epochs": args.epochs,
            "optimizer": train_summary.get("optimizer", getattr(args, "optimizer", "adamw")),
            "learning_rate": args.learning_rate,
            "muon_learning_rate": train_summary.get("muon_learning_rate"),
            "adamw_learning_rate": train_summary.get("adamw_learning_rate"),
            "muon_weight_decay": train_summary.get("muon_weight_decay"),
            "muon_momentum": train_summary.get("muon_momentum"),
            "muon_ns_steps": train_summary.get("muon_ns_steps"),
            "weight_decay": args.weight_decay,
            "lr_schedule": args.lr_schedule,
            "warmup_ratio": args.warmup_ratio,
            "decay_ratio": args.decay_ratio,
            "min_lr_ratio": args.min_lr_ratio,
            "warmup_steps": train_summary.get("warmup_steps", 0),
            "stable_steps": train_summary.get("stable_steps", 0),
            "decay_steps": train_summary.get("decay_steps", 0),
            "auxk_coefficient": train_summary.get("auxk_coefficient"),
            "auxk": train_summary.get("auxk"),
            "auxk_dead_steps": train_summary.get("auxk_dead_steps"),
            "eval_fraction": args.eval_fraction,
            "final_train_mse": final_train_mse,
            "best_eval_mse": best_eval_mse,
            "compile_sae": bool(args.compile_sae),
            "compile_detector": bool(args.compile_detector),
            "compile_sae_fallback": bool(train_summary["compile_sae_fallback"]),
            "compile_detector_fallback": detector_compile_fallback,
            "compile_mode": args.compile_mode,
            "skip_latent_scoring": bool(args.skip_latent_scoring),
            "score_top_latents": args.score_top_latents,
            "ranked_latents": len(latent_rows),
            "example_rows": len(example_rows),
        }
        _save_sae(output_dir / "phase4_batchtopk_sae.pt", sae, summary)
        _write_csv(output_dir / "phase4_batchtopk_sae_train_log.csv", train_rows)
        _write_csv(output_dir / "phase4_batchtopk_sae_eval_log.csv", eval_rows)
        _write_csv(output_dir / "phase4_batchtopk_sae_latents.csv", latent_rows)
        _write_csv(output_dir / "phase4_batchtopk_sae_examples.csv", example_rows)
        _write_json(output_dir / "phase4_batchtopk_sae_summary.json", summary)
        _log_sae_wandb_outputs(
            wandb_run,
            summary=summary,
            train_rows=train_rows,
            eval_rows=eval_rows,
            latent_rows=latent_rows,
            example_rows=example_rows,
        )
        return summary
    finally:
        wandb_run.finish()


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(
        "Wrote Phase 4 BatchTopK SAE outputs to "
        f"{args.output_dir} ({summary['activation_vectors']} activations, "
        f"{summary['ranked_latents']} ranked latents)"
    )


if __name__ == "__main__":
    main()
