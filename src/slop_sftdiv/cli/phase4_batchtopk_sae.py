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

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from slop_sftdiv.generation_text import (
    FeatureTextMode,
    feature_text_for_mode,
    strip_chat_role_markers,
)


@dataclass(frozen=True)
class SAEDoc:
    source: str
    doc_id: str
    text: str
    label: int


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
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=decoder_bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight, a=5**0.5)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, a=5**0.5)
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
        threshold = torch.topk(dense_codes.flatten(), keep).values[-1]
        return dense_codes * (dense_codes >= threshold)

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        return self.batch_topk(self.encode_dense(activations))

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
    parser.add_argument("--feature-text-mode", choices=["raw", "final_answer"], default="final_answer")
    parser.add_argument("--max-docs-per-source", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--collect-batch-size", type=int, default=8)
    parser.add_argument("--score-batch-size", type=int, default=4)
    parser.add_argument("--max-activations", type=int, default=200_000)
    parser.add_argument("--latent-dim", type=int, default=4096)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--score-top-latents", type=int, default=128)
    parser.add_argument("--max-examples-per-latent", type=int, default=8)
    parser.add_argument("--example-context-tokens", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--progress-every", type=int, default=25)
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
    if not docs:
        raise ValueError("provide at least one --reference-jsonl or --generation-jsonl input")
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
) -> dict[str, Any]:
    encoded = tokenizer(
        [doc.text for doc in docs],
        padding=True,
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
) -> torch.Tensor:
    if max_activations <= 0:
        raise ValueError("--max-activations must be positive")
    chunks: list[torch.Tensor] = []
    collected = 0
    batches = _token_batches(docs, batch_size)
    for batch_index, batch_docs in enumerate(batches, start=1):
        encoded = _encode_docs(tokenizer, batch_docs, max_length=max_length, device=device)
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[-2]
        vectors = hidden[_activation_mask(encoded)].detach().float().cpu()
        remaining = max_activations - collected
        if remaining <= 0:
            break
        if vectors.shape[0] > remaining:
            vectors = vectors[:remaining]
        chunks.append(vectors)
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
    return torch.cat(chunks, dim=0)


def train_sae(
    *,
    activations: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[BatchTopKSAE, list[dict[str, Any]]]:
    sae = BatchTopKSAE(
        input_dim=int(activations.shape[1]),
        latent_dim=args.latent_dim,
        k=args.k,
    ).to(device)
    optimizer = torch.optim.AdamW(
        sae.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    train_rows: list[dict[str, Any]] = []
    start = time.time()
    step = 0
    sae.train()
    for epoch in range(1, args.epochs + 1):
        for (batch_cpu,) in loader:
            step += 1
            batch = batch_cpu.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, sparse_codes = sae(batch)
            residual = reconstruction - batch
            mse = residual.pow(2).mean()
            loss = mse
            loss.backward()
            optimizer.step()
            sae.normalize_decoder_weights()
            if step == 1 or step % args.progress_every == 0:
                active_per_vector = (sparse_codes > 0).float().sum(dim=1).mean()
                row = {
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "mse": float(mse.detach().cpu()),
                    "active_latents_per_vector": float(active_per_vector.detach().cpu()),
                    "elapsed_seconds": time.time() - start,
                }
                train_rows.append(row)
                print(
                    f"SAE step {step}: loss={row['loss']:.6f} "
                    f"active/vector={row['active_latents_per_vector']:.2f}",
                    flush=True,
                )
    return sae, train_rows


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
        hidden = kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0]
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
    latent_sum = torch.zeros(sae.latent_dim, dtype=torch.float64)
    latent_active_docs: Counter[int] = Counter()
    latent_max = torch.zeros(sae.latent_dim, dtype=torch.float32)
    example_heaps: dict[int, list[tuple[float, int, dict[str, Any]]]] = defaultdict(list)
    serial = 0

    batches = _token_batches(candidate_docs, args.score_batch_size)
    for batch_index, batch_docs in enumerate(batches, start=1):
        encoded = _encode_docs(tokenizer, batch_docs, max_length=args.max_length, device=device)
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
        encoded = _encode_docs(tokenizer, batch_docs, max_length=args.max_length, device=device)
        baseline_logits, _ = _logits_with_sae(model=model, encoded=encoded, sae=sae)
        baseline_llm_logits = baseline_logits[:, 1]
        for latent_id in candidate_latents:
            ablated_logits, _ = _logits_with_sae(
                model=model,
                encoded=encoded,
                sae=sae,
                ablate_latent=latent_id,
            )
            effects = (baseline_llm_logits - ablated_logits[:, 1]).detach().cpu().tolist()
            ablation_effects[latent_id].extend(float(value) for value in effects)
            baseline_logit_sums[latent_id] += float(baseline_llm_logits.detach().sum().cpu())
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
            },
            "summary": summary,
        },
        path,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_detector(args.checkpoint, device=device)
    docs = _load_docs(args)
    activations = collect_penultimate_activations(
        model=model,
        tokenizer=tokenizer,
        docs=docs,
        max_length=args.max_length,
        batch_size=args.collect_batch_size,
        max_activations=args.max_activations,
        device=device,
        progress_every=args.progress_every,
    )
    torch.save(
        {
            "activations": activations.half(),
            "source": "modernbert_penultimate_hidden_states",
            "checkpoint": str(args.checkpoint),
        },
        output_dir / "phase4_batchtopk_sae_activation_cache.pt",
    )
    sae, train_rows = train_sae(activations=activations, args=args, device=device)
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
        "activation_vectors": int(activations.shape[0]),
        "input_dim": int(activations.shape[1]),
        "latent_dim": args.latent_dim,
        "k": args.k,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "score_top_latents": args.score_top_latents,
        "ranked_latents": len(latent_rows),
        "example_rows": len(example_rows),
    }
    _save_sae(output_dir / "phase4_batchtopk_sae.pt", sae, summary)
    _write_csv(output_dir / "phase4_batchtopk_sae_train_log.csv", train_rows)
    _write_csv(output_dir / "phase4_batchtopk_sae_latents.csv", latent_rows)
    _write_csv(output_dir / "phase4_batchtopk_sae_examples.csv", example_rows)
    _write_json(output_dir / "phase4_batchtopk_sae_summary.json", summary)
    return summary


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
