from __future__ import annotations

import json
from argparse import Namespace

import pandas as pd
import torch
from torch import nn

from slop_sftdiv.cli.phase4_batchtopk_sae import (
    BatchTopKSAE,
    _load_jsonl_docs,
    _load_parquet_docs,
    _load_sae,
    _logits_with_sae,
    _save_sae,
    _target_class_ids,
    train_sae,
)


def test_batchtopk_sae_keeps_batch_level_average_k() -> None:
    sae = BatchTopKSAE(input_dim=3, latent_dim=6, k=2)
    dense_codes = torch.arange(24, dtype=torch.float32).reshape(4, 6)

    sparse_codes = sae.batch_topk(dense_codes)

    assert int((sparse_codes > 0).sum()) == 8
    assert sparse_codes[0, 0] == 0
    assert sparse_codes[-1, -1] == dense_codes[-1, -1]


def test_load_jsonl_docs_strips_generation_chat_markers(tmp_path) -> None:
    path = tmp_path / "generations.jsonl"
    path.write_text(
        json.dumps(
            {
                "record_id": "doc-1",
                "generation": "assistant\n\nHere is the final text.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    docs = _load_jsonl_docs(
        [f"gen={path}"],
        label=1,
        default_field="generation",
        max_docs_per_source=10,
        feature_text_mode="final_answer",
    )

    assert len(docs) == 1
    assert docs[0].source == "gen"
    assert docs[0].doc_id == "doc-1"
    assert docs[0].label == 1
    assert docs[0].text == "Here is the final text."


def test_train_sae_returns_loss_rows() -> None:
    activations = torch.randn(16, 4)
    args = Namespace(
        latent_dim=8,
        k=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        lr_schedule="wsd",
        warmup_ratio=0.5,
        decay_ratio=0.25,
        min_lr_ratio=0.0,
        train_batch_size=8,
        epochs=1,
        eval_fraction=0.25,
        seed=1729,
        compile_sae=False,
        compile_mode="default",
        num_workers=0,
        progress_every=1,
    )

    sae, rows, eval_rows, train_summary = train_sae(
        activations=activations,
        args=args,
        device=torch.device("cpu"),
    )

    assert sae.input_dim == 4
    assert sae.latent_dim == 8
    assert rows
    assert rows[-1]["loss"] >= 0
    assert eval_rows
    assert eval_rows[-1]["eval_mse"] >= 0
    assert train_summary["compile_sae_fallback"] is False
    assert train_summary["lr_schedule"] == "wsd"
    assert train_summary["warmup_steps"] == 1
    assert "learning_rate" in rows[-1]


def test_save_and_load_sae_round_trip(tmp_path) -> None:
    path = tmp_path / "phase4_batchtopk_sae.pt"
    sae = BatchTopKSAE(input_dim=4, latent_dim=8, k=2)
    summary = {"best_eval_mse": 1.25}

    _save_sae(path, sae, summary)
    loaded, loaded_summary = _load_sae(path, device=torch.device("cpu"))

    assert loaded.input_dim == sae.input_dim
    assert loaded.latent_dim == sae.latent_dim
    assert loaded.k == sae.k
    assert loaded_summary == summary
    for key, value in sae.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


def test_load_parquet_docs_labels_human_and_llm_rows(tmp_path) -> None:
    path = tmp_path / "continuations.parquet"
    pd.DataFrame(
        [
            {
                "record_id": "human-1",
                "role": "human_continuation",
                "model": "human",
                "text": "Human continuation.",
            },
            {
                "record_id": "ai-1",
                "role": "llm_continuation",
                "model": "gpt-5.5",
                "text": "Generated continuation.",
            },
        ]
    ).to_parquet(path, index=False)

    docs = _load_parquet_docs(path, max_docs_per_source=10, seed=1729)

    labels = {doc.doc_id: doc.label for doc in docs}
    assert labels == {"human-1": 0, "ai-1": 1}


class _DummyLayer(nn.Module):
    def forward(self, hidden_states, attention_mask=None):  # noqa: ANN001
        return hidden_states


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_DummyLayer()])
        self.embedding = nn.Embedding(8, 4)
        self.classifier = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.classifier.weight.zero_()
            self.classifier.weight[1, 0] = 1.0

    def forward(self, input_ids, attention_mask=None):  # noqa: ANN001
        hidden = self.embedding(input_ids)
        hidden = self.model.layers[-1](hidden_states=hidden, attention_mask=attention_mask)
        pooled = hidden[:, 0, :]
        return Namespace(logits=self.classifier(pooled))


class _DummyConfigModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = Namespace(id2label={0: "gemini-3.5-flash", 1: "gpt-5.5", 2: "human"})


def test_target_class_ids_default_to_non_human_labels() -> None:
    assert _target_class_ids(_DummyConfigModel(), []) == [0, 1]


def test_logits_with_sae_hook_can_ablate_latent() -> None:
    model = _DummyModel()
    sae = BatchTopKSAE(input_dim=4, latent_dim=2, k=1)
    with torch.no_grad():
        sae.encoder.weight.zero_()
        sae.encoder.bias[:] = torch.tensor([1.0, 0.0])
        sae.decoder.weight.zero_()
        sae.decoder.weight[0, 0] = 2.0
        sae.decoder.weight[0, 1] = 0.0
        sae.decoder.bias.zero_()
    encoded = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }

    logits, codes = _logits_with_sae(model=model, encoded=encoded, sae=sae, record_codes=True)
    ablated, _ = _logits_with_sae(
        model=model,
        encoded=encoded,
        sae=sae,
        ablate_latent=0,
    )

    assert codes is not None
    assert logits[0, 1] > ablated[0, 1]
