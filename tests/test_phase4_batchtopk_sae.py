from __future__ import annotations

import json
from argparse import Namespace

import torch
from torch import nn

from slop_sftdiv.cli.phase4_batchtopk_sae import (
    BatchTopKSAE,
    _load_jsonl_docs,
    _logits_with_sae,
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
        train_batch_size=8,
        epochs=1,
        progress_every=1,
    )

    sae, rows = train_sae(activations=activations, args=args, device=torch.device("cpu"))

    assert sae.input_dim == 4
    assert sae.latent_dim == 8
    assert rows
    assert rows[-1]["loss"] >= 0


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
