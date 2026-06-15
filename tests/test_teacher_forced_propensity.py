import csv
import json

import pytest
import torch

from slop_sftdiv.cli.teacher_forced_propensity import build_parser, run_teacher_forced_propensity
from slop_sftdiv.cli.teacher_forced_propensity import (
    FeatureAccumulator,
    _initiator_token_sequences,
    _sequence_prob_mass,
    _sequence_prob_mass_batch,
    _sequence_prob_mass_batch_cached,
    _sequence_prob_mass_batch_cached_multi,
    _summary_rows,
)


class FakeTokenizer:
    bos_token_id = 0
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [max(1, ord(char) % 97) for char in text[:3]]


def test_teacher_forced_propensity_writes_outputs_and_logs_summary(tmp_path, monkeypatch):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "opportunities.csv"
    summary_path = tmp_path / "summary.csv"
    input_path.write_text(
        json.dumps(
            {
                "id": "a",
                "text": "Great question. It is not noise, it is signal. Delve deeper.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    logged_payloads = []
    logged_tables = {}
    init_kwargs = {}
    load_kwargs = {}

    class FakeRun:
        def log(self, payload):
            logged_payloads.append(payload)

        def finish(self):
            pass

    def fake_init(**kwargs):
        init_kwargs.update(kwargs)
        return FakeRun()

    monkeypatch.setattr(
        "slop_sftdiv.cli.teacher_forced_propensity._load_model",
        lambda **kwargs: load_kwargs.update(kwargs) or (FakeTokenizer(), object(), "cpu"),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.teacher_forced_propensity._prefix_input_ids",
        lambda *_args, **_kwargs: {"input_ids": object(), "attention_mask": object(), "prefix_tokens": 3},
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.teacher_forced_propensity._sequence_prob_mass",
        lambda *_args, **_kwargs: 0.25,
    )
    monkeypatch.setattr("slop_sftdiv.cli.teacher_forced_propensity.init_wandb", fake_init)
    monkeypatch.setattr(
        "slop_sftdiv.cli.teacher_forced_propensity.log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )

    args = build_parser().parse_args(
        [
            "--model",
            "fake-model",
            "--model-revision",
            "it-SFT",
            "--input",
            str(input_path),
            "--sample-size",
            "1",
            "--feature",
            "contrastive_negation",
            "--feature",
            "stock_openers",
            "--max-opportunities",
            "8",
            "--opportunity-batch-size",
            "1",
            "--no-sequence-cache",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    summary = run_teacher_forced_propensity(args)

    assert output_path.exists()
    assert summary_path.exists()
    with output_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with summary_path.open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert rows
    assert {row["feature"] for row in rows} == {"contrastive_negation", "stock_openers"}
    assert all(int(row["initiator_sequences"]) > 0 for row in rows)
    assert "amplification_factor_ci_low" in summary_rows[0]
    assert "amplification_factor_ci_high" in summary_rows[0]
    assert "text" not in rows[0]
    assert "Great question" not in json.dumps(logged_tables)
    assert summary == logged_tables["propensity_summary"]
    assert logged_payloads[-1]["propensity/opportunities"] == len(rows)
    assert load_kwargs["model_revision"] == "it-SFT"
    assert init_kwargs["tags"][:4] == ["stage2", "phase2", "teacher-forced", "smoke"]
    assert init_kwargs["config"]["model_revision"] == "it-SFT"
    assert init_kwargs["config"]["bootstrap_samples"] == 1000


def test_teacher_forced_propensity_writes_reference_subset_summary(tmp_path, monkeypatch):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "opportunities.csv"
    summary_path = tmp_path / "summary.csv"
    subset_summary_path = tmp_path / "subset_summary.csv"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "human",
                        "text": "Great question. It is not noise, it is signal.",
                        "provenance": "Human",
                    }
                ),
                json.dumps(
                    {
                        "id": "synthetic",
                        "text": "Great question. It is not noise, it is signal.",
                        "provenance": "Synthetic",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    logged_tables = {}

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    monkeypatch.setattr(
        "slop_sftdiv.cli.teacher_forced_propensity._load_model",
        lambda **_kwargs: (FakeTokenizer(), object(), "cpu"),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.teacher_forced_propensity._prefix_input_ids",
        lambda *_args, **_kwargs: {"input_ids": object(), "attention_mask": object(), "prefix_tokens": 3},
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.teacher_forced_propensity._sequence_prob_mass",
        lambda *_args, **_kwargs: 0.5,
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.teacher_forced_propensity.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.teacher_forced_propensity.log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )

    args = build_parser().parse_args(
        [
            "--model",
            "fake-model",
            "--input",
            str(input_path),
            "--sample-size",
            "2",
            "--feature",
            "stock_openers",
            "--max-opportunities",
            "4",
            "--opportunity-batch-size",
            "1",
            "--no-sequence-cache",
            "--reference-subset",
            "human:provenance=Human",
            "--reference-subset",
            "nonhuman:provenance!=Human",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--reference-subset-summary-output",
            str(subset_summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    run_teacher_forced_propensity(args)

    with subset_summary_path.open(encoding="utf-8", newline="") as handle:
        subset_rows = list(csv.DictReader(handle))
    by_subset = {row["reference_subset"]: row for row in subset_rows}

    assert set(by_subset) == {"human", "nonhuman"}
    assert by_subset["human"]["reference_subset_filter"] == "provenance=Human"
    assert by_subset["nonhuman"]["reference_subset_filter"] == "provenance!=Human"
    assert by_subset["human"]["opportunities"] == "1"
    assert by_subset["nonhuman"]["opportunities"] == "1"
    assert by_subset["human"]["reference_initiations"] == "1"
    assert "propensity_reference_subset_summary" in logged_tables
    assert "Great question" not in json.dumps(logged_tables)


def test_summary_rows_can_report_neutral_normalized_af():
    accumulators = {
        "slop_lexicon": FeatureAccumulator(
            opportunities=10,
            reference_initiations=1,
            prob_mass_sum=0.2,
        ),
        "neutral_common_controls": FeatureAccumulator(
            opportunities=10,
            reference_initiations=1,
            prob_mass_sum=0.1,
        ),
    }

    rows = _summary_rows(
        accumulators,
        normalization_feature="neutral_common_controls",
    )
    by_feature = {row["feature"]: row for row in rows}

    assert by_feature["neutral_common_controls"]["normalized_amplification_factor"] == 1.0
    assert by_feature["slop_lexicon"]["normalized_amplification_factor"] == 2.0
    assert by_feature["slop_lexicon"]["normalization_feature"] == "neutral_common_controls"


def test_initiator_token_sequences_include_sentence_case_variants():
    sequences = _initiator_token_sequences(FakeTokenizer(), "neutral_for_example")

    assert tuple(FakeTokenizer().encode("for example", add_special_tokens=False)) in sequences
    assert tuple(FakeTokenizer().encode("For example", add_special_tokens=False)) in sequences


def test_initiator_token_sequences_support_rule_of_three_extension():
    sequences = _initiator_token_sequences(FakeTokenizer(), "rule_of_three_approx")

    assert tuple(FakeTokenizer().encode(", and", add_special_tokens=False)) in sequences
    assert tuple(FakeTokenizer().encode(", or", add_special_tokens=False)) in sequences
    assert tuple(FakeTokenizer().encode("and", add_special_tokens=False)) in sequences
    assert tuple(FakeTokenizer().encode("or", add_special_tokens=False)) in sequences


def test_sequence_prob_mass_batches_by_depth():
    class UniformModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, *, input_ids, attention_mask):
            del attention_mask
            self.calls += 1
            return type("Output", (), {"logits": torch.zeros((*input_ids.shape, 5))})()

    model = UniformModel()
    mass = _sequence_prob_mass(
        model,
        {
            "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        },
        ((1,), (2, 3)),
    )

    assert mass == pytest.approx(0.24)
    assert model.calls == 2


def test_sequence_prob_mass_batches_shared_continuation_prefixes():
    class UniformModel:
        def __init__(self):
            self.calls = 0
            self.batch_sizes = []

        def __call__(self, *, input_ids, attention_mask):
            del attention_mask
            self.calls += 1
            self.batch_sizes.append(input_ids.shape[0])
            return type("Output", (), {"logits": torch.zeros((*input_ids.shape, 6))})()

    model = UniformModel()
    mass = _sequence_prob_mass(
        model,
        {
            "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        },
        ((1, 2), (1, 3), (4, 2)),
    )

    assert mass == pytest.approx(3 * (1 / 6) * (1 / 6))
    assert model.calls == 2
    assert model.batch_sizes == [1, 2]


def test_sequence_prob_mass_batch_matches_scalar_path():
    class UniformModel:
        def __call__(self, *, input_ids, attention_mask):
            del attention_mask
            return type("Output", (), {"logits": torch.zeros((*input_ids.shape, 7))})()

    model = UniformModel()
    batch_inputs = {
        "input_ids": torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
    }
    sequences = ((1,), (2, 3), (4, 5, 6))

    batch = _sequence_prob_mass_batch(model, batch_inputs, sequences)
    scalar = [
        _sequence_prob_mass(
            model,
            {
                "input_ids": batch_inputs["input_ids"][index : index + 1],
                "attention_mask": batch_inputs["attention_mask"][index : index + 1],
            },
            sequences,
        )
        for index in range(2)
    ]

    assert batch == pytest.approx(scalar)


def test_sequence_prob_mass_cached_batch_matches_scalar_path():
    class FakeCache:
        def batch_repeat_interleave(self, _repeats):
            pass

    class UniformModel:
        def __call__(self, *, input_ids, attention_mask, **_kwargs):
            del attention_mask
            return type(
                "Output",
                (),
                {
                    "logits": torch.zeros((*input_ids.shape, 7)),
                    "past_key_values": FakeCache(),
                },
            )()

    model = UniformModel()
    batch_inputs = {
        "input_ids": torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
    }
    sequences = ((1,), (2, 3), (4, 5, 6))

    cached = _sequence_prob_mass_batch_cached(
        model,
        batch_inputs,
        sequences,
        cache_branch_batch_size=2,
    )
    scalar = [
        _sequence_prob_mass(
            model,
            {
                "input_ids": batch_inputs["input_ids"][index : index + 1],
                "attention_mask": batch_inputs["attention_mask"][index : index + 1],
            },
            sequences,
        )
        for index in range(2)
    ]

    assert cached == pytest.approx(scalar)


def test_sequence_prob_mass_cached_multi_shares_prefix_forward():
    class FakeCache:
        def batch_repeat_interleave(self, _repeats):
            pass

    class UniformModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, *, input_ids, attention_mask, **_kwargs):
            del attention_mask
            self.calls += 1
            return type(
                "Output",
                (),
                {
                    "logits": torch.zeros((*input_ids.shape, 7)),
                    "past_key_values": FakeCache(),
                },
            )()

    model = UniformModel()
    batch_inputs = {
        "input_ids": torch.tensor([[0, 1], [0, 2]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
    }

    masses = _sequence_prob_mass_batch_cached_multi(
        model,
        batch_inputs,
        {
            "short": ((1,), (2, 3)),
            "long": ((1, 4), (5, 6)),
        },
        cache_branch_batch_size=4,
    )

    assert masses["short"] == pytest.approx([1 / 7 + 1 / 49, 1 / 7 + 1 / 49])
    assert masses["long"] == pytest.approx([2 / 49, 2 / 49])
    assert model.calls == 2


def test_sequence_prob_mass_cached_multi_matches_scalar_with_nonuniform_logits():
    class FakeCache:
        def batch_repeat_interleave(self, _repeats):
            pass

    class NonuniformModel:
        def __call__(self, *, input_ids, attention_mask, **_kwargs):
            del attention_mask
            vocab = 9
            base_logits = torch.arange(vocab, dtype=torch.float32, device=input_ids.device)
            token_offsets = input_ids.to(torch.float32).unsqueeze(-1) * 0.03
            logits = base_logits.reshape(1, 1, vocab) + token_offsets
            return type(
                "Output",
                (),
                {
                    "logits": logits,
                    "past_key_values": FakeCache(),
                },
            )()

    model = NonuniformModel()
    batch_inputs = {
        "input_ids": torch.tensor([[0, 1], [0, 4]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
    }
    feature_sequences: dict[str, tuple[tuple[int, ...], ...]] = {
        "short": ((1,), (2, 3)),
        "long": ((1, 4), (5, 6), (7, 8)),
    }

    cached = _sequence_prob_mass_batch_cached_multi(
        model,
        batch_inputs,
        feature_sequences,
        cache_branch_batch_size=2,
    )
    for feature, sequences in feature_sequences.items():
        scalar = [
            _sequence_prob_mass(
                model,
                {
                    "input_ids": batch_inputs["input_ids"][index : index + 1],
                    "attention_mask": batch_inputs["attention_mask"][index : index + 1],
                },
                sequences,
            )
            for index in range(2)
        ]
        assert cached[feature] == pytest.approx(scalar)
