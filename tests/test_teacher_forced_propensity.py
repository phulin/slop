import csv
import json

from slop_sftdiv.cli.teacher_forced_propensity import build_parser, run_teacher_forced_propensity


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
        lambda **_kwargs: (FakeTokenizer(), object(), "cpu"),
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
    assert rows
    assert {row["feature"] for row in rows} == {"contrastive_negation", "stock_openers"}
    assert all(int(row["initiator_sequences"]) > 0 for row in rows)
    assert "text" not in rows[0]
    assert "Great question" not in json.dumps(logged_tables)
    assert summary == logged_tables["propensity_summary"]
    assert logged_payloads[-1]["propensity/opportunities"] == len(rows)
    assert init_kwargs["tags"][:4] == ["stage2", "phase2", "teacher-forced", "smoke"]
