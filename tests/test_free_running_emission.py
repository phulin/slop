import csv
import json

from slop_sftdiv.cli.free_running_emission import build_parser, run_free_running_emission


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [max(1, ord(char) % 97) for char in text[:5]]


def test_free_running_emission_writes_generation_cache_and_redacted_wandb(tmp_path, monkeypatch):
    input_path = tmp_path / "prompts.jsonl"
    generations_path = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "summary.csv"
    input_path.write_text(
        json.dumps(
            {
                "phase2_prompt_id": "pkg-p1",
                "prompt_id": "p1",
                "prompt": "Give a short answer.",
                "text": "Reference.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    logged_tables = {}
    logged_payloads = []

    class FakeRun:
        def log(self, payload):
            logged_payloads.append(payload)

        def finish(self):
            pass

    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission._load_model",
        lambda **_kwargs: (FakeTokenizer(), object(), "cpu"),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission._generate_batch",
        lambda prompts, **_kwargs: [
            ("Certainly, delve into robust tests.", 5, 5) for _prompt in prompts
        ],
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission.log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )
    args = build_parser().parse_args(
        [
            "--model",
            "fake",
            "--input",
            str(input_path),
            "--sample-size",
            "1",
            "--feature",
            "slop_lexicon",
            "--feature",
            "stock_openers",
            "--temperature",
            "0.0",
            "--generations-output",
            str(generations_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    summary = run_free_running_emission(args)

    assert generations_path.exists()
    generation_rows = [json.loads(line) for line in generations_path.read_text().splitlines()]
    assert generation_rows[0]["generation"] == "Certainly, delve into robust tests."
    assert set(generation_rows[0]) == {
        "completion_index",
        "feature_count_total",
        "features_json",
        "generated_tokens",
        "generation",
        "generation_chars",
        "prompt_tokens",
        "record_id",
        "repeated_feature_count_total",
        "repeated_features_json",
        "row_index",
        "seed",
        "source",
        "source_kind",
        "split",
        "temperature",
        "top_p",
    }
    with summary_path.open(encoding="utf-8", newline="") as handle:
        summary_csv = list(csv.DictReader(handle))
    assert summary == summary_csv or len(summary) == len(summary_csv)
    assert {row["feature"] for row in summary} == {"slop_lexicon", "stock_openers"}
    assert logged_payloads[-1]["generation/generations"] == 1
    assert "Certainly, delve" not in json.dumps(logged_tables["generation_samples_redacted"])


def test_free_running_emission_batches_generation_and_tracks_repeats(tmp_path, monkeypatch):
    input_path = tmp_path / "prompts.jsonl"
    generations_path = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "summary.csv"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "p1", "prompt": "One", "text": "Reference."}),
                json.dumps({"id": "p2", "prompt": "Two", "text": "Reference."}),
                json.dumps({"id": "p3", "prompt": "Three", "text": "Reference."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    batch_sizes = []

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    def fake_generate_batch(prompts, **_kwargs):
        batch_sizes.append(len(prompts))
        return [("delve delve", 3, 2) for _prompt in prompts]

    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission._load_model",
        lambda **_kwargs: (FakeTokenizer(), object(), "cpu"),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission._generate_batch",
        fake_generate_batch,
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission.log_summary_table",
        lambda *_args: None,
    )
    args = build_parser().parse_args(
        [
            "--model",
            "fake",
            "--input",
            str(input_path),
            "--sample-size",
            "3",
            "--feature",
            "slop_lexicon",
            "--temperature",
            "0.0",
            "--top-p",
            "0.90",
            "--top-p",
            "0.95",
            "--generation-batch-size",
            "2",
            "--generations-output",
            str(generations_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    summary = run_free_running_emission(args)

    assert batch_sizes == [2, 2, 1, 1]
    assert len(generations_path.read_text().splitlines()) == 6
    assert {row["top_p"] for row in summary} == {0.9, 0.95}
    for row in summary:
        assert row["count"] == 6
        assert row["repeated_count"] == 3
        assert row["generations_with_feature"] == 3
        assert row["repeat_generations"] == 3
        assert row["repeat_share_after_first"] == 0.5


def test_free_running_emission_streams_completed_batches_before_failure(tmp_path, monkeypatch):
    input_path = tmp_path / "prompts.jsonl"
    generations_path = tmp_path / "generations.jsonl"
    summary_path = tmp_path / "summary.csv"
    input_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "p1", "prompt": "One", "text": "Reference."}),
                json.dumps({"id": "p2", "prompt": "Two", "text": "Reference."}),
                json.dumps({"id": "p3", "prompt": "Three", "text": "Reference."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    calls = 0

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    def fake_generate_batch(prompts, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated generation failure")
        return [("delve", 3, 1) for _prompt in prompts]

    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission._load_model",
        lambda **_kwargs: (FakeTokenizer(), object(), "cpu"),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission._generate_batch",
        fake_generate_batch,
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.free_running_emission.log_summary_table",
        lambda *_args: None,
    )
    args = build_parser().parse_args(
        [
            "--model",
            "fake",
            "--input",
            str(input_path),
            "--sample-size",
            "3",
            "--feature",
            "slop_lexicon",
            "--temperature",
            "0.0",
            "--generation-batch-size",
            "2",
            "--generations-output",
            str(generations_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    try:
        run_free_running_emission(args)
    except RuntimeError as exc:
        assert str(exc) == "simulated generation failure"
    else:
        raise AssertionError("expected simulated generation failure")

    rows = [json.loads(line) for line in generations_path.read_text().splitlines()]
    assert len(rows) == 2
    assert {row["record_id"] for row in rows} == {"p1", "p2"}
