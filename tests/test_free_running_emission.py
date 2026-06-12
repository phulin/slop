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
        json.dumps({"id": "p1", "prompt": "Give a short answer.", "text": "Reference."}) + "\n",
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
        "slop_sftdiv.cli.free_running_emission._generate_one",
        lambda **_kwargs: ("Certainly, delve into robust tests.", 5, 5),
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
    with summary_path.open(encoding="utf-8", newline="") as handle:
        summary_csv = list(csv.DictReader(handle))
    assert summary == summary_csv or len(summary) == len(summary_csv)
    assert {row["feature"] for row in summary} == {"slop_lexicon", "stock_openers"}
    assert logged_payloads[-1]["generation/generations"] == 1
    assert "Certainly, delve" not in json.dumps(logged_tables["generation_samples_redacted"])
