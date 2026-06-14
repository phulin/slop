import csv
import json

from slop_sftdiv.cli.prepare_phase2_prompts import build_parser, run_prepare_phase2_prompts


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_prepare_phase2_prompts_deduplicates_and_redacts_wandb(tmp_path, monkeypatch):
    input_path = tmp_path / "sft.jsonl"
    output_path = tmp_path / "phase2_prompts.jsonl"
    manifest_path = tmp_path / "phase2_prompts_manifest.csv"
    summary_path = tmp_path / "phase2_prompts_summary.json"
    _write_jsonl(
        input_path,
        [
            {
                "id": "p1",
                "source_dataset": "unit/source",
                "stratum": "synthetic_math",
                "provenance": "unit",
                "messages": [
                    {"role": "user", "content": "Explain robust testing with examples."},
                    {"role": "assistant", "content": "Use assertions and fixtures."},
                ],
            },
            {
                "id": "p2",
                "messages": [
                    {"role": "user", "content": "Explain robust testing with example."},
                    {"role": "assistant", "content": "Use tests."},
                ],
            },
            {
                "id": "p3",
                "messages": [
                    {"role": "user", "content": "Summarize caching."},
                    {"role": "assistant", "content": "Caching stores reusable values."},
                ],
            },
            {
                "id": "p3",
                "messages": [
                    {"role": "user", "content": "Summarize caching differently."},
                    {"role": "assistant", "content": "Duplicate prompt id."},
                ],
            },
            {"id": "missing", "text": "No prompt."},
        ],
    )
    logged_tables = {}
    logged_payloads = []

    class FakeRun:
        def log(self, payload):
            logged_payloads.append(payload)

        def finish(self):
            pass

    monkeypatch.setattr(
        "slop_sftdiv.cli.prepare_phase2_prompts.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.prepare_phase2_prompts.log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--sample-size",
            "10",
            "--sampling-strategy",
            "first",
            "--near-duplicate-threshold",
            "0.5",
            "--minhash-shingle-tokens",
            "1",
            "--output",
            str(output_path),
            "--manifest-output",
            str(manifest_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    manifest_rows = run_prepare_phase2_prompts(args)

    package_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["phase2_prompt_id"] for row in package_rows] == ["p1", "p3"]
    assert package_rows[0]["prompt"] == "Explain robust testing with examples."
    assert package_rows[0]["text"] == "Use assertions and fixtures."
    assert package_rows[0]["source_dataset"] == "unit/source"
    assert package_rows[0]["stratum"] == "synthetic_math"
    assert package_rows[0]["provenance"] == "unit"
    assert all("prompt" not in row and "text" not in row for row in manifest_rows)
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        assert [row["phase2_prompt_id"] for row in csv.DictReader(handle)] == ["p1", "p3"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["near_duplicate_prompt"] == 1
    assert summary["duplicate_prompt_id"] == 1
    assert summary["missing_prompt"] == 1
    assert logged_payloads[-1]["phase2_prompts/selected_prompts"] == 2
    logged = json.dumps(logged_tables["phase2_prompt_manifest"])
    assert "Explain robust testing" not in logged
    assert "Use assertions" not in logged


def test_prepare_phase2_prompts_hash_selection_is_deterministic(tmp_path, monkeypatch):
    input_path = tmp_path / "sft.jsonl"
    _write_jsonl(
        input_path,
        [
            {"id": f"p{index}", "prompt": f"Prompt {index}", "text": f"Target {index}"}
            for index in range(12)
        ],
    )

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    monkeypatch.setattr("slop_sftdiv.cli.prepare_phase2_prompts.init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr("slop_sftdiv.cli.prepare_phase2_prompts.log_summary_table", lambda *_args: None)

    def run_once(prefix):
        args = build_parser().parse_args(
            [
                "--input",
                str(input_path),
                "--sample-size",
                "4",
                "--seed",
                "99",
                "--output",
                str(tmp_path / f"{prefix}.jsonl"),
                "--manifest-output",
                str(tmp_path / f"{prefix}.csv"),
                "--wandb-mode",
                "disabled",
            ]
        )
        return [row["phase2_prompt_id"] for row in run_prepare_phase2_prompts(args)]

    assert run_once("a") == run_once("b")


def test_prepare_phase2_prompts_materializes_metadata_bucket(tmp_path, monkeypatch):
    input_path = tmp_path / "sft.jsonl"
    output_path = tmp_path / "phase2_prompts.jsonl"
    manifest_path = tmp_path / "phase2_prompts_manifest.csv"
    bucket_map_path = tmp_path / "reference_subset_map.json"
    _write_jsonl(
        input_path,
        [
            {
                "id": "aya",
                "prompt": "Explain one thing.",
                "text": "One answer.",
                "source_dataset": "Aya",
            },
            {
                "id": "code",
                "prompt": "Write code.",
                "text": "Use a function.",
                "source_dataset": "Evol CodeAlpaca",
            },
        ],
    )
    bucket_map_path.write_text(
        json.dumps(
            {
                "output_field": "reference_subset",
                "source_field": "source_dataset",
                "default": "unknown",
                "values": {
                    "Aya": "human_reference",
                    "Evol CodeAlpaca": "code",
                },
            }
        ),
        encoding="utf-8",
    )
    logged_tables = {}

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    monkeypatch.setattr("slop_sftdiv.cli.prepare_phase2_prompts.init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr(
        "slop_sftdiv.cli.prepare_phase2_prompts.log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )

    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--sample-size",
            "2",
            "--sampling-strategy",
            "first",
            "--metadata-bucket-map",
            str(bucket_map_path),
            "--output",
            str(output_path),
            "--manifest-output",
            str(manifest_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    manifest_rows = run_prepare_phase2_prompts(args)
    package_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    by_id = {row["phase2_prompt_id"]: row for row in package_rows}

    assert by_id["aya"]["reference_subset"] == "human_reference"
    assert by_id["code"]["reference_subset"] == "code"
    assert {row["reference_subset"] for row in manifest_rows} == {"human_reference", "code"}
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest_csv_rows = list(csv.DictReader(handle))
    assert {row["reference_subset"] for row in manifest_csv_rows} == {"human_reference", "code"}
    logged = json.dumps(logged_tables["phase2_prompt_manifest"])
    assert "Explain one thing" not in logged
    assert "One answer" not in logged


def test_prepare_phase2_prompts_can_require_reference_feature(tmp_path, monkeypatch):
    input_path = tmp_path / "sft.jsonl"
    output_path = tmp_path / "phase2_positive_prompts.jsonl"
    manifest_path = tmp_path / "phase2_positive_prompts_manifest.csv"
    _write_jsonl(
        input_path,
        [
            {"id": "plain", "prompt": "Explain tests.", "text": "Tests check behavior."},
            {"id": "slop", "prompt": "Explain APIs.", "text": "A robust API can unlock value."},
            {"id": "neutral", "prompt": "Explain examples.", "text": "For example, use a fixture."},
        ],
    )

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    monkeypatch.setattr("slop_sftdiv.cli.prepare_phase2_prompts.init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr("slop_sftdiv.cli.prepare_phase2_prompts.log_summary_table", lambda *_args: None)
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--sample-size",
            "4",
            "--sampling-strategy",
            "first",
            "--require-reference-feature",
            "slop_lexicon",
            "--output",
            str(output_path),
            "--manifest-output",
            str(manifest_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    manifest_rows = run_prepare_phase2_prompts(args)
    package_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert [row["phase2_prompt_id"] for row in package_rows] == ["slop"]
    assert [row["phase2_prompt_id"] for row in manifest_rows] == ["slop"]
