import csv
import json

import pandas as pd

from slop_sftdiv.cli.sample_corpus import build_parser, run_sample


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_run_sample_writes_retained_jsonl_and_manifest_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    input_path = tmp_path / "dolma.jsonl"
    output_path = tmp_path / "sample.jsonl"
    manifest_path = tmp_path / "sample_manifest.parquet"
    manifest_csv_path = tmp_path / "sample_manifest.csv"
    manifest_json_path = tmp_path / "sample_manifest.json"
    summary_path = tmp_path / "sample_manifest.md"
    _write_jsonl(
        input_path,
        [
            {
                "id": "web-1",
                "text": "Common Crawl document. It has two sentences.",
                "source": "common_crawl",
                "metadata": json.dumps({"cc_dump": "CC-MAIN-2024-18"}),
            },
            {
                "id": "wiki-1",
                "text": "Wikipedia document.",
                "source": "wikipedia",
            },
            {
                "id": "science-1",
                "text": "Scientific paper text.",
                "source": "olmocr_science_pdfs",
            },
            {
                "id": "code-1",
                "text": "def f(): pass",
                "source": "github",
            },
        ],
    )
    logged_tables = {}
    metric_payloads = []

    class FakeRun:
        def log(self, payload):
            metric_payloads.append(payload)

        def finish(self):
            pass

    monkeypatch.setattr("slop_sftdiv.cli.sample_corpus.init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr(
        "slop_sftdiv.cli.sample_corpus.log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--corpus",
            "dolma3_pretrain_sample",
            "--ladder",
            "olmo3",
            "--source-dataset",
            "allenai/dolma3_mix-6T-1025-7B",
            "--target-rows-per-stratum",
            "1",
            "--exclude-stratum",
            "code",
            "--output",
            str(output_path),
            "--manifest-output",
            str(manifest_path),
            "--manifest-csv-output",
            str(manifest_csv_path),
            "--manifest-json-output",
            str(manifest_json_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    manifest_rows = run_sample(args)

    retained = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(retained) == 3
    assert {row["stratum"] for row in retained} == {"web_cc", "wiki", "scientific"}
    assert all(row["source_dataset"] == "allenai/dolma3_mix-6T-1025-7B" for row in retained)
    assert all(row["text"] for row in retained)
    assert all(row["content_hash"] for row in retained)
    assert all(row["token_count"] > 0 for row in retained)
    assert len(manifest_rows) == 3
    assert all("text" not in row for row in manifest_rows)
    assert set(pd.read_parquet(manifest_path)["stratum"]) == {"web_cc", "wiki", "scientific"}
    with manifest_csv_path.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    assert json.loads(manifest_json_path.read_text(encoding="utf-8"))["summary"]["rows_retained"] == 3
    assert "`rows_retained` | 3" in summary_path.read_text(encoding="utf-8")
    assert metric_payloads[-1]["corpus/rows"] == 3
    assert metric_payloads[-1]["corpus/strata"] == 3
    logged_manifest_payload = json.dumps(logged_tables["corpus_manifest_sample"])
    assert "Common Crawl document" not in logged_manifest_payload
    assert "Wikipedia document" not in logged_manifest_payload
    assert "Scientific paper text" not in logged_manifest_payload


def test_run_sample_hash_reservoir_is_deterministic_within_stratum(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    input_path = tmp_path / "rows.jsonl"
    rows = [
        {"id": f"web-{index}", "text": f"Web row {index}.", "source": "common_crawl"}
        for index in range(10)
    ]
    _write_jsonl(input_path, rows)

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    monkeypatch.setattr("slop_sftdiv.cli.sample_corpus.init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr("slop_sftdiv.cli.sample_corpus.log_summary_table", lambda *_args: None)

    def run_once(prefix):
        args = build_parser().parse_args(
            [
                "--input",
                str(input_path),
                "--corpus",
                "dolma3_pretrain_sample",
                "--ladder",
                "olmo3",
                "--source-dataset",
                "allenai/dolma3_mix-6T-1025-7B",
                "--target-rows-per-stratum",
                "3",
                "--output",
                str(tmp_path / f"{prefix}.jsonl"),
                "--manifest-output",
                str(tmp_path / f"{prefix}.parquet"),
                "--wandb-mode",
                "disabled",
            ]
        )
        return [row["doc_id"] for row in run_sample(args)]

    assert run_once("a") == run_once("b")


def test_run_sample_expands_preference_roles_without_splitting_pairs(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    input_path = tmp_path / "pairs.jsonl"
    output_path = tmp_path / "dpo_sample.jsonl"
    manifest_path = tmp_path / "dpo_manifest.parquet"
    _write_jsonl(
        input_path,
        [
            {
                "prompt_id": "pair-1",
                "preference_type": "delta_learning",
                "chosen_model": "strong",
                "rejected_model": "weak",
                "chosen": [
                    {"role": "user", "content": "Explain tests."},
                    {"role": "assistant", "content": "Chosen answer with detail."},
                ],
                "rejected": [
                    {"role": "user", "content": "Explain tests."},
                    {"role": "assistant", "content": "Rejected answer."},
                ],
            }
        ],
    )
    logged_tables = {}

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    monkeypatch.setattr("slop_sftdiv.cli.sample_corpus.init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr(
        "slop_sftdiv.cli.sample_corpus.log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--corpus",
            "dolci_instruct_dpo",
            "--ladder",
            "olmo3",
            "--source-dataset",
            "allenai/Dolci-Instruct-DPO",
            "--target-rows",
            "1",
            "--expand-preference-roles",
            "--output",
            str(output_path),
            "--manifest-output",
            str(manifest_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    manifest_rows = run_sample(args)

    retained = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["text_role"] for row in retained] == ["chosen", "rejected"]
    assert {row["prompt_id"] for row in retained} == {"pair-1"}
    assert {row["source_record_id"] for row in retained} == {"pair-1"}
    assert {row["pair_id"] for row in retained} == {"pair-1#row0"}
    assert {row["doc_id"] for row in retained} == {"pair-1#row0:chosen", "pair-1#row0:rejected"}
    assert [row["text"] for row in retained] == ["Chosen answer with detail.", "Rejected answer."]
    assert [row["text_role"] for row in manifest_rows] == ["chosen", "rejected"]
    assert all("text" not in row for row in manifest_rows)
    assert list(pd.read_parquet(manifest_path)["text_role"]) == ["chosen", "rejected"]
    logged_manifest_payload = json.dumps(logged_tables["corpus_manifest_sample"])
    assert "Chosen answer" not in logged_manifest_payload
    assert "Rejected answer" not in logged_manifest_payload


def test_run_sample_skips_incomplete_preference_pairs_when_expanding(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    input_path = tmp_path / "pairs.jsonl"
    output_path = tmp_path / "dpo_sample.jsonl"
    manifest_path = tmp_path / "dpo_manifest.parquet"
    _write_jsonl(
        input_path,
        [
            {"prompt_id": "missing-rejected", "chosen": "Chosen only."},
            {"prompt_id": "complete", "chosen": "Chosen.", "rejected": "Rejected."},
        ],
    )

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    monkeypatch.setattr("slop_sftdiv.cli.sample_corpus.init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr("slop_sftdiv.cli.sample_corpus.log_summary_table", lambda *_args: None)
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--corpus",
            "dolci_instruct_dpo",
            "--ladder",
            "olmo3",
            "--source-dataset",
            "allenai/Dolci-Instruct-DPO",
            "--target-rows",
            "1",
            "--expand-preference-roles",
            "--output",
            str(output_path),
            "--manifest-output",
            str(manifest_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    manifest_rows = run_sample(args)

    assert [row["prompt_id"] for row in manifest_rows] == ["complete", "complete"]
    assert [row["pair_id"] for row in manifest_rows] == ["complete#row1", "complete#row1"]
    assert [row["text_role"] for row in manifest_rows] == ["chosen", "rejected"]


def test_run_sample_keeps_duplicate_prompt_ids_as_distinct_pairs(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "wandb"))
    input_path = tmp_path / "pairs.jsonl"
    output_path = tmp_path / "dpo_sample.jsonl"
    manifest_path = tmp_path / "dpo_manifest.parquet"
    _write_jsonl(
        input_path,
        [
            {"prompt_id": "duplicate", "chosen": "Chosen one.", "rejected": "Rejected one."},
            {"prompt_id": "duplicate", "chosen": "Chosen two.", "rejected": "Rejected two."},
        ],
    )

    class FakeRun:
        def log(self, _payload):
            pass

        def finish(self):
            pass

    monkeypatch.setattr("slop_sftdiv.cli.sample_corpus.init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr("slop_sftdiv.cli.sample_corpus.log_summary_table", lambda *_args: None)
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--corpus",
            "dolci_instruct_dpo",
            "--ladder",
            "olmo3",
            "--source-dataset",
            "allenai/Dolci-Instruct-DPO",
            "--target-rows",
            "2",
            "--expand-preference-roles",
            "--output",
            str(output_path),
            "--manifest-output",
            str(manifest_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    manifest_rows = run_sample(args)

    assert [row["prompt_id"] for row in manifest_rows] == ["duplicate"] * 4
    assert {row["pair_id"] for row in manifest_rows} == {"duplicate#row0", "duplicate#row1"}
    assert all(
        [row["text_role"] for row in manifest_rows if row["pair_id"] == pair_id]
        == ["chosen", "rejected"]
        for pair_id in {"duplicate#row0", "duplicate#row1"}
    )
