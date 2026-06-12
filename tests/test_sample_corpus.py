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
