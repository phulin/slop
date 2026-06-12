import csv

from slop_sftdiv.cli.manifest_artifacts import build_parser, manifest_path, run_manifest


def test_manifest_path_hashes_and_counts_csv_records(tmp_path):
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text('a,b\n1,"two\nlines"\n3,4\n', encoding="utf-8")

    row = manifest_path(csv_path)

    assert row.path == str(csv_path)
    assert row.format == "csv"
    assert row.records == 2
    assert len(row.sha256) == 64
    assert row.bytes == csv_path.stat().st_size


def test_run_manifest_writes_outputs_and_logs_wandb(tmp_path, monkeypatch):
    input_path = tmp_path / "rows.csv"
    output_path = tmp_path / "manifest.csv"
    json_path = tmp_path / "manifest.json"
    summary_path = tmp_path / "manifest.md"
    input_path.write_text("a\n1\n", encoding="utf-8")
    logged = []
    tables = {}

    class FakeRun:
        def log(self, payload):
            logged.append(payload)

        def finish(self):
            pass

    monkeypatch.setattr("slop_sftdiv.cli.manifest_artifacts.init_wandb", lambda **_kwargs: FakeRun())
    monkeypatch.setattr(
        "slop_sftdiv.cli.manifest_artifacts.log_summary_table",
        lambda _run, table_name, rows: tables.setdefault(table_name, rows),
    )
    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--json-output",
            str(json_path),
            "--summary-output",
            str(summary_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    rows = run_manifest(args)

    assert rows[0]["records"] == 1
    assert logged[-1]["manifest/artifacts"] == 1
    assert tables["artifact_manifest"] == rows
    assert json_path.exists()
    assert summary_path.exists()
    with output_path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["records"] == "1"
