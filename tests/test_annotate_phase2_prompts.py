import csv
import json

from slop_sftdiv.cli.annotate_phase2_prompts import build_parser, run_annotate_phase2_prompts


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_annotate_phase2_prompts_preserves_rows_and_redacts_wandb(tmp_path, monkeypatch):
    input_path = tmp_path / "phase2_prompts.jsonl"
    output_path = tmp_path / "phase2_prompts_annotated.jsonl"
    manifest_path = tmp_path / "phase2_prompts_annotated_manifest.csv"
    summary_path = tmp_path / "phase2_prompts_annotated_summary.json"
    map_path = tmp_path / "bucket_map.json"
    _write_jsonl(
        input_path,
        [
            {
                "phase2_prompt_id": "a",
                "prompt": "Explain testing.",
                "text": "Use fixtures.",
                "source_dataset": "Aya",
            },
            {
                "phase2_prompt_id": "b",
                "prompt": "Write code.",
                "text": "Use functions.",
                "source_dataset": "Evol CodeAlpaca",
            },
        ],
    )
    map_path.write_text(
        json.dumps(
            {
                "output_field": "reference_subset",
                "source_field": "source_dataset",
                "default": "unknown",
                "values": {"Evol CodeAlpaca": "code"},
            }
        ),
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

    monkeypatch.setattr("slop_sftdiv.cli.annotate_phase2_prompts.init_wandb", fake_init)
    monkeypatch.setattr(
        "slop_sftdiv.cli.annotate_phase2_prompts.log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )

    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--metadata-bucket-map",
            str(map_path),
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

    manifest_rows = run_annotate_phase2_prompts(args)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["phase2_prompt_id"] for row in rows] == ["a", "b"]
    assert [row["reference_subset"] for row in rows] == ["unknown", "code"]
    assert all("prompt" not in row and "text" not in row for row in manifest_rows)
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["phase2_prompt_id"] for row in csv_rows] == ["a", "b"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["rows"] == 2
    assert {"field": "reference_subset", "value": "code", "count": 1} in summary["bucket_counts"]
    assert logged_payloads[-1]["phase2_prompt_annotation/rows"] == 2
    assert init_kwargs["tags"][:4] == ["stage2", "phase2", "prompts", "annotation"]
    logged = json.dumps(logged_tables)
    assert "Explain testing" not in logged
    assert "Use fixtures" not in logged
