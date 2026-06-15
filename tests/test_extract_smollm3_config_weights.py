import csv

import yaml

from slop_sftdiv.cli.extract_smollm3_config_weights import build_parser, run


def _write_yaml(path, payload):
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_extract_smollm3_config_weights_uses_stage_boundaries(tmp_path):
    config = tmp_path / "stage3_9T_11T.yaml"
    detail = tmp_path / "detail.csv"
    aggregate = tmp_path / "aggregate.csv"
    groups = tmp_path / "groups.csv"
    summary = tmp_path / "summary.md"
    base_payload = {
        "parallelism": {"dp": 2},
        "tokens": {
            "batch_accumulation_per_replica": 1,
            "micro_batch_size": 1,
            "sequence_length": 10,
            "train_steps": 20,
        },
        "data_stages": [
            {
                "name": "stable",
                "start_training_step": 1,
                "data": {
                    "dataset": {
                        "dataset_read_path": ["/data/fineweb-edu", "/data/stackexchange"],
                        "dataset_folder": ["s3://fineweb", "s3://stackexchange"],
                        "dataset_weights": [0.75, 0.25],
                    }
                },
            },
            {
                "name": "decay",
                "start_training_step": 11,
                "data": {
                    "dataset": {
                        "dataset_read_path": ["/data/fineweb-edu", "/data/stack-edu-Python"],
                        "dataset_folder": ["s3://fineweb", "s3://python"],
                        "dataset_weights": [0.5, 0.5],
                    }
                },
            },
        ],
    }
    _write_yaml(config, base_payload)

    args = build_parser().parse_args(
        [
            "--config",
            str(config),
            "--detail-output",
            str(detail),
            "--aggregate-output",
            str(aggregate),
            "--group-output",
            str(groups),
            "--summary-output",
            str(summary),
        ]
    )

    rows = run(args)

    assert len(rows) == 4
    assert rows[0].segment_steps == 10
    assert rows[0].tokens_per_step == 20
    assert rows[0].segment_tokens == 200
    with aggregate.open(encoding="utf-8", newline="") as handle:
        by_source = {row["source_name"]: row for row in csv.DictReader(handle)}
    assert float(by_source["fineweb-edu"]["tokens"]) == 250.0
    assert float(by_source["stackexchange"]["tokens"]) == 50.0
    assert float(by_source["stack-edu-Python"]["tokens"]) == 100.0
    assert float(by_source["fineweb-edu"]["share"]) == 0.625
    with groups.open(encoding="utf-8", newline="") as handle:
        by_group = {row["source_group"]: row for row in csv.DictReader(handle)}
    assert float(by_group["web"]["tokens"]) == 250.0
    assert float(by_group["qa_forum"]["tokens"]) == 50.0
    assert float(by_group["code"]["tokens"]) == 100.0
    summary_text = summary.read_text(encoding="utf-8")
    assert "Total config-implied tokens" in summary_text
    assert "Feature-Rate Coverage" in summary_text
    assert "Currently Sampled Phase 3 Baseline Sources" not in summary_text
