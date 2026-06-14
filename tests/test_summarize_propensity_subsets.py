import csv
import json

from slop_sftdiv.cli.summarize_propensity_subsets import (
    build_parser,
    run_summarize_propensity_subsets,
)


def _write_csv(path, rows):
    columns = [
        "source",
        "record_id",
        "role",
        "feature",
        "opportunity_kind",
        "char_offset",
        "prefix_tokens",
        "reference_initiates",
        "matched_subtype",
        "prob_mass",
        "initiator_sequences",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_propensity_subsets_joins_metadata_and_logs(tmp_path, monkeypatch):
    opportunities_path = tmp_path / "opportunities.csv"
    metadata_path = tmp_path / "metadata.jsonl"
    output_path = tmp_path / "subset_summary.csv"
    markdown_path = tmp_path / "subset_summary.md"
    _write_csv(
        opportunities_path,
        [
            {
                "source": "pkg",
                "record_id": "a",
                "role": "target_response",
                "feature": "slop_lexicon",
                "opportunity_kind": "token_start",
                "char_offset": "0",
                "prefix_tokens": "1",
                "reference_initiates": "1",
                "matched_subtype": "delve",
                "prob_mass": "0.5",
                "initiator_sequences": "2",
            },
            {
                "source": "pkg",
                "record_id": "a",
                "role": "target_response",
                "feature": "neutral_common_controls",
                "opportunity_kind": "token_start",
                "char_offset": "1",
                "prefix_tokens": "1",
                "reference_initiates": "1",
                "matched_subtype": "the",
                "prob_mass": "0.25",
                "initiator_sequences": "2",
            },
            {
                "source": "pkg",
                "record_id": "b",
                "role": "target_response",
                "feature": "slop_lexicon",
                "opportunity_kind": "token_start",
                "char_offset": "0",
                "prefix_tokens": "1",
                "reference_initiates": "0",
                "matched_subtype": "",
                "prob_mass": "0.1",
                "initiator_sequences": "2",
            },
        ],
    )
    metadata_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "phase2_prompt_id": "a",
                        "text_role": "target_response",
                        "reference_subset": "code",
                        "prompt": "hidden prompt",
                        "text": "hidden text",
                    }
                ),
                json.dumps(
                    {
                        "phase2_prompt_id": "b",
                        "text_role": "target_response",
                        "reference_subset": "unknown",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    logged_payloads = []
    logged_tables = {}

    class FakeRun:
        def log(self, payload):
            logged_payloads.append(payload)

        def finish(self):
            pass

    monkeypatch.setattr(
        "slop_sftdiv.cli.summarize_propensity_subsets.init_wandb",
        lambda **_kwargs: FakeRun(),
    )
    monkeypatch.setattr(
        "slop_sftdiv.cli.summarize_propensity_subsets.log_summary_table",
        lambda _run, table_name, rows: logged_tables.setdefault(table_name, rows),
    )
    args = build_parser().parse_args(
        [
            "--opportunities",
            f"sft={opportunities_path}",
            "--metadata-jsonl",
            str(metadata_path),
            "--normalization-feature",
            "neutral_common_controls",
            "--bootstrap-samples",
            "0",
            "--output",
            str(output_path),
            "--summary-output",
            str(markdown_path),
            "--wandb-mode",
            "disabled",
        ]
    )

    rows = run_summarize_propensity_subsets(args)

    by_subset_feature = {
        (row["reference_subset"], row["feature"]): row
        for row in rows
    }
    assert by_subset_feature[("code", "slop_lexicon")]["opportunities"] == 1
    assert by_subset_feature[("code", "slop_lexicon")]["normalized_amplification_factor"] == 2.0
    assert by_subset_feature[("unknown", "slop_lexicon")]["opportunities"] == 1
    assert output_path.exists()
    assert markdown_path.exists()
    assert logged_payloads[-1]["propensity_reference_subsets/input_rows"] == 3
    assert logged_tables["propensity_reference_subset_summary"] == rows
    assert "hidden prompt" not in json.dumps(logged_tables)
    assert "hidden text" not in json.dumps(logged_tables)
