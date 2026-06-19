import csv
import json

import pytest

from slop_sftdiv.cli.pybiber_full import _ensure_unique_doc_ids, build_parser, run
from slop_sftdiv.features.pybiber_full import (
    PYBIBER_FEATURES,
    extract_pybiber_full_features,
)


pytest.importorskip("pybiber")
pytest.importorskip("spacy")


def test_extract_pybiber_full_features_returns_67_feature_vector():
    features = extract_pybiber_full_features(
        "I think you should test this because it is useful.",
        doc_id="doc-a",
    )

    assert features.doc_id == "doc-a"
    assert len(PYBIBER_FEATURES) == 67
    assert set(features.rates_or_values) == set(PYBIBER_FEATURES)
    assert features.rates_or_values["f_03_present_tense"] > 0
    assert features.rates_or_values["f_07_second_person_pronouns"] > 0
    assert features.counts["f_07_second_person_pronouns"] == 1


def test_pybiber_full_cli_writes_wide_and_long_outputs(tmp_path):
    input_path = tmp_path / "records.jsonl"
    input_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_id": "a",
                        "generation": "I think you should test this because it is useful.",
                    }
                ),
                json.dumps(
                    {
                        "record_id": "b",
                        "generation": "The result was produced by the system.",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "pybiber.csv"
    long_path = tmp_path / "pybiber_long.csv"

    args = build_parser().parse_args(
        [
            "--input",
            str(input_path),
            "--text-field",
            "generation",
            "--id-field",
            "record_id",
            "--output",
            str(output_path),
            "--long-output",
            str(long_path),
        ]
    )

    rows = run(args)

    assert len(rows) == 2
    with output_path.open(encoding="utf-8", newline="") as handle:
        wide_rows = list(csv.DictReader(handle))
    assert [row["doc_id"] for row in wide_rows] == ["a", "b"]
    assert "f_18_by_passives" in wide_rows[0]
    with long_path.open(encoding="utf-8", newline="") as handle:
        long_rows = list(csv.DictReader(handle))
    assert len(long_rows) == 2 * len(PYBIBER_FEATURES)


def test_pybiber_full_row_qualifies_duplicate_doc_ids():
    records = _ensure_unique_doc_ids(
        [
            {"doc_id": "dup", "text": "First duplicate."},
            {"doc_id": "unique", "text": "Unique document."},
            {"doc_id": "dup", "text": "Second duplicate."},
        ]
    )

    assert [record["doc_id"] for record in records] == ["dup#row0", "unique", "dup#row2"]
