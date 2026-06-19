from __future__ import annotations

from pathlib import Path

from slop_sftdiv.cli.audit_paper_numeric_claims import NumericCheck, audit_numeric_claims


def test_audit_numeric_claims_marks_present_claim_ready() -> None:
    checks = [
        NumericCheck(
            "wrapped_hyphen_claim",
            source_path=Path("source.csv"),
            expected_text="chosen-minus-rejected direction is negative",
        )
    ]
    text = "The chosen-minus-\nrejected direction is negative in the manuscript."

    rows = audit_numeric_claims(text, checks)

    assert rows == [
        {
            "check_id": "wrapped_hyphen_claim",
            "status": "ready",
            "source_path": "source.csv",
            "expected_text": "chosen-minus-rejected direction is negative",
            "missing_text": "",
        }
    ]


def test_audit_numeric_claims_flags_missing_claim() -> None:
    checks = [
        NumericCheck(
            "missing_claim",
            source_path=Path("source.csv"),
            expected_text="score rises from base `1.000` to SFT `2.000`",
        )
    ]

    rows = audit_numeric_claims("score falls instead", checks)

    assert rows[0]["status"] == "review"
    assert rows[0]["missing_text"] == "score rises from base `1.000` to SFT `2.000`"
