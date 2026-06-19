from __future__ import annotations

import csv
from pathlib import Path

from slop_sftdiv.cli.audit_paper_claim_evidence import (
    audit_claim_evidence_map,
    build_parser,
    run_audit_paper_claim_evidence,
)


def _claim_row(claim_id: str, evidence: str = "`docs/evidence.md`") -> str:
    return (
        f"| {claim_id} | Role | Section | Figure 1 | {evidence} | Caveat present. |"
    )


def _claim_map(rows: list[str]) -> str:
    return "\n".join(
        [
            "# Claim Map",
            "",
            "| Claim ID | Claim role | Manuscript location | Figure/table support | Primary evidence artifacts | Residual caveat |",
            "|---|---|---|---|---|---|",
            *rows,
        ]
    )


def test_audit_claim_evidence_map_marks_existing_paths_ready(tmp_path: Path) -> None:
    evidence = tmp_path / "docs/evidence.md"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("evidence\n", encoding="utf-8")

    rows = audit_claim_evidence_map(
        _claim_map([_claim_row("C1"), _claim_row("C2", "Same evidence as C1")]),
        root=tmp_path,
        expected_claim_ids=("C1", "C2"),
    )

    assert [row["status"] for row in rows] == ["ready", "ready"]
    assert rows[1]["referenced_claims"] == "C1"


def test_audit_claim_evidence_map_flags_missing_paths(tmp_path: Path) -> None:
    rows = audit_claim_evidence_map(
        _claim_map([_claim_row("C1", "`docs/missing.md`")]),
        root=tmp_path,
        expected_claim_ids=("C1",),
    )

    assert rows[0]["status"] == "review"
    assert rows[0]["missing_paths"] == "docs/missing.md"


def test_audit_claim_evidence_cli_writes_outputs(tmp_path: Path) -> None:
    evidence = tmp_path / "docs/evidence.md"
    claim_map = tmp_path / "docs/claim_map.md"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("evidence\n", encoding="utf-8")
    rows = [_claim_row(f"C{index}") for index in range(1, 15)]
    claim_map.write_text(_claim_map(rows), encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--claim-map",
            str(claim_map),
        ]
    )

    output_rows = run_audit_paper_claim_evidence(args)

    assert len(output_rows) == 14
    assert {row["status"] for row in output_rows} == {"ready"}
    output_csv = tmp_path / "artifacts/paper/claims/paper_claim_evidence_audit.csv"
    output_md = tmp_path / "docs/experiments/paper_claim_evidence_audit.md"
    with output_csv.open(encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert persisted[0]["claim_id"] == "C1"
    assert "Readiness status: `ready`." in output_md.read_text(encoding="utf-8")
