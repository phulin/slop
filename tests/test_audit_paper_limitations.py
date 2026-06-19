from __future__ import annotations

import csv
from pathlib import Path

from slop_sftdiv.cli.audit_paper_limitations import (
    LIMITATION_SPECS,
    audit_limitations,
    build_parser,
    run_audit_paper_limitations,
)


def _clean_manuscript() -> str:
    limitations = "\n\n".join(
        " ".join(spec.required_phrases) for spec in LIMITATION_SPECS
    )
    return (
        "# Title\n\n"
        "## 5. Discussion\n\n"
        "Body.\n\n"
        "## 6. Limitations\n\n"
        f"{limitations}\n\n"
        "## 7. Conclusion\n\n"
        "Conclusion.\n"
    )


def _run(tmp_path: Path, manuscript: str):
    manuscript_path = tmp_path / "manuscript.md"
    output_csv = tmp_path / "limitations.csv"
    output_md = tmp_path / "limitations.md"
    manuscript_path.write_text(manuscript, encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--manuscript",
            str(manuscript_path),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ]
    )
    return run_audit_paper_limitations(args), output_csv, output_md


def test_audit_limitations_marks_complete_section_ready(tmp_path: Path) -> None:
    rows, output_csv, output_md = _run(tmp_path, _clean_manuscript())

    assert len(rows) == len(LIMITATION_SPECS) + 1
    assert {row["status"] for row in rows} == {"ready"}
    assert "Readiness status: `ready`." in output_md.read_text(encoding="utf-8")
    with output_csv.open(encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert persisted[0]["check_id"] == "bounded_open_ladders"
    assert persisted[-1]["check_id"] == "forbidden_overclaims"


def test_audit_limitations_root_resolves_relative_paths(tmp_path: Path) -> None:
    manuscript_path = tmp_path / "docs/experiments/paper_manuscript_draft.md"
    manuscript_path.parent.mkdir(parents=True)
    manuscript_path.write_text(_clean_manuscript(), encoding="utf-8")
    args = build_parser().parse_args(["--root", str(tmp_path)])

    rows = run_audit_paper_limitations(args)

    assert {row["status"] for row in rows} == {"ready"}
    assert (
        tmp_path / "artifacts/paper/manuscript/paper_limitations_audit.csv"
    ).exists()
    summary = (tmp_path / "docs/experiments/paper_limitations_audit.md").read_text(
        encoding="utf-8"
    )
    assert (
        "Machine-readable audit: "
        "`artifacts/paper/manuscript/paper_limitations_audit.csv`"
    ) in summary


def test_audit_limitations_flags_missing_required_phrase() -> None:
    manuscript = _clean_manuscript().replace("output-bootstrap intervals", "")

    rows = audit_limitations(manuscript)

    eqbench = next(row for row in rows if row["check_id"] == "eqbench_aggregate_boundary")
    assert eqbench["status"] == "review"
    assert "output-bootstrap intervals" in eqbench["missing_required"]


def test_audit_limitations_flags_forbidden_overclaim() -> None:
    manuscript = _clean_manuscript().replace(
        "human perceptibility labels are still absent",
        "human perceptibility labels are complete",
    )

    rows = audit_limitations(manuscript)

    forbidden = next(row for row in rows if row["check_id"] == "forbidden_overclaims")
    assert forbidden["status"] == "review"
    assert "human perceptibility labels are complete" in forbidden["forbidden_hits"]
