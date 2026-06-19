from __future__ import annotations

import csv
from pathlib import Path

from slop_sftdiv.cli.audit_paper_citations import (
    audit_citations,
    build_parser,
    run_audit_paper_citations,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_audit_citations_marks_complete_reference_package_ready(tmp_path: Path) -> None:
    bib = tmp_path / "refs.bib"
    sources = tmp_path / "sources.md"
    manuscript = tmp_path / "manuscript.md"
    _write(bib, "@misc{known_key,\n  title={Known}\n}\n")
    _write(
        sources,
        "| Key | Source | Verified from | Notes for use |\n"
        "|---|---|---|---|\n"
        "| `known_key` | Source | https://example.com | Notes |\n",
    )
    _write(manuscript, "This cites [@known_key].\n")

    rows = audit_citations(
        root=tmp_path,
        bib_path=bib,
        source_path=sources,
        citation_texts=(manuscript,),
    )

    assert {row["status"] for row in rows} == {"ready"}


def test_audit_citations_flags_missing_bib_and_source_rows(tmp_path: Path) -> None:
    bib = tmp_path / "refs.bib"
    sources = tmp_path / "sources.md"
    manuscript = tmp_path / "manuscript.md"
    _write(bib, "@misc{known_key,\n  title={Known}\n}\n")
    _write(
        sources,
        "| Key | Source | Verified from | Notes for use |\n"
        "|---|---|---|---|\n",
    )
    _write(manuscript, "This cites [@known_key; @missing_key].\n")

    rows = {
        row["check_id"]: row
        for row in audit_citations(
            root=tmp_path,
            bib_path=bib,
            source_path=sources,
            citation_texts=(manuscript,),
        )
    }

    assert rows["cited_keys_have_bib_entries"]["status"] == "review"
    assert rows["cited_keys_have_bib_entries"]["missing_or_invalid"] == "missing_key"
    assert rows["bib_keys_have_source_rows"]["status"] == "review"
    assert rows["bib_keys_have_source_rows"]["missing_or_invalid"] == "known_key"


def test_audit_paper_citations_cli_writes_outputs(tmp_path: Path) -> None:
    _write(tmp_path / "refs.bib", "@misc{known_key,\n  title={Known}\n}\n")
    _write(
        tmp_path / "sources.md",
        "| Key | Source | Verified from | Notes for use |\n"
        "|---|---|---|---|\n"
        "| `known_key` | Source | https://example.com | Notes |\n",
    )
    _write(tmp_path / "manuscript.md", "This cites [@known_key].\n")
    args = build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--bib",
            "refs.bib",
            "--sources",
            "sources.md",
            "--citation-text",
            "manuscript.md",
        ]
    )

    rows = run_audit_paper_citations(args)

    assert {row["status"] for row in rows} == {"ready"}
    output_csv = tmp_path / "artifacts/paper/references/paper_citation_audit.csv"
    output_md = tmp_path / "docs/experiments/paper_citation_audit.md"
    with output_csv.open(encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert persisted[0]["check_id"] == "cited_keys_have_bib_entries"
    assert "Readiness status: `ready`." in output_md.read_text(encoding="utf-8")
