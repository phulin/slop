from __future__ import annotations

import csv
from pathlib import Path

from slop_sftdiv.cli.audit_paper_review_hygiene import (
    audit_review_hygiene,
    build_parser,
    run_audit_paper_review_hygiene,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _clean_package(root: Path) -> None:
    manuscript = (
        "# Paper\n\n"
        "## Abstract\n\n"
        "This paper measures style.\n\n"
        "## Reproducibility Appendices\n\n"
        "Appendix material may describe evidence inventories.\n"
    )
    _write(root / "docs/experiments/paper_manuscript_draft.md", manuscript)
    _write(
        root / "artifacts/paper/submission/draft_bundle/manuscript/paper_manuscript_draft.md",
        manuscript,
    )
    _write(
        root / "artifacts/paper/submission/draft_bundle/manuscript/paper_caption_appendix_draft.md",
        "Captions.\n",
    )
    _write(
        root / "artifacts/paper/submission/draft_bundle/tables/paper_camera_ready_tables.md",
        "Tables.\n",
    )
    _write(
        root / "artifacts/paper/submission/draft_bundle/tables/paper_latex_table_drafts.tex",
        "\\caption{Table.}\n",
    )
    _write(
        root / "artifacts/paper/submission/draft_bundle/references/paper_references.bib",
        "@article{key,title={Title}}\n",
    )


def test_review_hygiene_passes_clean_paper_facing_files(tmp_path: Path) -> None:
    _clean_package(tmp_path)

    rows = audit_review_hygiene(tmp_path)

    assert {row["status"] for row in rows} == {"ready"}


def test_review_hygiene_flags_main_body_local_paths(tmp_path: Path) -> None:
    _clean_package(tmp_path)
    manuscript = (tmp_path / "docs/experiments/paper_manuscript_draft.md").read_text(
        encoding="utf-8"
    )
    _write(
        tmp_path / "docs/experiments/paper_manuscript_draft.md",
        manuscript.replace("This paper measures style.", "See docs/experiments/paper.md."),
    )

    rows = audit_review_hygiene(tmp_path)
    by_id = {row["check_id"]: row for row in rows}

    assert by_id["manuscript_main_body_hygiene"]["status"] == "review"
    assert "docs/" in by_id["manuscript_main_body_hygiene"]["forbidden_hits"]


def test_review_hygiene_cli_writes_markdown_and_csv(tmp_path: Path) -> None:
    _clean_package(tmp_path)
    args = build_parser().parse_args(["--root", str(tmp_path)])

    rows = run_audit_paper_review_hygiene(args)

    assert len(rows) == 4
    with (
        tmp_path / "artifacts/paper/manuscript/paper_review_hygiene_audit.csv"
    ).open(encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert [row["check_id"] for row in persisted] == [row["check_id"] for row in rows]
    markdown = (
        tmp_path / "docs/experiments/paper_review_hygiene_audit.md"
    ).read_text(encoding="utf-8")
    assert "Readiness status: `ready`." in markdown
    assert "| paper_facing_bundle_hygiene | ready |" in markdown
