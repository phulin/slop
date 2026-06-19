from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from slop_sftdiv.cli.audit_paper_reproducibility_manifest import (
    ManifestItem,
    build_manifest_rows,
    build_parser,
    run_audit_paper_reproducibility_manifest,
)


def test_manifest_rows_include_size_and_sha(tmp_path: Path) -> None:
    document = tmp_path / "docs/experiments/paper.md"
    document.parent.mkdir(parents=True)
    document.write_text("paper\n", encoding="utf-8")

    rows = build_manifest_rows(
        tmp_path,
        (ManifestItem("document", Path("docs/experiments/paper.md")),),
    )

    assert rows == [
        {
            "category": "document",
            "path": "docs/experiments/paper.md",
            "status": "ready",
            "size_bytes": "6",
            "sha256": hashlib.sha256(b"paper\n").hexdigest(),
        }
    ]


def test_manifest_cli_writes_ready_markdown_and_csv(tmp_path: Path) -> None:
    document = tmp_path / "docs/experiments/paper.md"
    artifact = tmp_path / "artifacts/paper/figures/figure.svg"
    document.parent.mkdir(parents=True)
    artifact.parent.mkdir(parents=True)
    document.write_text("paper\n", encoding="utf-8")
    artifact.write_text("<svg></svg>\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            "--root",
            str(tmp_path),
            "--path",
            "docs/experiments/paper.md",
            "--path",
            "artifacts/paper/figures/figure.svg",
        ]
    )

    rows = run_audit_paper_reproducibility_manifest(args)

    assert {row["status"] for row in rows} == {"ready"}
    output_csv = tmp_path / "artifacts/paper/submission/paper_reproducibility_manifest.csv"
    output_md = tmp_path / "docs/experiments/paper_reproducibility_manifest.md"
    with output_csv.open(encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert [row["path"] for row in persisted] == [
        "docs/experiments/paper.md",
        "artifacts/paper/figures/figure.svg",
    ]
    markdown = output_md.read_text(encoding="utf-8")
    assert "Readiness status: `ready`." in markdown
    assert "`docs/experiments/paper.md`" in markdown


def test_manifest_cli_marks_missing_paths_review(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["--root", str(tmp_path), "--path", "docs/experiments/missing.md"]
    )

    rows = run_audit_paper_reproducibility_manifest(args)

    assert rows[0]["status"] == "missing"
    assert rows[0]["sha256"] == ""
    markdown = (
        tmp_path / "docs/experiments/paper_reproducibility_manifest.md"
    ).read_text(encoding="utf-8")
    assert "Readiness status: `review`." in markdown
