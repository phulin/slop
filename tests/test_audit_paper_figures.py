from __future__ import annotations

import csv

from slop_sftdiv.cli.audit_paper_figures import (
    FIGURE_SPECS,
    build_parser,
    run_audit_paper_figures,
)


def _svg(title: str, *, width: str = "500pt", height: str = "320pt") -> str:
    text_nodes = "\n".join(
        f'<text x="{index}" y="{index}">label {index}</text>' for index in range(8)
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<svg width="{width}" height="{height}" viewBox="0 0 500 320" '
        'xmlns="http://www.w3.org/2000/svg">\n'
        "<metadata>2026-06-18</metadata>\n"
        f"<text>{title}</text>\n"
        f"{text_nodes}\n"
        "</svg>\n"
    )


def test_audit_paper_figures_marks_clean_svg_set_ready(tmp_path):
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    export_dir = figure_dir / "submission_exports"
    export_dir.mkdir()
    for spec in FIGURE_SPECS:
        (figure_dir / spec.path.name).write_text(_svg(spec.title_fragment), encoding="utf-8")
        (export_dir / f"{spec.path.stem}.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
        (export_dir / f"{spec.path.stem}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    output_csv = tmp_path / "figures.csv"
    output_md = tmp_path / "figures.md"
    args = build_parser().parse_args(
        [
            "--figure-dir",
            str(figure_dir),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ]
    )

    rows = run_audit_paper_figures(args)

    assert len(rows) == 5
    assert {row["status"] for row in rows} == {"ready_for_visual_review"}
    assert {row["pdf_export_valid"] for row in rows} == {"True"}
    assert {row["png_export_valid"] for row in rows} == {"True"}
    assert "Readiness status: `ready_for_visual_review`." in output_md.read_text(
        encoding="utf-8"
    )
    with output_csv.open(encoding="utf-8", newline="") as handle:
        persisted = list(csv.DictReader(handle))
    assert persisted[0]["title_present"] == "True"


def test_audit_paper_figures_marks_missing_title_and_small_svg_for_review(tmp_path):
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    export_dir = figure_dir / "submission_exports"
    export_dir.mkdir()
    for spec in FIGURE_SPECS:
        title = spec.title_fragment
        width = "500pt"
        height = "320pt"
        if spec.figure_id == "Figure 2":
            title = "Wrong title"
            width = "200pt"
        (figure_dir / spec.path.name).write_text(
            _svg(title, width=width, height=height),
            encoding="utf-8",
        )
        (export_dir / f"{spec.path.stem}.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
        (export_dir / f"{spec.path.stem}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    args = build_parser().parse_args(
        [
            "--figure-dir",
            str(figure_dir),
            "--output-csv",
            str(tmp_path / "figures.csv"),
            "--output-md",
            str(tmp_path / "figures.md"),
        ]
    )

    rows = run_audit_paper_figures(args)

    figure2 = next(row for row in rows if row["figure_id"] == "Figure 2")
    assert figure2["status"] == "review"
    assert "narrow_svg" in figure2["warnings"]
    assert "missing_expected_title" in figure2["warnings"]
