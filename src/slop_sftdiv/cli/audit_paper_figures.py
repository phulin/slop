from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_FIGURE_DIR = Path("artifacts/paper/figures")
DEFAULT_OUTPUT_CSV = DEFAULT_FIGURE_DIR / "paper_figure_readiness_audit.csv"
DEFAULT_OUTPUT_MD = Path("docs/experiments/paper_figure_readiness_audit.md")

FORBIDDEN_SVG_PHRASES = ("docs/", "artifacts/", "src/", "uv run")


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    path: Path
    title_fragment: str
    min_width_pt: float = 450.0
    min_height_pt: float = 300.0
    min_text_elements: int = 8


FIGURE_SPECS = (
    FigureSpec(
        "Figure 1",
        DEFAULT_FIGURE_DIR / "figure1_pybiber_register_selected.svg",
        "Phase 1 register shift",
    ),
    FigureSpec(
        "Figure 2",
        DEFAULT_FIGURE_DIR / "figure2_eqbench_stage_scores.svg",
        "EQ-Bench Slop Score by checkpoint",
    ),
    FigureSpec(
        "Figure 3",
        DEFAULT_FIGURE_DIR / "figure3_olmo_slop_lexicon_views.svg",
        "OLMo slop_lexicon amplification views",
    ),
    FigureSpec(
        "Figure 4",
        DEFAULT_FIGURE_DIR / "figure4_cross_ladder_af_scatter.svg",
        "Cross-ladder AF comparison",
    ),
    FigureSpec(
        "Figure 5",
        DEFAULT_FIGURE_DIR / "figure5_phase4_tier3_raw_af.svg",
        "Phase 4 Tier-3 raw AF by checkpoint",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit rendered paper SVG figure readiness.")
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def run_audit_paper_figures(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = [
        FigureSpec(
            spec.figure_id,
            args.figure_dir / spec.path.name,
            spec.title_fragment,
            spec.min_width_pt,
            spec.min_height_pt,
            spec.min_text_elements,
        )
        for spec in FIGURE_SPECS
    ]
    rows = [_audit_one_figure(spec) for spec in specs]
    _write_csv(args.output_csv, rows)
    _write_markdown(args.output_md, rows, output_csv=args.output_csv)
    return rows


def _audit_one_figure(spec: FigureSpec) -> dict[str, Any]:
    warnings: list[str] = []
    exists = spec.path.exists()
    width_pt = height_pt = 0.0
    has_viewbox = False
    text_count = 0
    title_present = False
    forbidden_hits: list[str] = []
    metadata_date = False
    if not exists:
        warnings.append("missing_svg")
    else:
        raw = spec.path.read_text(encoding="utf-8")
        forbidden_hits = [phrase for phrase in FORBIDDEN_SVG_PHRASES if phrase in raw]
        if forbidden_hits:
            warnings.append("forbidden_svg_phrase")
        metadata_date = "2026-06-18" in raw
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            warnings.append("invalid_xml")
        else:
            width_pt = _parse_pt(root.attrib.get("width", ""))
            height_pt = _parse_pt(root.attrib.get("height", ""))
            has_viewbox = bool(root.attrib.get("viewBox"))
            texts = [
                _normalize_text("".join(element.itertext()))
                for element in root.iter()
                if element.tag.endswith("text")
            ]
            texts = [text for text in texts if text]
            text_count = len(texts)
            joined_text = " ".join(texts)
            title_present = spec.title_fragment in joined_text
    pdf_path = spec.path.parent / "submission_exports" / f"{spec.path.stem}.pdf"
    png_path = spec.path.parent / "submission_exports" / f"{spec.path.stem}.png"
    pdf_export_present = pdf_path.exists()
    png_export_present = png_path.exists()
    pdf_export_valid = _has_binary_prefix(pdf_path, b"%PDF")
    png_export_valid = _has_binary_prefix(png_path, b"\x89PNG\r\n\x1a\n")
    if not pdf_export_present:
        warnings.append("missing_pdf_export")
    elif not pdf_export_valid:
        warnings.append("invalid_pdf_export")
    if not png_export_present:
        warnings.append("missing_png_export")
    elif not png_export_valid:
        warnings.append("invalid_png_export")
    if width_pt < spec.min_width_pt:
        warnings.append("narrow_svg")
    if height_pt < spec.min_height_pt:
        warnings.append("short_svg")
    if not has_viewbox:
        warnings.append("missing_viewbox")
    if text_count < spec.min_text_elements:
        warnings.append("low_editable_text_count")
    if not title_present:
        warnings.append("missing_expected_title")
    if not metadata_date:
        warnings.append("missing_deterministic_date")
    status = "ready_for_visual_review" if not warnings else "review"
    return {
        "figure_id": spec.figure_id,
        "path": str(spec.path),
        "status": status,
        "width_pt": f"{width_pt:.3f}",
        "height_pt": f"{height_pt:.3f}",
        "has_viewbox": str(has_viewbox),
        "editable_text_elements": str(text_count),
        "title_fragment": spec.title_fragment,
        "title_present": str(title_present),
        "forbidden_svg_phrases": ";".join(forbidden_hits),
        "pdf_export_path": str(pdf_path),
        "pdf_export_present": str(pdf_export_present),
        "pdf_export_valid": str(pdf_export_valid),
        "png_export_path": str(png_path),
        "png_export_present": str(png_export_present),
        "png_export_valid": str(png_export_valid),
        "warnings": ";".join(warnings),
    }


def _parse_pt(value: str) -> float:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)pt", value.strip())
    return float(match.group(1)) if match else 0.0


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _has_binary_prefix(path: Path, prefix: bytes) -> bool:
    if not path.exists():
        return False
    with path.open("rb") as handle:
        return handle.read(len(prefix)) == prefix


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "figure_id",
        "path",
        "status",
        "width_pt",
        "height_pt",
        "has_viewbox",
        "editable_text_elements",
        "title_fragment",
        "title_present",
        "forbidden_svg_phrases",
        "pdf_export_path",
        "pdf_export_present",
        "pdf_export_valid",
        "png_export_path",
        "png_export_present",
        "png_export_valid",
        "warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_ready = all(row["status"] == "ready_for_visual_review" for row in rows)
    readiness = "ready_for_visual_review" if all_ready else "review"
    lines = [
        "# Paper Figure Readiness Audit",
        "",
        f"Machine-readable audit: `{output_csv}`",
        "",
        f"Readiness status: `{readiness}`.",
        "This audit checks SVG package hygiene and complements the rendered-layout review; "
        "remaining figure work is final venue sizing/export.",
        "",
        "| Figure | Status | Size pt | Text elements | Export candidates | Expected title | Warnings |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        size = f"{row['width_pt']} x {row['height_pt']}"
        warning_text = row["warnings"] or "none"
        exports = "PDF+PNG" if row["pdf_export_valid"] == "True" and row["png_export_valid"] == "True" else "review"
        lines.append(
            "| {figure_id} | {status} | {size} | {editable_text_elements} | "
            "{exports} | {title_fragment} | {warning_text} |".format(
                **row,
                size=size,
                exports=exports,
                warning_text=warning_text,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_audit_paper_figures(args)
    print(f"Wrote {len(rows)} paper figure-readiness rows to {args.output_csv}")
    print(f"Wrote paper figure-readiness summary to {args.output_md}")


if __name__ == "__main__":
    main()
