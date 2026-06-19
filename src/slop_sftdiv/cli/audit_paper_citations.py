from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path.cwd()
DEFAULT_BIB = Path("docs/experiments/paper_references.bib")
DEFAULT_SOURCES = Path("docs/experiments/paper_reference_sources.md")
DEFAULT_OUTPUT_CSV = Path("artifacts/paper/references/paper_citation_audit.csv")
DEFAULT_OUTPUT_MD = Path("docs/experiments/paper_citation_audit.md")
DEFAULT_CITATION_TEXTS = (
    Path("docs/experiments/paper_manuscript_draft.md"),
    Path("docs/experiments/paper_methods_draft.md"),
    Path("docs/experiments/paper_results_draft.md"),
    Path("docs/experiments/paper_intro_discussion_draft.md"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit paper citation/source integrity.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bib", type=Path, default=DEFAULT_BIB)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--citation-text",
        type=Path,
        action="append",
        default=None,
        help="Manuscript-facing text file to scan for citation keys.",
    )
    return parser


def run_audit_paper_citations(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = args.root.resolve()
    bib_path = _resolve_path(root, args.bib)
    source_path = _resolve_path(root, args.sources)
    citation_texts = tuple(args.citation_text or DEFAULT_CITATION_TEXTS)
    rows = audit_citations(
        root=root,
        bib_path=bib_path,
        source_path=source_path,
        citation_texts=citation_texts,
    )
    output_csv = _resolve_path(root, args.output_csv)
    output_md = _resolve_path(root, args.output_md)
    _write_csv(output_csv, rows)
    _write_markdown(output_md, rows, output_csv=args.output_csv)
    return rows


def audit_citations(
    *,
    root: Path,
    bib_path: Path,
    source_path: Path,
    citation_texts: tuple[Path, ...],
) -> list[dict[str, Any]]:
    bib_text = bib_path.read_text(encoding="utf-8") if bib_path.exists() else ""
    source_text = source_path.read_text(encoding="utf-8") if source_path.exists() else ""
    bib_keys = _bib_keys(bib_text)
    source_rows = _source_rows(source_text)
    source_keys = set(source_rows)
    cited_by_file = _cited_by_file(root, citation_texts)
    cited_keys = set().union(*cited_by_file.values()) if cited_by_file else set()
    cited_missing_bib = sorted(cited_keys - bib_keys)
    bib_missing_source = sorted(bib_keys - source_keys)
    source_missing_bib = sorted(source_keys - bib_keys)
    source_rows_missing_fields = sorted(
        key
        for key, row in source_rows.items()
        if not row["source"] or not row["verified_from"] or not row["notes"]
    )
    source_rows_without_url = sorted(
        key for key, row in source_rows.items() if "http" not in row["verified_from"]
    )
    uncited_bib_keys = sorted(bib_keys - cited_keys)

    rows = [
        _row(
            "cited_keys_have_bib_entries",
            cited_missing_bib,
            observed=f"{len(cited_keys)} cited keys across {len(cited_by_file)} text files",
            notes=_citation_file_summary(cited_by_file),
        ),
        _row(
            "bib_keys_have_source_rows",
            bib_missing_source,
            observed=f"{len(bib_keys)} BibTeX keys; {len(source_keys)} source rows",
        ),
        _row(
            "source_rows_have_bib_entries",
            source_missing_bib,
            observed=f"{len(source_keys)} source rows; {len(bib_keys)} BibTeX keys",
        ),
        _row(
            "source_rows_are_populated",
            source_rows_missing_fields,
            observed=f"{len(source_rows)} source rows",
            notes="Source, verified-from, and notes cells must be populated.",
        ),
        _row(
            "source_rows_have_urls",
            source_rows_without_url,
            observed=f"{len(source_rows)} source rows",
            notes="Verified-from cells should contain URLs for paper provenance review.",
        ),
        _row(
            "uncited_bib_keys_recorded",
            [],
            observed=f"{len(uncited_bib_keys)} uncited BibTeX keys",
            notes="Not a blocker; uncited keys: " + (", ".join(uncited_bib_keys) or "none"),
        ),
    ]
    return rows


def _row(
    check_id: str,
    missing: list[str],
    *,
    observed: str,
    notes: str = "",
) -> dict[str, str]:
    return {
        "check_id": check_id,
        "status": "ready" if not missing else "review",
        "observed": observed,
        "missing_or_invalid": "; ".join(missing),
        "notes": notes,
    }


def _bib_keys(text: str) -> set[str]:
    return set(re.findall(r"@\w+\{([^,\s]+)", text))


def _source_rows(text: str) -> dict[str, dict[str, str]]:
    rows = {}
    for line in text.splitlines():
        if not re.match(r"^\|\s*`[^`]+`\s*\|", line):
            continue
        cells = _split_markdown_row(line)
        if len(cells) < 4:
            continue
        key = cells[0].strip("`")
        rows[key] = {
            "source": cells[1],
            "verified_from": cells[2],
            "notes": cells[3],
        }
    return rows


def _cited_by_file(root: Path, paths: tuple[Path, ...]) -> dict[str, set[str]]:
    cited = {}
    for path in paths:
        resolved = _resolve_path(root, path)
        if not resolved.exists():
            continue
        relative = path.as_posix()
        cited[relative] = set(
            re.findall(r"@([A-Za-z0-9_:-]+)", resolved.read_text(encoding="utf-8"))
        )
    return cited


def _citation_file_summary(cited_by_file: dict[str, set[str]]) -> str:
    if not cited_by_file:
        return "No citation text files found."
    return "; ".join(f"{path}: {len(keys)} keys" for path, keys in sorted(cited_by_file.items()))


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["check_id", "status", "observed", "missing_or_invalid", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readiness = "ready" if all(row["status"] == "ready" for row in rows) else "review"
    lines = [
        "# Paper Citation Audit",
        "",
        f"Machine-readable audit: `{output_csv}`",
        "",
        f"Readiness status: `{readiness}`.",
        (
            "This audit checks cited manuscript keys, BibTeX entries, and "
            "verified source-inventory rows for the paper reference package."
        ),
        "",
        "| Check | Status | Observed | Missing or invalid | Notes |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        missing = row["missing_or_invalid"] or "none"
        notes = row["notes"] or "none"
        lines.append(
            f"| {row['check_id']} | {row['status']} | {row['observed']} | {missing} | {notes} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_audit_paper_citations(args)
    print(f"Wrote {len(rows)} citation audit rows to {args.output_csv}")
    print(f"Wrote citation audit summary to {args.output_md}")


if __name__ == "__main__":
    main()
