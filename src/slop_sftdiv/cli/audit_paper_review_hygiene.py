from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path.cwd()
DEFAULT_OUTPUT_CSV = Path("artifacts/paper/manuscript/paper_review_hygiene_audit.csv")
DEFAULT_OUTPUT_MD = Path("docs/experiments/paper_review_hygiene_audit.md")

SOURCE_MANUSCRIPT = Path("docs/experiments/paper_manuscript_draft.md")
BUNDLE_MANUSCRIPT = Path(
    "artifacts/paper/submission/draft_bundle/manuscript/paper_manuscript_draft.md"
)
PAPER_FACING_BUNDLE_PATHS = (
    Path("artifacts/paper/submission/draft_bundle/manuscript/paper_manuscript_draft.md"),
    Path("artifacts/paper/submission/draft_bundle/tables/paper_camera_ready_tables.md"),
    Path("artifacts/paper/submission/draft_bundle/tables/paper_latex_table_drafts.tex"),
    Path("artifacts/paper/submission/draft_bundle/references/paper_references.bib"),
)

MAIN_BODY_FORBIDDEN = (
    "/home/",
    "docs/",
    "artifacts/",
    "src/",
    "uv run",
    "GitHub",
    "Codex",
    "OpenAI",
    "TODO",
    "TBD",
    "placeholder",
)

PAPER_FACING_BUNDLE_FORBIDDEN = (
    "/home/",
    "docs/",
    "artifacts/",
    "src/",
    "uv run",
    "GitHub",
    "Codex",
    "OpenAI",
    "TODO",
    "TBD",
    "placeholder",
)

AUTHOR_PLACEHOLDERS = (
    "Author Name",
    "Anonymous Author",
    "Affiliation",
    "email@example.com",
    "corresponding author",
)


@dataclass(frozen=True)
class HygieneCheck:
    check_id: str
    scope: str
    forbidden_hits: tuple[str, ...]
    missing_required: tuple[str, ...]
    purpose: str

    @property
    def status(self) -> str:
        return "ready" if not self.forbidden_hits and not self.missing_required else "review"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit paper-facing review/anonymity hygiene.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def run_audit_paper_review_hygiene(args: argparse.Namespace) -> list[dict[str, str]]:
    root = args.root.resolve()
    rows = audit_review_hygiene(root)
    output_csv = _resolve_path(root, args.output_csv)
    output_md = _resolve_path(root, args.output_md)
    _write_csv(output_csv, rows)
    _write_markdown(output_md, rows, output_csv=args.output_csv)
    return rows


def audit_review_hygiene(root: Path) -> list[dict[str, str]]:
    checks = [
        _main_body_check(root),
        _author_placeholder_check(root),
        _bundle_match_check(root),
        _paper_facing_bundle_check(root),
    ]
    return [_row(check) for check in checks]


def _main_body_check(root: Path) -> HygieneCheck:
    path = root / SOURCE_MANUSCRIPT
    if not path.exists():
        return HygieneCheck(
            "manuscript_main_body_hygiene",
            str(SOURCE_MANUSCRIPT),
            (),
            (str(SOURCE_MANUSCRIPT),),
            "Keeps local paths, commands, and drafting markers out of the paper main body.",
        )
    text = _main_body(path.read_text(encoding="utf-8"))
    hits = _hits(text, MAIN_BODY_FORBIDDEN)
    return HygieneCheck(
        "manuscript_main_body_hygiene",
        str(SOURCE_MANUSCRIPT),
        hits,
        (),
        "Keeps local paths, commands, and drafting markers out of the paper main body.",
    )


def _author_placeholder_check(root: Path) -> HygieneCheck:
    path = root / SOURCE_MANUSCRIPT
    if not path.exists():
        return HygieneCheck(
            "author_placeholder_hygiene",
            str(SOURCE_MANUSCRIPT),
            (),
            (str(SOURCE_MANUSCRIPT),),
            "Prevents author, affiliation, and email placeholders from entering the draft.",
        )
    text = path.read_text(encoding="utf-8")
    hits = _hits(text, AUTHOR_PLACEHOLDERS)
    return HygieneCheck(
        "author_placeholder_hygiene",
        str(SOURCE_MANUSCRIPT),
        hits,
        (),
        "Prevents author, affiliation, and email placeholders from entering the draft.",
    )


def _bundle_match_check(root: Path) -> HygieneCheck:
    source_path = root / SOURCE_MANUSCRIPT
    bundle_path = root / BUNDLE_MANUSCRIPT
    missing = tuple(
        str(path)
        for path, exists in (
            (SOURCE_MANUSCRIPT, source_path.exists()),
            (BUNDLE_MANUSCRIPT, bundle_path.exists()),
        )
        if not exists
    )
    if missing:
        return HygieneCheck(
            "bundle_manuscript_matches_source",
            f"{SOURCE_MANUSCRIPT}; {BUNDLE_MANUSCRIPT}",
            (),
            missing,
            "Ensures the staged manuscript is the audited source manuscript.",
        )
    mismatch = source_path.read_text(encoding="utf-8") != bundle_path.read_text(encoding="utf-8")
    return HygieneCheck(
        "bundle_manuscript_matches_source",
        f"{SOURCE_MANUSCRIPT}; {BUNDLE_MANUSCRIPT}",
        ("bundle manuscript differs from source manuscript",) if mismatch else (),
        (),
        "Ensures the staged manuscript is the audited source manuscript.",
    )


def _paper_facing_bundle_check(root: Path) -> HygieneCheck:
    missing_paths = tuple(str(path) for path in PAPER_FACING_BUNDLE_PATHS if not (root / path).exists())
    hits: list[str] = []
    for relative in PAPER_FACING_BUNDLE_PATHS:
        path = root / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in _hits(text, PAPER_FACING_BUNDLE_FORBIDDEN):
            hits.append(f"{relative}:{marker}")
    return HygieneCheck(
        "paper_facing_bundle_hygiene",
        "draft_bundle manuscript/table/reference files",
        tuple(hits),
        missing_paths,
        "Keeps paper-facing staged files free of repository paths, commands, and draft placeholders.",
    )


def _main_body(text: str) -> str:
    marker = "\n## Reproducibility Appendices"
    return text.split(marker, 1)[0]


def _hits(text: str, patterns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(pattern for pattern in patterns if pattern in text)


def _row(check: HygieneCheck) -> dict[str, str]:
    return {
        "check_id": check.check_id,
        "status": check.status,
        "scope": check.scope,
        "forbidden_hits": "; ".join(check.forbidden_hits),
        "missing_required": "; ".join(check.missing_required),
        "purpose": check.purpose,
    }


def _resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "check_id",
        "status",
        "scope",
        "forbidden_hits",
        "missing_required",
        "purpose",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, str]], *, output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readiness = "ready" if all(row["status"] == "ready" for row in rows) else "review"
    lines = [
        "# Paper Review Hygiene Audit",
        "",
        f"Machine-readable audit: `{output_csv}`",
        "",
        f"Readiness status: `{readiness}`.",
        (
            "This audit checks that the integrated manuscript and paper-facing "
            "submission-bundle files avoid local paths, internal tool commands, "
            "draft placeholders, and author placeholder text. Review-control "
            "files are intentionally out of scope because they preserve evidence paths."
        ),
        "",
        "| Check | Status | Scope | Forbidden hits | Missing required | Purpose |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['check_id']} | {row['status']} | {row['scope']} | "
            f"{row['forbidden_hits'] or 'none'} | {row['missing_required'] or 'none'} | "
            f"{row['purpose']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_audit_paper_review_hygiene(args)
    print(f"Wrote {len(rows)} review-hygiene audit rows to {args.output_csv}")
    print(f"Wrote review-hygiene audit summary to {args.output_md}")


if __name__ == "__main__":
    main()
