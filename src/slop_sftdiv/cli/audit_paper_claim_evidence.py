from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path.cwd()
DEFAULT_CLAIM_MAP = Path("docs/experiments/paper_claim_evidence_map.md")
DEFAULT_OUTPUT_CSV = Path("artifacts/paper/claims/paper_claim_evidence_audit.csv")
DEFAULT_OUTPUT_MD = Path("docs/experiments/paper_claim_evidence_audit.md")
EXPECTED_CLAIM_IDS = tuple(f"C{index}" for index in range(1, 15))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit paper claim evidence-map integrity.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--claim-map", type=Path, default=DEFAULT_CLAIM_MAP)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def run_audit_paper_claim_evidence(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = args.root.resolve()
    claim_map_path = _resolve_path(root, args.claim_map)
    text = claim_map_path.read_text(encoding="utf-8") if claim_map_path.exists() else ""
    rows = audit_claim_evidence_map(text, root=root)
    output_csv = _resolve_path(root, args.output_csv)
    output_md = _resolve_path(root, args.output_md)
    _write_csv(output_csv, rows)
    _write_markdown(output_md, rows, output_csv=args.output_csv)
    return rows


def audit_claim_evidence_map(
    text: str,
    *,
    root: Path,
    expected_claim_ids: tuple[str, ...] = EXPECTED_CLAIM_IDS,
) -> list[dict[str, Any]]:
    claim_rows = _parse_claim_rows(text)
    by_claim = {row["claim_id"]: row for row in claim_rows}
    rows: list[dict[str, Any]] = []
    for claim_id in expected_claim_ids:
        row = by_claim.get(claim_id)
        if row is None:
            rows.append(
                {
                    "claim_id": claim_id,
                    "status": "review",
                    "existing_paths": "",
                    "missing_paths": "",
                    "referenced_claims": "",
                    "missing_fields": "claim row",
                    "evidence_cell": "",
                    "residual_caveat": "",
                }
            )
            continue

        paths = _extract_backtick_paths(row["evidence_cell"])
        existing_paths = [path for path in paths if (root / path).exists()]
        missing_paths = [path for path in paths if not (root / path).exists()]
        referenced_claims = sorted(
            claim for claim in set(re.findall(r"\bC\d+\b", row["evidence_cell"])) if claim != claim_id
        )
        missing_fields = [
            field
            for field in (
                "claim_role",
                "manuscript_location",
                "figure_table_support",
                "evidence_cell",
                "residual_caveat",
            )
            if not row[field].strip()
        ]
        has_evidence = bool(existing_paths or referenced_claims)
        status = "ready" if has_evidence and not missing_paths and not missing_fields else "review"
        rows.append(
            {
                "claim_id": claim_id,
                "status": status,
                "existing_paths": "; ".join(existing_paths),
                "missing_paths": "; ".join(missing_paths),
                "referenced_claims": "; ".join(referenced_claims),
                "missing_fields": "; ".join(missing_fields),
                "evidence_cell": row["evidence_cell"],
                "residual_caveat": row["residual_caveat"],
            }
        )
    return rows


def _parse_claim_rows(text: str) -> list[dict[str, str]]:
    rows = []
    for line in text.splitlines():
        if not re.match(r"^\|\s*C\d+\s*\|", line):
            continue
        cells = _split_markdown_row(line)
        if len(cells) < 6:
            continue
        rows.append(
            {
                "claim_id": cells[0],
                "claim_role": cells[1],
                "manuscript_location": cells[2],
                "figure_table_support": cells[3],
                "evidence_cell": cells[4],
                "residual_caveat": cells[5],
            }
        )
    return rows


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _extract_backtick_paths(text: str) -> list[str]:
    paths = []
    for match in re.finditer(r"`([^`]+)`", text):
        candidate = match.group(1).strip()
        if candidate.startswith(("docs/", "artifacts/", "src/", "EXPERIMENTS.md")):
            paths.append(candidate)
    return sorted(set(paths))


def _resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "claim_id",
        "status",
        "existing_paths",
        "missing_paths",
        "referenced_claims",
        "missing_fields",
        "evidence_cell",
        "residual_caveat",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], *, output_csv: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    readiness = "ready" if all(row["status"] == "ready" for row in rows) else "review"
    lines = [
        "# Paper Claim Evidence Audit",
        "",
        f"Machine-readable audit: `{output_csv}`",
        "",
        f"Readiness status: `{readiness}`.",
        (
            "This audit checks that the claim evidence map covers C1-C14, "
            "that mapped source paths exist, and that claim-to-claim evidence "
            "inheritance is explicit."
        ),
        "",
        "| Claim | Status | Existing paths | Referenced claims | Missing paths | Missing fields |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        existing = row["existing_paths"] or "none"
        referenced = row["referenced_claims"] or "none"
        missing_paths = row["missing_paths"] or "none"
        missing_fields = row["missing_fields"] or "none"
        lines.append(
            f"| {row['claim_id']} | {row['status']} | {existing} | {referenced} | "
            f"{missing_paths} | {missing_fields} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_audit_paper_claim_evidence(args)
    print(f"Wrote {len(rows)} claim-evidence audit rows to {args.output_csv}")
    print(f"Wrote claim-evidence audit summary to {args.output_md}")


if __name__ == "__main__":
    main()
