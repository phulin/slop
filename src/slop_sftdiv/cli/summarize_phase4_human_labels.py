from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ALLOWED_PERCEPTION_LABELS = {"", "yes", "no", "context_dependent", "unclear"}
ALLOWED_TAXONOMY_LABELS = {
    "",
    "tone",
    "structure",
    "coherence",
    "relevance",
    "density",
    "factuality",
    "bias",
    "none",
    "domain_artifact",
    "unclear",
}
PERCEIVED_VALUES = {"yes", "context_dependent"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize Phase 4 human-perceptibility annotation labels."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_annotation_package.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/phase4/modernbert_detector_combined_v2_clean/"
            "phase4_human_perceptibility_summary.csv"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("docs/experiments/phase4_human_perceptibility_summary.md"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on unknown labels instead of recording validation warnings.",
    )
    return parser


def run_summarize_phase4_human_labels(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows, warnings = _load_rows(args.input, strict=args.strict)
    summary_rows = _summarize(rows)
    _write_csv(args.output, summary_rows)
    _write_markdown(args.summary_output, summary_rows, warnings)
    return summary_rows


def _load_rows(path: Path, *, strict: bool) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    warnings = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            perception = str(row.get("human_perceived_slop", "")).strip().lower()
            taxonomy = str(row.get("shaib_taxonomy_label", "")).strip().lower()
            if perception not in ALLOWED_PERCEPTION_LABELS:
                message = f"{path}:{line_number}: unknown human_perceived_slop={perception!r}"
                if strict:
                    raise ValueError(message)
                warnings.append(message)
            if taxonomy not in ALLOWED_TAXONOMY_LABELS:
                message = f"{path}:{line_number}: unknown shaib_taxonomy_label={taxonomy!r}"
                if strict:
                    raise ValueError(message)
                warnings.append(message)
            row["human_perceived_slop"] = perception
            row["shaib_taxonomy_label"] = taxonomy
            rows.append(row)
    return rows, warnings


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_feature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_feature[str(row.get("feature", ""))].append(row)

    summary_rows = []
    for feature, feature_rows in sorted(by_feature.items()):
        perception_counts = Counter(row.get("human_perceived_slop", "") for row in feature_rows)
        taxonomy_counts = Counter(row.get("shaib_taxonomy_label", "") for row in feature_rows)
        labeled = sum(perception_counts[label] for label in ALLOWED_PERCEPTION_LABELS if label)
        perceived = sum(perception_counts[label] for label in PERCEIVED_VALUES)
        yes = perception_counts["yes"]
        context = perception_counts["context_dependent"]
        no = perception_counts["no"]
        unclear = perception_counts["unclear"]
        blank = perception_counts[""]
        total = len(feature_rows)
        candidate_label = str(feature_rows[0].get("candidate_label", ""))
        cluster = str(feature_rows[0].get("cluster", ""))
        region = _venn_region(total=total, labeled=labeled, yes=yes, context=context, no=no)
        top_taxonomy = _top_taxonomy(taxonomy_counts)
        summary_rows.append(
            {
                "feature": feature,
                "candidate_label": candidate_label,
                "cluster": cluster,
                "total": total,
                "labeled": labeled,
                "blank": blank,
                "yes": yes,
                "context_dependent": context,
                "no": no,
                "unclear": unclear,
                "human_perceived_or_context_count": perceived,
                "human_perceived_or_context_rate": perceived / labeled if labeled else "",
                "yes_rate": yes / labeled if labeled else "",
                "top_shaib_taxonomy_label": top_taxonomy,
                "venn_region": region,
            }
        )
    return summary_rows


def _venn_region(*, total: int, labeled: int, yes: int, context: int, no: int) -> str:
    if labeled == 0:
        return "unlabeled"
    perceived_rate = (yes + context) / labeled
    yes_rate = yes / labeled
    no_rate = no / labeled
    if yes_rate >= 0.5 or perceived_rate >= 0.65:
        return "detector_and_human_perceived"
    if no_rate >= 0.5 and labeled >= min(20, total):
        return "detector_only_or_domain_artifact"
    return "mixed_or_unresolved"


def _top_taxonomy(counts: Counter[str]) -> str:
    nonblank = {key: value for key, value in counts.items() if key}
    if not nonblank:
        return ""
    return max(nonblank.items(), key=lambda item: (item[1], item[0]))[0]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "feature",
        "candidate_label",
        "cluster",
        "total",
        "labeled",
        "blank",
        "yes",
        "context_dependent",
        "no",
        "unclear",
        "human_perceived_or_context_count",
        "human_perceived_or_context_rate",
        "yes_rate",
        "top_shaib_taxonomy_label",
        "venn_region",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], warnings: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 4 Human-Perceptibility Summary",
        "",
        "This summary is generated from the Phase 4 annotation package. Blank rows",
        "mean human annotation has not been completed. A candidate should remain",
        "detector-only until enough rows are labeled and the Venn region is",
        "`detector_and_human_perceived`.",
        "",
        "| Feature | Labeled / total | Yes | Context | No | Unclear | Top taxonomy | Venn region |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {feature} | {labeled}/{total} | {yes} | {context} | {no} | {unclear} | {taxonomy} | {region} |".format(
                feature=row["feature"],
                labeled=row["labeled"],
                total=row["total"],
                yes=row["yes"],
                context=row["context_dependent"],
                no=row["no"],
                unclear=row["unclear"],
                taxonomy=row["top_shaib_taxonomy_label"] or "",
                region=row["venn_region"],
            )
        )
    if warnings:
        lines.extend(["", "## Label Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    rows = run_summarize_phase4_human_labels(args)
    print(f"Wrote {len(rows)} Phase 4 human-label summary rows to {args.output}")
    print(f"Wrote Phase 4 human-label summary to {args.summary_output}")


if __name__ == "__main__":
    main()
