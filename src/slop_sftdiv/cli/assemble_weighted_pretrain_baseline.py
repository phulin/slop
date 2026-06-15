from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble a coverage-aware weighted pretraining baseline from recipe "
            "source weights and sampled feature-rate rows."
        )
    )
    parser.add_argument("--source-weights", type=Path, required=True)
    parser.add_argument("--feature-rates", action="append", type=Path, required=True)
    parser.add_argument(
        "--source-map",
        action="append",
        default=[],
        help="Map feature-rate source name to recipe source name as FEATURE_SOURCE=RECIPE_SOURCE.",
    )
    parser.add_argument("--role", default="pretrain_document")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/phase3/analysis/weighted_pretrain_baseline.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("artifacts/phase3/analysis/weighted_pretrain_baseline_summary.md"),
    )
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)


def _parse_source_map(items: list[str]) -> dict[str, str]:
    source_map: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --source-map entry {item!r}; expected FEATURE_SOURCE=RECIPE_SOURCE")
        left, right = item.split("=", 1)
        if not left or not right:
            raise ValueError(f"Invalid --source-map entry {item!r}; both sides are required")
        source_map[left] = right
    return source_map


def _source_weights(path: Path) -> dict[str, float]:
    weights = {}
    for row in _read_csv(path):
        source_name = str(row["source_name"])
        weights[source_name] = _float(row["share"])
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Source weights sum to zero")
    return weights


def _sample_rates(
    paths: list[Path],
    *,
    source_map: dict[str, str],
    role: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    counts: dict[tuple[str, str], float] = defaultdict(float)
    tokens: dict[tuple[str, str], float] = defaultdict(float)
    fallback_rates: dict[tuple[str, str], list[float]] = defaultdict(list)
    feature_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    for path in paths:
        for row in _read_csv(path):
            if str(row.get("role", "")) != role:
                continue
            feature = str(row.get("feature", ""))
            raw_source = str(row.get("source", ""))
            recipe_source = source_map.get(raw_source)
            if not feature or recipe_source is None:
                continue
            key = (feature, recipe_source)
            feature_sources[key].add(raw_source)
            row_count = _float(row.get("count"), default=-1.0)
            row_tokens = _float(row.get("tokens"), default=-1.0)
            if row_count >= 0 and row_tokens > 0:
                counts[key] += row_count
                tokens[key] += row_tokens
            elif row.get("per_1k_tokens") not in (None, ""):
                fallback_rates[key].append(float(row["per_1k_tokens"]))

    rates: dict[tuple[str, str], dict[str, Any]] = {}
    for key in set(counts) | set(fallback_rates):
        if tokens.get(key, 0.0) > 0:
            rate = counts[key] / tokens[key] * 1000.0
            sample_tokens = tokens[key]
            sample_count = counts[key]
        else:
            values = fallback_rates[key]
            rate = sum(values) / len(values)
            sample_tokens = None
            sample_count = None
        rates[key] = {
            "per_1k_tokens": rate,
            "sample_tokens": sample_tokens,
            "sample_count": sample_count,
            "feature_rate_sources": sorted(feature_sources[key]),
        }
    return rates


def _assemble_rows(
    weights: dict[str, float],
    rates: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    by_feature: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for (feature, recipe_source), rate_info in rates.items():
        if recipe_source in weights:
            by_feature[feature].append((recipe_source, rate_info))
    rows: list[dict[str, Any]] = []
    for feature, entries in sorted(by_feature.items()):
        covered_share = sum(weights[source] for source, _ in entries)
        weighted_sum = sum(weights[source] * info["per_1k_tokens"] for source, info in entries)
        covered_only = weighted_sum / covered_share if covered_share > 0 else None
        matched_sources = []
        sample_tokens = 0.0
        sample_count = 0.0
        for source, info in sorted(entries):
            source_sample_tokens = info["sample_tokens"]
            source_sample_count = info["sample_count"]
            if source_sample_tokens is not None:
                sample_tokens += source_sample_tokens
            if source_sample_count is not None:
                sample_count += source_sample_count
            matched_sources.append(
                "{source}:{rate:.6f}@{share:.6f}".format(
                    source=source,
                    rate=info["per_1k_tokens"],
                    share=weights[source],
                )
            )
        rows.append(
            {
                "feature": feature,
                "role": "pretrain_document",
                "matched_recipe_sources": ";".join(matched_sources),
                "matched_source_count": len(entries),
                "covered_recipe_share": covered_share,
                "missing_recipe_share": max(0.0, 1.0 - covered_share),
                "weighted_per_1k_tokens_covered_only": covered_only,
                "weighted_per_1k_tokens_missing_as_zero": weighted_sum,
                "sample_count": sample_count,
                "sample_tokens": sample_tokens,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature",
        "role",
        "matched_recipe_sources",
        "matched_source_count",
        "covered_recipe_share",
        "missing_recipe_share",
        "weighted_per_1k_tokens_covered_only",
        "weighted_per_1k_tokens_missing_as_zero",
        "sample_count",
        "sample_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    lines = [
        "# Weighted Pretraining Baseline Coverage",
        "",
        "This table joins sampled feature-rate rows to exact recipe source weights.",
        "It is coverage-aware: missing recipe sources are reported explicitly instead of imputed.",
        "",
        f"Features: {len(rows)}",
        "",
        "| Feature | Covered recipe share | Covered-only /1k | Missing-as-zero /1k | Matched sources |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {feature} | {covered:.3%} | {covered_rate:.6f} | {zero_rate:.6f} | {sources} |".format(
                feature=row["feature"],
                covered=float(row["covered_recipe_share"]),
                covered_rate=float(row["weighted_per_1k_tokens_covered_only"] or 0.0),
                zero_rate=float(row["weighted_per_1k_tokens_missing_as_zero"]),
                sources=row["matched_recipe_sources"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `weighted_per_1k_tokens_covered_only` renormalizes over the recipe sources "
            "that currently have sampled feature rates.",
            "- `weighted_per_1k_tokens_missing_as_zero` is a lower-bound diagnostic, not a "
            "real estimate, because unsampled sources are treated as zero.",
            "- A production weighted baseline requires enough feature-rate coverage to make "
            "the missing recipe share acceptably small or an explicit source-group proxy.",
            "",
            "## Inputs",
            "",
            f"- Source weights: `{args.source_weights}`",
        ]
    )
    for path in args.feature_rates:
        lines.append(f"- Feature rates: `{path}`")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    source_map = _parse_source_map(args.source_map)
    weights = _source_weights(args.source_weights)
    rates = _sample_rates(args.feature_rates, source_map=source_map, role=args.role)
    rows = _assemble_rows(weights, rates)
    _write_csv(args.output, rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(_summary(rows, args), encoding="utf-8")
    return rows


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
