from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd


DISCOURSE_RE = re.compile(r"\b(however|furthermore|additionally|moreover|for example|in addition|such as)\b", re.I)
NEGATION_RE = re.compile(r"\b(didn['\u2019]t|doesn['\u2019]t|wasn['\u2019]t|don['\u2019]t|hadn['\u2019]t|couldn['\u2019]t|wouldn['\u2019]t)\b", re.I)
WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)?")
SENTENCE_RE = re.compile(r"[.!?]+(?:\s+|$)")


def _shorten(text: str, limit: int = 220) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _load_texts(target_csv: Path, doc_tokens_parquet: Path) -> dict[str, dict[str, Any]]:
    targets = pd.read_csv(target_csv)
    if "text" in targets.columns:
        return {
            str(row["doc_id"]): {
                "text": str(row["text"]),
                "variant": str(row.get("variant", "")),
                "model": str(row.get("model", row.get("source", ""))),
            }
            for row in targets.to_dict(orient="records")
        }
    doc_ids = list(dict.fromkeys(str(value) for value in targets["doc_id"].tolist()))
    docs = pd.read_parquet(doc_tokens_parquet, columns=["doc_id", "model", "text"])
    docs = docs[docs["doc_id"].astype(str).isin(set(doc_ids))]
    return {
        str(row.doc_id): {
            "text": str(row.text),
            "variant": "",
            "model": str(row.model),
        }
        for row in docs.itertuples(index=False)
    }


def _text_metrics(text: str, feature: str) -> dict[str, Any]:
    words = WORD_RE.findall(text)
    sentences = SENTENCE_RE.findall(text)
    pattern = DISCOURSE_RE if feature == "discourse_marker" else NEGATION_RE
    hits = pattern.findall(text)
    word_count = len(words)
    sentence_count = max(1, len(sentences))
    return {
        "chars": len(text),
        "words": word_count,
        "sentences": sentence_count,
        "feature_hits": len(hits),
        "feature_hits_per_100_words": (100.0 * len(hits) / word_count) if word_count else 0.0,
        "words_per_sentence": word_count / sentence_count if sentence_count else 0.0,
        "text_preview": _shorten(text),
    }


def _add_rows(
    rows: list[dict[str, Any]],
    *,
    feature: str,
    group: str,
    target_csv: Path,
    intervention_csv: Path,
    doc_tokens_parquet: Path,
) -> None:
    text_by_id = _load_texts(target_csv, doc_tokens_parquet)
    intervention = pd.read_csv(intervention_csv)
    for row in intervention.to_dict(orient="records"):
        doc_id = str(row["doc_id"])
        text_row = text_by_id.get(doc_id)
        if text_row is None:
            base_doc_id = doc_id.split("::", 1)[0]
            text_row = text_by_id.get(base_doc_id)
        if text_row is None:
            continue
        text = str(text_row["text"])
        variant = str(text_row.get("variant") or "")
        if not variant:
            if doc_id.endswith("_marked"):
                variant = "marked"
            elif doc_id.endswith("_plain"):
                variant = "plain"
            elif doc_id.endswith("::original"):
                variant = "original"
            elif doc_id.endswith("::edited"):
                variant = "edited"
        rows.append(
            {
                "feature": feature,
                "group": group,
                "variant": variant,
                "doc_id": doc_id,
                "model": str(text_row.get("model") or row.get("source", "")),
                "upstream_baseline_mass": float(row.get("upstream_baseline_mass", 0.0)),
                "upstream_baseline_max": float(row.get("upstream_baseline_max", 0.0)),
                "downstream_baseline_mass": float(row.get("downstream_baseline_mass", 0.0)),
                "downstream_baseline_max": float(row.get("downstream_baseline_max", 0.0)),
                "downstream_mass_delta": float(row.get("downstream_mass_delta", 0.0)),
                "target_logit_drop": float(row.get("target_logit_drop", 0.0)),
                **_text_metrics(text, feature),
            }
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metrics = [
        "chars",
        "words",
        "sentences",
        "feature_hits",
        "feature_hits_per_100_words",
        "words_per_sentence",
        "upstream_baseline_mass",
        "downstream_baseline_mass",
        "downstream_baseline_max",
        "downstream_mass_delta",
        "target_logit_drop",
    ]
    summary = (
        df.groupby(["feature", "group", "variant"], dropna=False)[metrics]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
    )
    summary.columns = ["_".join(str(part) for part in column if part) for column in summary.columns.to_flat_index()]
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return summary


def _markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in df.to_dict(orient="records"):
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.3f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_md(path: Path, rows: list[dict[str, Any]], summary: pd.DataFrame) -> None:
    df = pd.DataFrame(rows)
    summary_view = summary[
        [
            "feature",
            "group",
            "variant",
            "words_mean",
            "feature_hits_mean",
            "feature_hits_per_100_words_mean",
            "upstream_baseline_mass_mean",
            "downstream_baseline_mass_mean",
            "downstream_mass_delta_mean",
            "target_logit_drop_mean",
        ]
    ]
    lines = [
        "# Pangram SAE Lexical Context Analysis",
        "",
        "This compares corpus-derived target rows against short independent minimal pairs for the two lexical SAE paths.",
        "",
        "## Group Summary",
        "",
        _markdown_table(summary_view),
        "",
        "## Highest Downstream Activation Examples",
        "",
    ]
    for feature in sorted(df["feature"].unique()):
        lines.extend([f"### {feature}", ""])
        subset = df[df["feature"] == feature].sort_values("downstream_baseline_mass", ascending=False).head(8)
        for row in subset.to_dict(orient="records"):
            lines.extend(
                [
                    f"- `{row['group']}` `{row['variant']}` `{str(row['doc_id'])[:16]}` "
                    f"model `{row['model']}` downstream `{row['downstream_baseline_mass']:.2f}` "
                    f"upstream `{row['upstream_baseline_mass']:.2f}` words `{row['words']}` hits `{row['feature_hits']}`",
                    f"  - {row['text_preview']}",
                ]
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze lexical SAE context metrics across corpus and minimal-pair target sets.")
    parser.add_argument("--doc-tokens-parquet", type=Path, default=Path("apps/latent-explorer/public/data/layer19/pangram_llama_sae_doc_tokens.parquet"))
    parser.add_argument("--output-details-csv", type=Path, default=Path("artifacts/pangram_llama_sae/circuit_discovery/lexical_context_analysis_details.csv"))
    parser.add_argument("--output-summary-csv", type=Path, default=Path("artifacts/pangram_llama_sae/circuit_discovery/lexical_context_analysis_summary.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("docs/experiments/pangram_sae_lexical_context_analysis.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Path("artifacts/pangram_llama_sae/circuit_discovery")
    specs = [
        (
            "discourse_marker",
            "top_export",
            base / "targets_ai_discourse_marker_L19_3450_to_L24_3469.csv",
            base / "intervention_discourse_marker_L19_3450_to_L24_3469.csv",
        ),
        (
            "discourse_marker",
            "heldout",
            base / "heldout_discourse_marker_targets.csv",
            base / "heldout_intervention_discourse_marker_L19_3450_to_L24_3469.csv",
        ),
        (
            "discourse_marker",
            "counterfactual",
            base / "counterfactual_discourse_marker_targets.csv",
            base / "counterfactual_intervention_discourse_marker_L19_3450_to_L24_3469.csv",
        ),
        (
            "discourse_marker",
            "minimal_pair",
            base / "minimal_pair_discourse_marker_targets.csv",
            base / "minimal_pair_intervention_discourse_marker_L19_3450_to_L24_3469.csv",
        ),
        (
            "discourse_marker",
            "long_stress",
            base / "long_stress_discourse_marker_targets.csv",
            base / "long_stress_intervention_discourse_marker_L19_3450_to_L24_3469.csv",
        ),
        (
            "discourse_marker",
            "combined_long_stress",
            base / "combined_long_stress_discourse_marker_targets.csv",
            base / "combined_long_stress_intervention_discourse_marker_L19_3450_to_L24_3469.csv",
        ),
        (
            "negation_fragment",
            "top_export",
            base / "targets_ai_negation_fragment_L19_913_to_L24_1310.csv",
            base / "intervention_negation_fragment_L19_913_to_L24_1310.csv",
        ),
        (
            "negation_fragment",
            "heldout",
            base / "heldout_negation_fragment_targets.csv",
            base / "heldout_intervention_negation_fragment_L19_913_to_L24_1310.csv",
        ),
        (
            "negation_fragment",
            "counterfactual",
            base / "counterfactual_negation_fragment_targets.csv",
            base / "counterfactual_intervention_negation_fragment_L19_913_to_L24_1310.csv",
        ),
        (
            "negation_fragment",
            "minimal_pair",
            base / "minimal_pair_negation_fragment_targets.csv",
            base / "minimal_pair_intervention_negation_fragment_L19_913_to_L24_1310.csv",
        ),
        (
            "negation_fragment",
            "long_stress",
            base / "long_stress_negation_fragment_targets.csv",
            base / "long_stress_intervention_negation_fragment_L19_913_to_L24_1310.csv",
        ),
        (
            "negation_fragment",
            "combined_long_stress",
            base / "combined_long_stress_negation_fragment_targets.csv",
            base / "combined_long_stress_intervention_negation_fragment_L19_913_to_L24_1310.csv",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for feature, group, target_csv, intervention_csv in specs:
        _add_rows(
            rows,
            feature=feature,
            group=group,
            target_csv=target_csv,
            intervention_csv=intervention_csv,
            doc_tokens_parquet=args.doc_tokens_parquet,
        )
    _write_csv(args.output_details_csv, rows)
    summary = _write_summary(args.output_summary_csv, rows)
    _write_md(args.output_md, rows, summary)
    print(
        {
            "rows": len(rows),
            "output_details_csv": str(args.output_details_csv),
            "output_summary_csv": str(args.output_summary_csv),
            "output_md": str(args.output_md),
        }
    )


if __name__ == "__main__":
    main()
