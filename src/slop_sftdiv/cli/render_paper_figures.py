from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "axes.edgecolor": "#6f6f6f",
        "axes.labelcolor": "#202020",
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "legend.fontsize": 9,
        "legend.title_fontsize": 9,
        "svg.fonttype": "none",
        "xtick.color": "#303030",
        "ytick.color": "#303030",
    }
)

DEFAULT_PYBIBER_MEANS = Path("artifacts/stage1/census/phase1_pybiber_register_means.csv")
DEFAULT_PYBIBER_INTERVALS = Path("artifacts/stage1/census/phase1_pybiber_register_intervals.csv")
DEFAULT_EQBENCH = Path("artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_stage_comparison.csv")
DEFAULT_EQBENCH_INTERVALS = Path(
    "artifacts/phase2/analysis/eqbench_slop/phase2_eqbench_slop_intervals.csv"
)
DEFAULT_OLMO_SPECTRUM = Path(
    "artifacts/phase3/analysis/olmo3_phase3_reduced_sglang_amplification_spectrum_t07_long1024.csv"
)
DEFAULT_CROSS_LADDER_ALIGNED = Path(
    "artifacts/phase3/analysis/olmo3_reduced_sglang_vs_smollm3_no_think_production_aligned.csv"
)
DEFAULT_PHASE4_STAGE_GRID = Path(
    "artifacts/phase4/modernbert_detector_combined_v2_clean/"
    "tier3_teacher_forced_exact_512/olmo3_phase4_tier3_512_exact_sequence_stage_grid.csv"
)

STAGE_ORDER = ("base", "sft", "dpo", "final")
STAGE_LABELS = {
    "base": "Base",
    "sft": "SFT",
    "dpo": "DPO",
    "final": "Final",
}
LADDER_LABELS = {
    "olmo3": "OLMo 3",
    "smollm3_no_think": "SmolLM3 no-think",
}
PYBIBER_FEATURE_LABELS = {
    "f_01_past_tense": "Past tense",
    "f_06_first_person_pronouns": "First-person pronouns",
    "f_08_third_person_pronouns": "Third-person pronouns",
    "f_14_nominalizations": "Nominalizations",
    "f_39_prepositions": "Prepositions",
    "f_40_adj_attr": "Attributive adjectives",
    "f_42_adverbs": "Adverbs",
    "f_47_hedges": "Hedges",
    "f_48_amplifiers": "Amplifiers",
    "f_49_emphatics": "Emphatics",
}
SAMPLE_LABELS = {
    "dolma_pretrain": "Dolma",
    "sft_target": "SFT",
    "dpo_chosen": "DPO chosen",
    "dpo_rejected": "DPO rejected",
}
PHASE4_FEATURE_LABELS = {
    "phase4_ig_process_framing": "Process framing",
    "phase4_ig_additive_transition": "Additive transition",
    "phase4_ig_prescriptive_instruction": "Prescriptive instruction",
    "phase4_ig_response_constraint": "Response constraint",
    "phase4_ig_followup_offer": "Follow-up offer",
    "neutral_common_controls": "Neutral controls",
}
SAMPLE_COLORS = {
    "Dolma": "#4c78a8",
    "SFT": "#f58518",
    "DPO chosen": "#54a24b",
    "DPO rejected": "#b279a2",
}
LADDER_COLORS = {
    "olmo3": "#3568a6",
    "smollm3_no_think": "#c44e52",
}
STAGE_COLORS = {
    "Base": "#4c78a8",
    "SFT": "#f58518",
    "DPO": "#54a24b",
    "Final/RLVR": "#b279a2",
}
GRID_COLOR = "#dddddd"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render paper draft figures from cached artifacts.")
    parser.add_argument("--pybiber-means", type=Path, default=DEFAULT_PYBIBER_MEANS)
    parser.add_argument("--pybiber-intervals", type=Path, default=DEFAULT_PYBIBER_INTERVALS)
    parser.add_argument("--eqbench-stage-comparison", type=Path, default=DEFAULT_EQBENCH)
    parser.add_argument("--eqbench-intervals", type=Path, default=DEFAULT_EQBENCH_INTERVALS)
    parser.add_argument("--olmo-spectrum", type=Path, default=DEFAULT_OLMO_SPECTRUM)
    parser.add_argument("--cross-ladder-aligned", type=Path, default=DEFAULT_CROSS_LADDER_ALIGNED)
    parser.add_argument("--phase4-stage-grid", type=Path, default=DEFAULT_PHASE4_STAGE_GRID)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/paper/figures"))
    return parser


def run_render_paper_figures(args: argparse.Namespace) -> list[Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        _render_pybiber_register_figure(
            args.pybiber_means,
            getattr(args, "pybiber_intervals", DEFAULT_PYBIBER_INTERVALS),
            args.output_dir,
        ),
        _render_eqbench_stage_figure(
            args.eqbench_stage_comparison,
            getattr(args, "eqbench_intervals", DEFAULT_EQBENCH_INTERVALS),
            args.output_dir,
        ),
        _render_olmo_slop_lexicon_figure(args.olmo_spectrum, args.output_dir),
        _render_cross_ladder_figure(args.cross_ladder_aligned, args.output_dir),
        _render_phase4_tier3_figure(args.phase4_stage_grid, args.output_dir),
    ]
    return outputs


def _render_pybiber_register_figure(
    input_path: Path,
    intervals_path: Path | None,
    output_dir: Path,
) -> Path:
    df = pd.read_csv(input_path)
    _require_columns(
        df,
        {"sample", "feature", "token_weighted_mean"},
        input_path,
    )
    if intervals_path is not None and intervals_path.exists():
        intervals = pd.read_csv(intervals_path)
        _require_columns(
            intervals,
            {"statistic", "sample", "feature", "ci_low", "ci_high"},
            intervals_path,
        )
        intervals = intervals[intervals["statistic"] == "mean"][
            ["sample", "feature", "ci_low", "ci_high"]
        ]
        df = df.merge(intervals, on=["sample", "feature"], how="left")
    features = list(PYBIBER_FEATURE_LABELS)
    plot_df = df[df["feature"].isin(features)].copy()
    plot_df["feature_label"] = plot_df["feature"].map(PYBIBER_FEATURE_LABELS)
    plot_df["sample_label"] = plot_df["sample"].map(SAMPLE_LABELS).fillna(plot_df["sample"])
    pivot = plot_df.pivot_table(
        index="feature_label",
        columns="sample_label",
        values="token_weighted_mean",
        aggfunc="first",
    )
    pivot = pivot.reindex([PYBIBER_FEATURE_LABELS[feature] for feature in features])
    ordered_columns = [label for label in SAMPLE_LABELS.values() if label in pivot.columns]
    pivot = pivot[ordered_columns]

    fig, ax = plt.subplots(figsize=(10, 6.8))
    if {"ci_low", "ci_high"}.issubset(plot_df.columns) and not plot_df["ci_low"].isna().all():
        low = plot_df.pivot_table(
            index="feature_label",
            columns="sample_label",
            values="ci_low",
            aggfunc="first",
        ).reindex(pivot.index)[ordered_columns]
        high = plot_df.pivot_table(
            index="feature_label",
            columns="sample_label",
            values="ci_high",
            aggfunc="first",
        ).reindex(pivot.index)[ordered_columns]
        y_base = np.arange(len(pivot.index))
        bar_height = 0.78 / max(len(pivot.columns), 1)
        for column_index, column in enumerate(pivot.columns):
            offset = (column_index - (len(pivot.columns) - 1) / 2) * bar_height
            values = pivot[column]
            xerr = np.vstack(
                [
                    values - low[column],
                    high[column] - values,
                ]
            )
            ax.barh(
                y_base + offset,
                values,
                height=bar_height * 0.94,
                color=SAMPLE_COLORS.get(column, "#666666"),
                label=column,
                xerr=xerr,
                error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
            )
        ax.set_yticks(y_base)
        ax.set_yticklabels(pivot.index)
    else:
        pivot.plot(
            kind="barh",
            ax=ax,
            width=0.78,
            color=[SAMPLE_COLORS.get(column, "#666666") for column in pivot.columns],
        )
    ax.set_title("Phase 1 register shift in pybiber features")
    ax.set_xlabel("Token-weighted pybiber rate")
    ax.set_ylabel("")
    _style_axis(ax, grid_axis="x")
    ax.legend(title="", loc="lower right", frameon=False, ncol=2)
    _finish_figure(fig)
    output_path = output_dir / "figure1_pybiber_register_selected.svg"
    _save_svg(fig, output_path)
    plt.close(fig)
    return output_path


def _render_eqbench_stage_figure(input_path: Path, intervals_path: Path | None, output_dir: Path) -> Path:
    df = pd.read_csv(input_path)
    _require_columns(df, {"ladder", "stage", "eqbench_slop_score"}, input_path)
    df = df[df["stage"].isin(STAGE_ORDER)].copy()
    if intervals_path is not None and intervals_path.exists():
        intervals = pd.read_csv(intervals_path)
        _require_columns(
            intervals,
            {
                "ladder",
                "stage",
                "eqbench_slop_score_ci_low",
                "eqbench_slop_score_ci_high",
            },
            intervals_path,
        )
        df = df.merge(
            intervals[
                [
                    "ladder",
                    "stage",
                    "eqbench_slop_score_ci_low",
                    "eqbench_slop_score_ci_high",
                ]
            ],
            on=["ladder", "stage"],
            how="left",
        )
    df["stage"] = pd.Categorical(df["stage"], categories=STAGE_ORDER, ordered=True)
    df = df.sort_values(["ladder", "stage"])

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for ladder, group in df.groupby("ladder", sort=False):
        x = [STAGE_LABELS[str(stage)] for stage in group["stage"]]
        y = group["eqbench_slop_score"]
        yerr = None
        if {
            "eqbench_slop_score_ci_low",
            "eqbench_slop_score_ci_high",
        }.issubset(group.columns) and not group["eqbench_slop_score_ci_low"].isna().all():
            yerr = [
                y - group["eqbench_slop_score_ci_low"],
                group["eqbench_slop_score_ci_high"] - y,
            ]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=LADDER_LABELS.get(str(ladder), str(ladder)),
            color=LADDER_COLORS.get(str(ladder), "#666666"),
            capsize=3 if yerr is not None else 0,
        )
    ax.set_title("EQ-Bench Slop Score by checkpoint")
    ax.set_ylabel("EQ-Bench Slop Score")
    ax.set_xlabel("")
    _style_axis(ax, grid_axis="y")
    ax.legend(frameon=False)
    _finish_figure(fig)
    output_path = output_dir / "figure2_eqbench_stage_scores.svg"
    _save_svg(fig, output_path)
    plt.close(fig)
    return output_path


def _render_olmo_slop_lexicon_figure(input_path: Path, output_dir: Path) -> Path:
    df = pd.read_csv(input_path)
    _require_columns(
        df,
        {
            "feature",
            "stage",
            "teacher_forced_normalized_af",
            "free_run_per_1k_tokens",
            "compounding_excess_per_1k_opportunities",
        },
        input_path,
    )
    df = df[(df["feature"] == "slop_lexicon") & df["stage"].isin(STAGE_ORDER)].copy()
    df["stage"] = pd.Categorical(df["stage"], categories=STAGE_ORDER, ordered=True)
    df = df.sort_values("stage")
    x = [STAGE_LABELS[str(stage)] for stage in df["stage"]]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharex=True)
    series = (
        ("teacher_forced_normalized_af", "Teacher-forced normalized AF"),
        ("free_run_per_1k_tokens", "Free-run hits / 1k tokens"),
        ("compounding_excess_per_1k_opportunities", "Compounding excess / 1k opps"),
    )
    for ax, (column, title) in zip(axes, series, strict=True):
        ax.plot(x, df[column], marker="o", linewidth=2.2, markersize=6, color="#3568a6")
        ax.set_title(title, fontsize=11)
        _style_axis(ax, grid_axis="y")
    fig.suptitle("OLMo slop_lexicon amplification views", y=1.01, fontsize=13, weight="bold")
    _finish_figure(fig)
    output_path = output_dir / "figure3_olmo_slop_lexicon_views.svg"
    _save_svg(fig, output_path)
    plt.close(fig)
    return output_path


def _render_cross_ladder_figure(input_path: Path, output_dir: Path) -> Path:
    df = pd.read_csv(input_path)
    _require_columns(
        df,
        {"feature", "stage_label", "left_af", "right_af", "left_label", "right_label"},
        input_path,
    )
    df = df.dropna(subset=["left_af", "right_af"]).copy()
    if df.empty:
        raise ValueError(f"{input_path} has no rows with both left_af and right_af")

    fig, ax = plt.subplots(figsize=(7, 5.6))
    for stage_label, group in df.groupby("stage_label", sort=False):
        ax.scatter(
            group["left_af"],
            group["right_af"],
            s=64,
            alpha=0.85,
            label=str(stage_label),
            color=STAGE_COLORS.get(str(stage_label), "#666666"),
            edgecolor="white",
            linewidth=0.6,
        )
    for feature, group in df.groupby("feature", sort=False):
        x = float(np.exp(np.log(group["left_af"]).mean()))
        y = float(np.exp(np.log(group["right_af"]).mean()))
        x_offset = 6 if x < 1 else 8
        y_offset = 6 if y < 1 else 8
        ax.annotate(
            _short_feature_label(str(feature)),
            (x, y),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=8.5,
            weight="bold",
        )
    min_value = max(0.01, float(min(df["left_af"].min(), df["right_af"].min())) * 0.8)
    max_value = float(max(df["left_af"].max(), df["right_af"].max())) * 1.2
    ax.plot([min_value, max_value], [min_value, max_value], color="#777777", linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min_value, max_value)
    ax.set_ylim(min_value, max_value)
    ax.set_title("Cross-ladder AF comparison")
    ax.set_xlabel("OLMo AF")
    ax.set_ylabel("SmolLM3 AF")
    _style_axis(ax, grid_axis="both")
    ax.legend(frameon=False, title="Stage")
    _finish_figure(fig)
    output_path = output_dir / "figure4_cross_ladder_af_scatter.svg"
    _save_svg(fig, output_path)
    plt.close(fig)
    return output_path


def _render_phase4_tier3_figure(input_path: Path, output_dir: Path) -> Path:
    df = pd.read_csv(input_path)
    _require_columns(
        df,
        {"feature", "stage", "amplification_factor", "reference_initiations"},
        input_path,
    )
    features = list(PHASE4_FEATURE_LABELS)
    df = df[df["feature"].isin(features) & df["stage"].isin(STAGE_ORDER)].copy()
    df["feature_label"] = df["feature"].map(PHASE4_FEATURE_LABELS)
    df["stage"] = pd.Categorical(df["stage"], categories=STAGE_ORDER, ordered=True)
    df = df.sort_values(["feature_label", "stage"])

    fig, ax = plt.subplots(figsize=(11, 6.2))
    for feature, group in df.groupby("feature_label", sort=False):
        ax.plot(
            [STAGE_LABELS[str(stage)] for stage in group["stage"]],
            group["amplification_factor"],
            marker="o",
            linewidth=2,
            markersize=5.5,
            label=f"{feature} (n={int(group['reference_initiations'].max())})",
        )
    ax.set_title("Phase 4 Tier-3 raw AF by checkpoint")
    ax.set_ylabel("Raw amplification factor")
    ax.set_xlabel("")
    ax.set_yscale("symlog", linthresh=1.0)
    _style_axis(ax, grid_axis="y")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _finish_figure(fig)
    output_path = output_dir / "figure5_phase4_tier3_raw_af.svg"
    _save_svg(fig, output_path)
    plt.close(fig)
    return output_path


def _require_columns(df: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = sorted(columns.difference(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {', '.join(missing)}")


def _finish_figure(fig: Any) -> None:
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()


def _style_axis(ax: Any, grid_axis: str) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color=GRID_COLOR, linewidth=0.7)
    ax.tick_params(length=3.5, width=0.8)


def _save_svg(fig: Any, output_path: Path) -> None:
    fig.savefig(output_path, format="svg", metadata={"Date": "2026-06-18"}, bbox_inches="tight")
    export_dir = output_path.parent / "submission_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_base = export_dir / output_path.stem
    fig.savefig(export_base.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(export_base.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")


def _short_feature_label(feature: str) -> str:
    return {
        "slop_lexicon": "slop",
        "rule_of_three_approx": "rule3",
        "contrastive_negation": "contrast",
        "stock_openers": "openers",
        "stock_closers": "closers",
        "stock_openers_closers": "stock",
    }.get(feature, feature)


def main() -> None:
    args = build_parser().parse_args()
    outputs = run_render_paper_figures(args)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
