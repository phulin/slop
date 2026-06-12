from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


PAIR_OUTPUT_COLUMNS = [
    "source",
    "subset",
    "pair_id",
    "feature",
    "chosen_count",
    "rejected_count",
    "chosen_tokens",
    "rejected_tokens",
    "chosen_rate_per_1k_tokens",
    "rejected_rate_per_1k_tokens",
    "chosen_minus_rejected",
]
MAX_WANDB_PAIR_ROWS = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sampled Phase 1 feature census.")
    parser.add_argument("--input", action="append", required=True, help="Local JSONL file or HF dataset id.")
    parser.add_argument("--split", default="train", help="Dataset split for HF inputs.")
    parser.add_argument("--hf-config", default=None, help="Optional Hugging Face dataset config name.")
    parser.add_argument("--text-field", default=None, help="Override text field name.")
    parser.add_argument("--sample-size", type=int, default=10_000, help="Maximum records per input.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--sampling-strategy",
        default="first",
        choices=["first", "hash_reservoir"],
        help="Deterministic row-selection strategy.",
    )
    parser.add_argument("--max-scan", type=int, default=None, help="Maximum source rows to scan.")
    parser.add_argument("--output", type=Path, default=Path("runs/stage1_census/summary.csv"))
    parser.add_argument(
        "--pair-output",
        type=Path,
        default=None,
        help="Optional CSV path for per-pair chosen/rejected feature deltas.",
    )
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _load_components():
    # Imported lazily so `--help` works before optional dependencies are installed.
    from slop_sftdiv.data import CorpusSource, SamplingConfig, iter_corpus_records
    from slop_sftdiv.data.corpora import DEFAULT_ID_FIELDS
    from slop_sftdiv.features import extract_tier1_features

    return CorpusSource, SamplingConfig, iter_corpus_records, extract_tier1_features, DEFAULT_ID_FIELDS


def _source_from_input(raw_input: str, *, split: str, hf_config: str | None):
    CorpusSource, _, _, _, _ = _load_components()
    path = Path(raw_input)
    if path.exists():
        return CorpusSource.jsonl(path)
    return CorpusSource.hf(raw_input, split=split, hf_config=hf_config, streaming=True)


def _measurement_texts(record: Any) -> list[tuple[str, str, str | None]]:
    if record.chosen and record.rejected:
        pair_id = record.record_id
        return [("chosen", record.chosen, pair_id), ("rejected", record.rejected, pair_id)]
    return [("text", record.text, None)]


def _id_fields_with_pair_id(default_id_fields: tuple[str, ...]) -> tuple[str, ...]:
    fields = ("pair_id", "pair.id", "preference_pair_id", "prompt_id", *default_id_fields)
    return tuple(dict.fromkeys(fields))


def _metadata_without_raw_text(row: dict[str, Any]) -> dict[str, Any]:
    raw_text_keys = {"text", "prompt", "chosen", "rejected", "raw"}
    return {key: value for key, value in row.items() if key not in raw_text_keys}


def _rate_per_1k(count: int, tokens: int) -> float:
    return 1000.0 * count / max(tokens, 1)


def _pair_rows(
    *,
    source: str,
    subset: str,
    pair_id: str,
    chosen_counts: Counter[str],
    rejected_counts: Counter[str],
    chosen_tokens: int,
    rejected_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in sorted(set(chosen_counts) | set(rejected_counts)):
        chosen_count = chosen_counts[feature]
        rejected_count = rejected_counts[feature]
        chosen_rate = _rate_per_1k(chosen_count, chosen_tokens)
        rejected_rate = _rate_per_1k(rejected_count, rejected_tokens)
        rows.append(
            {
                "source": source,
                "subset": subset,
                "pair_id": pair_id,
                "feature": feature,
                "chosen_count": chosen_count,
                "rejected_count": rejected_count,
                "chosen_tokens": chosen_tokens,
                "rejected_tokens": rejected_tokens,
                "chosen_rate_per_1k_tokens": chosen_rate,
                "rejected_rate_per_1k_tokens": rejected_rate,
                "chosen_minus_rejected": chosen_rate - rejected_rate,
            }
        )
    return rows


def run_census(args: argparse.Namespace) -> pd.DataFrame:
    _, SamplingConfig, iter_corpus_records, extract_tier1_features, DEFAULT_ID_FIELDS = _load_components()
    feature_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    token_counts: Counter[tuple[str, str, str]] = Counter()
    doc_counts: Counter[tuple[str, str, str]] = Counter()
    sample_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    text_fields = [args.text_field] if args.text_field else None
    id_fields = _id_fields_with_pair_id(DEFAULT_ID_FIELDS)
    sampling = SamplingConfig(
        cap=args.sample_size,
        seed=args.seed,
        strategy=args.sampling_strategy,
        max_scan=args.max_scan,
    )

    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        mode=args.wandb_mode,
        tags=["stage1", "phase1", "census"],
        config={
            "inputs": args.input,
            "split": args.split,
            "hf_config": args.hf_config,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "text_field": args.text_field,
            "sampling_strategy": args.sampling_strategy,
            "max_scan": args.max_scan,
            "pair_output": str(args.pair_output) if args.pair_output is not None else None,
        },
    )

    try:
        for raw_source in args.input:
            source = _source_from_input(raw_source, split=args.split, hf_config=args.hf_config)
            records_kwargs = {"sampling": sampling, "id_fields": id_fields}
            if text_fields:
                records_kwargs["text_fields"] = text_fields
            records = iter_corpus_records(source, **records_kwargs)
            for record in tqdm(records, desc=f"census:{source.name}", unit="doc"):
                subset = str(
                    record.metadata.get("stratum")
                    or record.metadata.get("provenance")
                    or record.split
                    or "default"
                )
                pair_analyses: dict[str, tuple[Counter[str], int]] = {}
                for role, text, pair_id in _measurement_texts(record):
                    analysis = extract_tier1_features(text)
                    token_count = int(analysis.helpers["token_count"])
                    key = (record.source, subset, role)
                    feature_counts[key].update(analysis.counts)
                    token_counts[key] += token_count
                    doc_counts[key] += 1
                    if pair_id is not None and role in {"chosen", "rejected"}:
                        pair_analyses[role] = (Counter(analysis.counts), token_count)
                    if len(sample_rows) < 200:
                        row = record.wandb_row(include_raw=False)
                        sample_rows.append(
                            {
                                "source": record.source,
                                "subset": subset,
                                "role": role,
                                "record_id": record.record_id,
                                "pair_id": pair_id,
                                "token_count": token_count,
                                "metadata": json.dumps(
                                    _metadata_without_raw_text(row),
                                    sort_keys=True,
                                    default=str,
                                ),
                            }
                        )
                if "chosen" in pair_analyses and "rejected" in pair_analyses:
                    chosen_counts, chosen_tokens = pair_analyses["chosen"]
                    rejected_counts, rejected_tokens = pair_analyses["rejected"]
                    pair_rows.extend(
                        _pair_rows(
                            source=record.source,
                            subset=subset,
                            pair_id=record.record_id,
                            chosen_counts=chosen_counts,
                            rejected_counts=rejected_counts,
                            chosen_tokens=chosen_tokens,
                            rejected_tokens=rejected_tokens,
                        )
                    )

        rows: list[dict[str, Any]] = []
        for key, counts in sorted(feature_counts.items()):
            source, subset, role = key
            tokens = max(token_counts[key], 1)
            docs = max(doc_counts[key], 1)
            for feature, count in sorted(counts.items()):
                rows.append(
                    {
                        "source": source,
                        "subset": subset,
                        "role": role,
                        "feature": feature,
                        "count": count,
                        "docs": docs,
                        "tokens": tokens,
                        "per_1k_tokens": 1000.0 * count / tokens,
                        "per_doc": count / docs,
                    }
                )

        frame = pd.DataFrame(rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        if args.pair_output is not None:
            pair_frame = pd.DataFrame(pair_rows, columns=PAIR_OUTPUT_COLUMNS)
            args.pair_output.parent.mkdir(parents=True, exist_ok=True)
            pair_frame.to_csv(args.pair_output, index=False)
        run.log(
            {
                "summary_rows": len(frame),
                "pair_rows": len(pair_rows),
                "pair_rows_logged": min(len(pair_rows), MAX_WANDB_PAIR_ROWS),
            }
        )
        if rows:
            log_summary_table(run, "feature_rates", rows)
        if pair_rows:
            log_summary_table(run, "preference_pair_deltas", pair_rows[:MAX_WANDB_PAIR_ROWS])
        log_summary_table(run, "sampled_records", sample_rows)
        return frame
    finally:
        run.finish()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    frame = run_census(args)
    print(f"Wrote {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
