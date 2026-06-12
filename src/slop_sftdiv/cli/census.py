from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


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
    fields = ("pair_id", "pair.id", "preference_pair_id", *default_id_fields)
    return tuple(dict.fromkeys(fields))


def _metadata_without_raw_text(row: dict[str, Any]) -> dict[str, Any]:
    raw_text_keys = {"text", "prompt", "chosen", "rejected", "raw"}
    return {key: value for key, value in row.items() if key not in raw_text_keys}


def run_census(args: argparse.Namespace) -> pd.DataFrame:
    _, SamplingConfig, iter_corpus_records, extract_tier1_features, DEFAULT_ID_FIELDS = _load_components()
    feature_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    token_counts: Counter[tuple[str, str, str]] = Counter()
    doc_counts: Counter[tuple[str, str, str]] = Counter()
    sample_rows: list[dict[str, Any]] = []
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
                for role, text, pair_id in _measurement_texts(record):
                    analysis = extract_tier1_features(text)
                    token_count = int(analysis.helpers["token_count"])
                    key = (record.source, subset, role)
                    feature_counts[key].update(analysis.counts)
                    token_counts[key] += token_count
                    doc_counts[key] += 1
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
        run.log({"summary_rows": len(frame)})
        if rows:
            log_summary_table(run, "feature_rates", rows)
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
