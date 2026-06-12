from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from slop_sftdiv.cli.census import _id_fields_with_pair_id, _measurement_texts, _source_from_input
from slop_sftdiv.wandb_utils import init_wandb, log_summary_table


LABEL_COLUMNS = [
    "label",
    "labeler",
    "notes",
    "source",
    "subset",
    "role",
    "record_id",
    "pair_id",
    "feature",
    "subtype",
    "start",
    "end",
    "hit_text",
    "context",
    "sample_key",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample Tier-1 matcher hits for precision labeling.")
    parser.add_argument("--input", action="append", required=True, help="Local JSONL file or HF dataset id.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--text-field", default=None)
    parser.add_argument("--sample-size", type=int, default=10_000, help="Maximum records per input.")
    parser.add_argument("--hits-per-feature", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-scan", type=int, default=None)
    parser.add_argument("--context-chars", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-project", default="slop-stage1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default="precision-validation")
    parser.add_argument("--wandb-job-type", default="hit-sampling")
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-mode", default=None, choices=[None, "online", "offline", "disabled"])
    return parser


def _load_components():
    from slop_sftdiv.data import SamplingConfig, iter_corpus_records
    from slop_sftdiv.data.corpora import DEFAULT_ID_FIELDS
    from slop_sftdiv.features import iter_tier1_hits

    return SamplingConfig, iter_corpus_records, DEFAULT_ID_FIELDS, iter_tier1_hits


def _subset_for_record(metadata: dict[str, Any] | Any, split: str | None) -> str:
    from slop_sftdiv.cli.census import _subset_for_record as census_subset_for_record

    return census_subset_for_record(metadata, split)


def _sample_key(*parts: Any, seed: int) -> str:
    payload = "\0".join(str(part) for part in (seed, *parts)).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _candidate_row(
    *,
    source: str,
    subset: str,
    role: str,
    record_id: str,
    pair_id: str | None,
    hit: Any,
    seed: int,
) -> dict[str, Any]:
    sample_key = _sample_key(
        source,
        subset,
        role,
        record_id,
        pair_id or "",
        hit.feature,
        hit.subtype,
        hit.start,
        hit.end,
        seed=seed,
    )
    return {
        "label": "",
        "labeler": "",
        "notes": "",
        "source": source,
        "subset": subset,
        "role": role,
        "record_id": record_id,
        "pair_id": pair_id or "",
        "feature": hit.feature,
        "subtype": hit.subtype,
        "start": str(hit.start),
        "end": str(hit.end),
        "hit_text": hit.text,
        "context": hit.context,
        "sample_key": sample_key,
    }


def _bounded_add(
    buckets: dict[str, list[dict[str, Any]]],
    row: dict[str, Any],
    *,
    hits_per_feature: int,
) -> None:
    bucket = buckets[row["feature"]]
    if len(bucket) < hits_per_feature:
        bucket.append(row)
        bucket.sort(key=lambda item: item["sample_key"])
        return
    if row["sample_key"] < bucket[-1]["sample_key"]:
        bucket[-1] = row
        bucket.sort(key=lambda item: item["sample_key"])


def run_sample_hits(args: argparse.Namespace) -> list[dict[str, Any]]:
    SamplingConfig, iter_corpus_records, DEFAULT_ID_FIELDS, iter_tier1_hits = _load_components()
    text_fields = [args.text_field] if args.text_field else None
    id_fields = _id_fields_with_pair_id(DEFAULT_ID_FIELDS)
    sampling = SamplingConfig(cap=args.sample_size, seed=args.seed, max_scan=args.max_scan)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_counts: Counter[str] = Counter()
    doc_count = 0
    measurement_count = 0

    run = init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        tags=["stage1", "precision-validation", "hit-sampling", *args.wandb_tag],
        config={
            "inputs": args.input,
            "split": args.split,
            "hf_config": args.hf_config,
            "sample_size": args.sample_size,
            "hits_per_feature": args.hits_per_feature,
            "seed": args.seed,
            "max_scan": args.max_scan,
            "context_chars": args.context_chars,
            "output": str(args.output),
        },
    )

    try:
        for raw_source in args.input:
            source = _source_from_input(raw_source, split=args.split, hf_config=args.hf_config)
            records_kwargs = {"sampling": sampling, "id_fields": id_fields}
            if text_fields:
                records_kwargs["text_fields"] = text_fields
            records = iter_corpus_records(source, **records_kwargs)
            for record in tqdm(records, desc=f"sample-hits:{source.name}", unit="doc"):
                doc_count += 1
                subset = _subset_for_record(record.metadata, record.split)
                for role, text, pair_id in _measurement_texts(record):
                    measurement_count += 1
                    for hit in iter_tier1_hits(text, context_chars=args.context_chars):
                        candidate_counts[hit.feature] += 1
                        row = _candidate_row(
                            source=record.source,
                            subset=subset,
                            role=role,
                            record_id=record.record_id,
                            pair_id=pair_id,
                            hit=hit,
                            seed=args.seed,
                        )
                        _bounded_add(buckets, row, hits_per_feature=args.hits_per_feature)

        rows = [row for feature in sorted(buckets) for row in buckets[feature]]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        feature_rows = [
            {
                "feature": feature,
                "candidate_hits": candidate_counts[feature],
                "sampled_hits": len(buckets.get(feature, [])),
            }
            for feature in sorted(candidate_counts)
        ]
        run.log(
            {
                "precision/docs": doc_count,
                "precision/measurement_rows": measurement_count,
                "precision/candidate_hits": sum(candidate_counts.values()),
                "precision/sampled_hits": len(rows),
                "precision/features_with_hits": len(feature_rows),
            }
        )
        log_summary_table(run, "precision_feature_counts", feature_rows)
        return rows
    finally:
        run.finish()


def main() -> None:
    args = build_parser().parse_args()
    rows = run_sample_hits(args)
    print(f"Wrote {len(rows)} hit rows to {args.output}")


if __name__ == "__main__":
    main()
