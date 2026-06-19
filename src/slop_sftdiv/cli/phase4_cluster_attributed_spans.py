from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
from typing import Any, cast

import torch
from transformers import AutoModel, AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed detector-attributed spans and cluster them with HDBSCAN."
    )
    parser.add_argument("--span-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-model", default="thenlper/gte-large")
    parser.add_argument("--max-spans", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=64)
    return parser


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    return float(value)


def _int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if value in (None, ""):
        return 0
    return int(float(value))


def _select_rows(rows: list[dict[str, Any]], *, max_spans: int) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if str(row.get("span", "")).strip()
        and len(str(row.get("span", "")).strip()) >= 3
        and any(char.isalpha() for char in str(row.get("span", "")))
    ]
    selected.sort(
        key=lambda row: (
            _int(row, "doc_count"),
            _float(row, "mean_attribution_score"),
            _float(row, "mean_llm_probability"),
        ),
        reverse=True,
    )
    return selected[:max_spans]


@torch.inference_mode()
def _embed_spans(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = cast(Any, AutoTokenizer.from_pretrained(model_name))
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    embeddings = []
    spans = [str(row["span"]) for row in rows]
    for offset in range(0, len(spans), batch_size):
        batch_spans = spans[offset : offset + batch_size]
        encoded = tokenizer(
            batch_spans,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(**encoded)
        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
        embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0)


def _cluster(embeddings: torch.Tensor, *, min_cluster_size: int, min_samples: int) -> list[int]:
    hdbscan = importlib.import_module("hdbscan")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings.numpy())
    return [int(label) for label in labels]


def _member_rows(rows: list[dict[str, Any]], labels: list[int]) -> list[dict[str, Any]]:
    out = []
    for row, label in zip(rows, labels):
        enriched = dict(row)
        enriched["hdbscan_cluster"] = label
        out.append(enriched)
    return sorted(out, key=lambda row: (int(row["hdbscan_cluster"]), -_int(row, "doc_count"), -_float(row, "mean_attribution_score")))


def _cluster_rows(member_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in member_rows:
        grouped.setdefault(int(row["hdbscan_cluster"]), []).append(row)
    out = []
    for label, rows in sorted(grouped.items()):
        if label < 0:
            cluster_name = "noise"
        else:
            cluster_name = f"cluster_{label}"
        rows = sorted(rows, key=lambda row: (_int(row, "doc_count"), _float(row, "mean_attribution_score")), reverse=True)
        out.append(
            {
                "hdbscan_cluster": label,
                "cluster_name": cluster_name,
                "span_count": len(rows),
                "total_doc_count": sum(_int(row, "doc_count") for row in rows),
                "top_spans": "; ".join(str(row["span"]) for row in rows[:8]),
                "mean_llm_probability": sum(_float(row, "mean_llm_probability") for row in rows)
                / max(1, len(rows)),
                "mean_attribution_score": sum(_float(row, "mean_attribution_score") for row in rows)
                / max(1, len(rows)),
            }
        )
    return sorted(out, key=lambda row: (int(row["hdbscan_cluster"] < 0), -int(row["total_doc_count"])))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "hdbscan_cluster",
        "cluster_name",
        "span",
        "span_count",
        "doc_count",
        "total_doc_count",
        "mean_attribution_score",
        "mean_llm_probability",
        "top_spans",
    ]
    columns = [column for column in preferred if column in columns] + [
        column for column in columns if column not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _select_rows(_read_csv(args.span_summary), max_spans=args.max_spans)
    embeddings = _embed_spans(
        rows,
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    labels = _cluster(
        embeddings,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    members = _member_rows(rows, labels)
    clusters = _cluster_rows(members)
    output_dir: Path = args.output_dir
    _write_csv(output_dir / "phase4_detector_ig_hdbscan_members.csv", members)
    _write_csv(output_dir / "phase4_detector_ig_hdbscan_clusters.csv", clusters)
    summary = {
        "span_summary": str(args.span_summary),
        "embedding_model": args.embedding_model,
        "input_spans": len(rows),
        "clusters_including_noise": len(clusters),
        "non_noise_clusters": sum(1 for row in clusters if int(row["hdbscan_cluster"]) >= 0),
        "noise_spans": sum(1 for row in members if int(row["hdbscan_cluster"]) < 0),
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
    }
    _write_json(output_dir / "phase4_detector_ig_hdbscan_summary.json", summary)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(
        "Wrote Phase 4 HDBSCAN span clusters to "
        f"{args.output_dir} ({summary['input_spans']} spans, "
        f"{summary['non_noise_clusters']} non-noise clusters)"
    )


if __name__ == "__main__":
    main()
