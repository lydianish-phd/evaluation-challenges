#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .constants import CORPORA_CONFIG
from .utils import read_config, read_file, write_json


DEFAULT_CORPORA = ["rocsmt", "pfsmb", "pfsmb-dev"]


def parse_labels(line: str) -> set[str]:
    line = line.strip()
    if not line:
        return set()
    return {x.strip() for x in line.split(",") if x.strip()}

def load_skipped_indices_from_entry(entry: dict) -> set[int]:
    src_path = Path(entry["src_file_path"])
    skipped_file = src_path.with_suffix(src_path.suffix + ".skipped_indices.json")

    if not skipped_file.exists():
        print(f"[WARN] Skipped indices file not found: {skipped_file}")
        return set()

    with open(skipped_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return set(data.get("skipped_indices", []))

def load_corpus(config: dict, corpus: str, subset_config: dict | None = None) -> list[dict]:
    entry = config[corpus]

    subset_entry = subset_config[corpus] if subset_config else None

    skipped_indices = (
        load_skipped_indices_from_entry(subset_entry)
        if subset_entry is not None
        else set()
    )

    src_file = Path(entry["src_file_path"])
    ref_file = Path(entry["ref_file_path"])
    ann_file = Path(entry["ugc_annotations_file_path"])
    norm_file = Path(entry["norm_file_path"])

    src_lines = read_file(src_file)
    ref_lines = read_file(ref_file)
    ann_lines = read_file(ann_file)
    norm_lines = read_file(norm_file)

    if not (len(src_lines) == len(ref_lines) == len(ann_lines) == len(norm_lines)):
        raise ValueError(
            f"Line mismatch for {corpus}: "
            f"src={len(src_lines)}, ref={len(ref_lines)}, "
            f"ann={len(ann_lines)}, norm={len(norm_lines)}"
        )
    
    rows = []
    for i, (src, ref, ann, norm) in enumerate(
        zip(src_lines, ref_lines, ann_lines, norm_lines)
    ):
        if i in skipped_indices:
            continue

        labels = parse_labels(ann)
        if not labels:
            continue

        rows.append({
            "corpus": corpus,
            "sentence_id": i,
            "source": src,
            "norm_source": norm,
            "reference": ref,
            "labels": labels,
        })
    
    print(f"{corpus}: filtered {len(skipped_indices)} skipped indices")

    return rows

def build_label_index(rows: list[dict]) -> dict[str, list[int]]:
    label_to_indices = defaultdict(list)

    for idx, row in enumerate(rows):
        for label in row["labels"]:
            label_to_indices[label].append(idx)

    return label_to_indices


def stratified_sample(
    rows: list[dict],
    total_samples: int,
    seed: int = 13,
    labels: list[str] | None = None,
) -> list[dict]:
    rng = random.Random(seed)

    label_to_indices = build_label_index(rows)

    if labels is None:
        labels = sorted(label_to_indices.keys())

    labels = [label for label in labels if label in label_to_indices]

    if not labels:
        raise ValueError("No labels available for stratified sampling.")

    per_label = total_samples // len(labels)
    remainder = total_samples % len(labels)

    target_per_label = {
        label: per_label + (1 if i < remainder else 0)
        for i, label in enumerate(labels)
    }

    selected_indices = set()
    selected_rows = []

    # First pass: sample target number per label without replacement where possible.
    for label in labels:
        candidates = label_to_indices[label].copy()
        rng.shuffle(candidates)

        chosen_for_label = 0
        for idx in candidates:
            if idx in selected_indices:
                continue

            selected_indices.add(idx)
            selected_rows.append({
                **rows[idx],
                "sampled_for_label": label,
            })
            chosen_for_label += 1

            if chosen_for_label >= target_per_label[label]:
                break

    # Second pass: top up if overlap between labels caused undersampling.
    if len(selected_rows) < total_samples:
        remaining = [i for i in range(len(rows)) if i not in selected_indices]
        rng.shuffle(remaining)

        for idx in remaining:
            selected_indices.add(idx)
            labels_for_row = sorted(rows[idx]["labels"])
            selected_rows.append({
                **rows[idx],
                "sampled_for_label": labels_for_row[0] if labels_for_row else "",
            })

            if len(selected_rows) >= total_samples:
                break

    rng.shuffle(selected_rows)
    return selected_rows[:total_samples]


def write_text_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def write_metadata_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "corpus",
        "sentence_id_0based",
        "sentence_id_1based",
        "sampled_for_label",
        "labels",
        "source",
        "norm_source",
        "reference",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sample_id, row in enumerate(rows):
            writer.writerow({
                "sample_id": sample_id,
                "corpus": row["corpus"],
                "sentence_id_0based": row["sentence_id"],
                "sentence_id_1based": row["sentence_id"] + 1,
                "sampled_for_label": row["sampled_for_label"],
                "labels": ",".join(sorted(row["labels"])),
                "source": row["source"],
                "norm_source": row["norm_source"],
                "reference": row["reference"],
            })


def build_summary(all_rows: list[dict], sampled_rows: list[dict]) -> dict:
    all_label_counts = Counter()
    sampled_label_counts = Counter()
    sampled_for_counts = Counter()
    corpus_counts = Counter()

    for row in all_rows:
        for label in row["labels"]:
            all_label_counts[label] += 1

    for row in sampled_rows:
        corpus_counts[row["corpus"]] += 1
        sampled_for_counts[row["sampled_for_label"]] += 1
        for label in row["labels"]:
            sampled_label_counts[label] += 1

    return {
        "n_available_annotated_sentences": len(all_rows),
        "n_sampled_sentences": len(sampled_rows),
        "corpus_counts": dict(sorted(corpus_counts.items())),
        "available_label_counts": dict(sorted(all_label_counts.items())),
        "sampled_label_counts": dict(sorted(sampled_label_counts.items())),
        "sampled_for_label_counts": dict(sorted(sampled_for_counts.items())),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stratified sampling for human evaluation from UGC-annotated corpora."
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--corpora-config", type=str, default=CORPORA_CONFIG)
    parser.add_argument(
        "--corpora",
        type=str,
        nargs="+",
        default=DEFAULT_CORPORA,
        help="Corpus entries to sample from.",
    )
    parser.add_argument(
        "-n", "--n-samples",
        type=int,
        default=100,
        help="Total number of samples to draw.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="Optional subset of UGC labels to stratify over.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for sampled data. Defaults to <data-dir>/human_eval.",
    )
    parser.add_argument(
        "--subset-data-dir",
        type=str,
        default=None,
        help=(
            "Optional directory containing *.skipped_indices.json files "
            "to exclude LLaMA refusals."
            "If provided, paths from config will be mirrored under this directory."
        ),
    )    
    args = parser.parse_args()

    config = read_config(args.corpora_config, args.data_dir)
    subset_config = (
        read_config(args.corpora_config, args.subset_data_dir)
        if args.subset_data_dir is not None
        else None
    )

    all_rows = []
    for corpus in args.corpora:
        rows = load_corpus(config, corpus, subset_config=subset_config)
        print(f"{corpus}: {len(rows)} annotated sentences (after filtering)")
        all_rows.extend(rows)

    sampled_rows = stratified_sample(
        rows=all_rows,
        total_samples=args.n_samples,
        seed=args.seed,
        labels=args.labels,
    )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        if args.data_dir is None:
            raise ValueError(
                "--data-dir must be provided when --output-dir is not specified."
            )
        output_dir = Path(args.data_dir) / "human_eval" / f"sample_{args.n_samples}_seed{args.seed}_{'_'.join(args.corpora)}"

    output_dir.mkdir(parents=True, exist_ok=True)

    write_text_file(output_dir / "source.txt", [row["source"] for row in sampled_rows])
    write_text_file(output_dir / "normed_source.txt", [row["norm_source"] for row in sampled_rows])
    write_text_file(output_dir / "reference.txt", [row["reference"] for row in sampled_rows])
    write_metadata_csv(output_dir / "metadata.csv", sampled_rows)

    summary = build_summary(all_rows, sampled_rows)
    summary.update({
        "corpora": args.corpora,
        "seed": args.seed,
        "requested_n_samples": args.n_samples,
        "labels_used_for_stratification": args.labels or sorted(summary["available_label_counts"].keys()),
    })
    write_json(output_dir / "sampling_summary.json", summary)

    print(f"\nSaved sampled source to: {output_dir / 'source.txt'}")
    print(f"Saved sampled reference to: {output_dir / 'reference.txt'}")
    print(f"Saved metadata to: {output_dir / 'metadata.csv'}")
    print(f"Saved summary to: {output_dir / 'sampling_summary.json'}")


if __name__ == "__main__":
    main()