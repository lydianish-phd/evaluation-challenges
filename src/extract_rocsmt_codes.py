#!/usr/bin/env python3
import argparse
import csv, re
from collections import defaultdict
from pathlib import Path

from .constants import CORPORA_CONFIG
from .utils import read_config, read_file, write_json


def split_manual_field(value: str) -> set[str]:
    """
    Parse the RoCS-MT manual annotation field.

    Handles empty fields and possible multi-label fields.
    """
    value = value.strip()
    if not value:
        return set()

    # Split on multiple separators at once
    parts = re.split(r"[;,|]", value)

    return {p.strip() for p in parts if p.strip()}


def extract_rocsmt_line_annotations(tsv_path: str | Path, n_lines: int | None = None) -> list[set[str]]:
    """
    Returns one set of RoCS-MT annotation labels per sentence ID.
    """
    sent_annotations = defaultdict(set)
    max_sent_id = -1

    with open(tsv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        required_cols = {"sentid", "manual"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {tsv_path}: {missing}")

        for row in reader:
            sent_id = int(row["sentid"])
            max_sent_id = max(max_sent_id, sent_id)

            labels = split_manual_field(row["manual"])
            sent_annotations[sent_id].update(labels)

    if n_lines is None:
        n_lines = max_sent_id + 1

    annotations = []
    for sent_id in range(n_lines):
        annotations.append(sent_annotations.get(sent_id, set()))

    return annotations


def write_annotation_file(path: str | Path, annotations: list[set[str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for labels in annotations:
            f.write(",".join(sorted(labels)) + "\n")

def build_summary(annotations: list[set[str]]) -> dict:
    n_total = len(annotations)
    n_annotated = sum(1 for labels in annotations if labels)
    all_labels = sorted({label for labels in annotations for label in labels})

    return {
        "total_sentences": n_total,
        "annotated_sentences": n_annotated,
        "annotation_ratio": n_annotated / n_total if n_total > 0 else 0.0,
        "labels": all_labels,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Extract sentence-level RoCS-MT UGC annotations from RoCS-annotated.tsv."
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--corpora-config", type=str, default=CORPORA_CONFIG)
    parser.add_argument(
        "--rocsmt-corpus",
        type=str,
        default="rocsmt",
        help="Config entry for the RoCS-MT corpus source file.",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default=None,
        help="Path to RoCS-annotated.tsv. Defaults to RoCS-annotated.tsv next to the RoCS-MT source file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output sentence-level annotation file. Defaults to rocsmt_annotations.txt next to the source file.",
    )
    args = parser.parse_args()

    config = read_config(args.corpora_config, args.data_dir)

    src_file = Path(config[args.rocsmt_corpus]["src_file_path"])
    n_lines = len(read_file(src_file))

    annotation_file = (
        Path(args.annotation_file)
        if args.annotation_file is not None
        else src_file.parent / "RoCS-annotated.tsv"
    )

    output_file = (
        Path(args.output_file)
        if args.output_file is not None
        else src_file.parent / "rocsmt_annotations.txt"
    )

    summary_file = output_file.with_name(output_file.stem + "_summary.json")

    print(f"RoCS-MT source file: {src_file}")
    print(f"RoCS-MT annotation TSV: {annotation_file}")
    print(f"Number of source lines: {n_lines}")

    annotations = extract_rocsmt_line_annotations(
        tsv_path=annotation_file,
        n_lines=n_lines,
    )

    write_annotation_file(output_file, annotations)

    summary = build_summary(annotations)
    write_json(summary_file, summary)

    print(f"Saved sentence-level annotations to: {output_file}")
    print(f"Saved summary to: {summary_file}")
    print(f"Annotated sentences: {summary['annotated_sentences']}/{summary['total_sentences']}")
    print(f"Labels found: {', '.join(summary['labels'])}")


if __name__ == "__main__":
    main()