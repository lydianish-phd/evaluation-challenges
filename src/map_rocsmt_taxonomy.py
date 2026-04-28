#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import Counter

from .constants import CORPORA_CONFIG
from .utils import read_config, read_file, write_json
from .taxonomies import ROCSMT_TO_UGC_TAXONOMY


def parse_rocs_line(line: str) -> set[str]:
    line = line.strip()
    if not line:
        return set()
    return {x.strip() for x in line.split(",") if x.strip()}


def map_rocs_to_ugc(rocs_labels: set[str]) -> set[str]:
    ugc = set()
    for label in rocs_labels:
        mapped = ROCSMT_TO_UGC_TAXONOMY.get(label, [])
        ugc.update(mapped)
    return ugc


def build_summary(annotations: list[set[str]]) -> dict:
    total = len(annotations)
    annotated = sum(1 for a in annotations if a)

    label_counts = Counter()
    for labels in annotations:
        for label in labels:
            label_counts[label] += 1

    return {
        "total_sentences": total,
        "annotated_sentences": annotated,
        "unannotated_sentences": total - annotated,
        "annotation_ratio": annotated / total if total else 0.0,
        "labels": sorted(label_counts.keys()),
        "label_counts": dict(sorted(label_counts.items())),
    }


def write_annotation_file(path: Path, annotations: list[set[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for labels in annotations:
            f.write(",".join(sorted(labels)) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Map RoCS-MT annotations to UGC taxonomy."
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--corpora-config", type=str, default=CORPORA_CONFIG)
    parser.add_argument("--rocsmt-corpus", type=str, default="rocsmt")

    parser.add_argument(
        "--annotations-file",
        type=str,
        default=None,
        help="rocsmt_annotations.txt (sentence-level RoCS-MT labels)",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output mapped UGC file",
    )

    args = parser.parse_args()

    config = read_config(args.corpora_config, args.data_dir)

    src_file = Path(config[args.rocsmt_corpus]["src_file_path"])
    rocs_file = (
        Path(args.annotations_file)
        if args.annotations_file
        else src_file.parent / "rocsmt_annotations.txt"
    )

    output_file = (
        Path(args.output_file)
        if args.output_file
        else src_file.parent / "rocsmt_ugc_annotations.txt"
    )

    summary_file = output_file.with_name(output_file.stem + "_summary.json")

    print(f"Source file: {src_file}")
    print(f"RoCS annotations: {rocs_file}")

    src_lines = read_file(src_file)
    rocs_lines = read_file(rocs_file)

    if len(src_lines) != len(rocs_lines):
        raise ValueError(
            f"Mismatch: {len(src_lines)} source lines vs {len(rocs_lines)} annotation lines"
        )

    ugc_annotations = []

    for line in rocs_lines:
        rocs_labels = parse_rocs_line(line)
        ugc_labels = map_rocs_to_ugc(rocs_labels)
        ugc_annotations.append(ugc_labels)

    write_annotation_file(output_file, ugc_annotations)

    summary = build_summary(ugc_annotations)
    write_json(summary_file, summary)

    print(f"Saved UGC annotations to: {output_file}")
    print(f"Saved summary to: {summary_file}")
    print(
        f"Annotated sentences: {summary['annotated_sentences']}/{summary['total_sentences']}"
    )
    print(f"Labels: {', '.join(summary['labels'])}")


if __name__ == "__main__":
    main()