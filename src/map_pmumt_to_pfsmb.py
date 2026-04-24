#!/usr/bin/env python3
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import csv

from .utils import read_file, read_config, write_json
from .constants import (
    CORPORA_CONFIG,
    PMUMT,
    PFSMB,
)


def normalise(text):
    return " ".join(text.strip().split())


def build_index(lines):
    index = defaultdict(list)
    for i, line in enumerate(lines):
        index[normalise(line)].append(i)
    return index


def find_matches(pmumt_lines, dev_lines, test_lines):
    dev_index = build_index(dev_lines)
    test_index = build_index(test_lines)

    rows = []

    for pmumt_idx, line in enumerate(pmumt_lines):
        key = normalise(line)

        dev_matches = dev_index.get(key, [])
        test_matches = test_index.get(key, [])

        if dev_matches and test_matches:
            source = "both"
        elif dev_matches:
            source = "dev"
        elif test_matches:
            source = "test"
        else:
            source = "unmatched"

        rows.append({
            "pmumt_line_id_0based": pmumt_idx,
            "pmumt_line_id_1based": pmumt_idx + 1,
            "source": source,
            "pfsmb_dev_line_ids_1based": ";".join(str(i + 1) for i in dev_matches),
            "pfsmb_test_line_ids_1based": ";".join(str(i + 1) for i in test_matches),
            "text": line,
        })

    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pmumt_line_id_0based",
        "pmumt_line_id_1based",
        "source",
        "pfsmb_dev_line_ids_1based",
        "pfsmb_test_line_ids_1based",
        "text",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def build_summary(rows):
    counts = Counter(row["source"] for row in rows)

    return {
        "total_pmumt_lines": len(rows),
        "from_pfsmb_dev": counts["dev"],
        "from_pfsmb_test": counts["test"],
        "from_both_duplicates": counts["both"],
        "unmatched": counts["unmatched"],
    }

def print_summary(rows):
    counts = Counter(row["source"] for row in rows)

    print("\nSummary")
    print("-------")
    print(f"Total PMUMT lines: {len(rows)}")
    print(f"From PFSMB dev: {counts['dev']}")
    print(f"From PFSMB test: {counts['test']}")
    print(f"From both (duplicates): {counts['both']}")
    print(f"Unmatched: {counts['unmatched']}")


def main():
    parser = argparse.ArgumentParser(
        description="Map PMUMT lines to PFSMB dev/test indices using config."
    )

    parser.add_argument("--data-dir", type=str, help="Optional data directory override")
    parser.add_argument("--corpora-config", type=str, default=CORPORA_CONFIG)

    # Corpus names (from config)
    parser.add_argument("--pmumt-corpus", type=str, default=PMUMT)
    parser.add_argument("--pfsmb-dev-corpus", type=str, default=f"{PFSMB}-dev")
    parser.add_argument("--pfsmb-test-corpus", type=str, default=PFSMB)

    # Optional direct file overrides
    parser.add_argument("--pmumt-file", type=str)
    parser.add_argument("--pfsmb-dev-file", type=str)
    parser.add_argument("--pfsmb-test-file", type=str)

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where mapping CSV and summary JSON will be saved. Defaults to PMUMT file directory.",
    )

    args = parser.parse_args()

    # Load config
    config = read_config(args.corpora_config, args.data_dir)

    
    # Resolve paths (priority: CLI override > config)
    pmumt_file = Path(args.pmumt_file or config[args.pmumt_corpus]["src_file_path"])
    dev_file = Path(args.pfsmb_dev_file or config[args.pfsmb_dev_corpus]["src_file_path"])
    test_file = Path(args.pfsmb_test_file or config[args.pfsmb_test_corpus]["src_file_path"])

    # Resolve output path
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = pmumt_file.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_csv = output_dir / "pmumt_pfsmb_mapping.csv"
    summary_json = output_dir / "pmumt_pfsmb_mapping_summary.json"


    print(f"PMUMT file: {pmumt_file}")
    print(f"PFSMB dev file: {dev_file}")
    print(f"PFSMB test file: {test_file}")

    # Read files
    pmumt_lines = read_file(pmumt_file)
    dev_lines = read_file(dev_file)
    test_lines = read_file(test_file)

    # Match
    rows = find_matches(pmumt_lines, dev_lines, test_lines)

    # Save
    write_csv(mapping_csv, rows)

    summary = build_summary(rows)
    write_json(summary_json, summary)

    # Print summary to console
    print_summary(rows)
    print(f"\nSaved mapping to: {mapping_csv}")
    print(f"Saved summary to: {summary_json}")


if __name__ == "__main__":
    main()