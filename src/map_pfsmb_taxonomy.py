#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from .constants import CORPORA_CONFIG
from .taxonomies import PMUMT_TO_UGC_TAXONOMY
from .utils import read_config, read_file


def read_csv_rows(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_code_line(line: str) -> set[str]:
    """
    Parse one line from annotations_codes.fr / annotations_codes.en.
    Expected format: space-separated PMUMT codes, e.g.:
      1 4 10
    Empty line means no code.
    """
    line = line.strip()
    if not line:
        return set()
    return set(line.split())


def map_pmumt_codes_to_ugc(pmumt_codes: set[str]) -> set[str]:
    ugc_codes = set()
    for code in pmumt_codes:
        mapped = PMUMT_TO_UGC_TAXONOMY.get(code, [])
        ugc_codes.update(mapped)
    return ugc_codes


def load_pmumt_code_lines(code_file: str | Path) -> list[set[str]]:
    return [parse_code_line(line) for line in read_file(code_file)]


def initialise_empty_annotations(n_lines: int) -> list[set[str]]:
    return [set() for _ in range(n_lines)]


def add_codes(target: list[set[str]], indices_1based: str, ugc_codes: set[str]) -> None:
    """
    Add UGC codes to one or more 1-based line IDs.
    indices_1based can be empty or semicolon-separated, e.g. "4;12".
    """
    if not indices_1based:
        return

    for idx_str in indices_1based.split(";"):
        if not idx_str.strip():
            continue
        idx = int(idx_str) - 1
        target[idx].update(ugc_codes)


def write_annotation_file(path: str | Path, annotations: list[set[str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for codes in annotations:
            f.write(",".join(sorted(codes)) + "\n")


def build_pfsmb_ugc_annotations(
    mapping_csv: str | Path,
    pmumt_code_file: str | Path,
    pfsmb_dev_file: str | Path,
    pfsmb_test_file: str | Path,
    output_dir: str | Path,
    output_prefix: str = "ugc_annotations",
) -> tuple[Path, Path]:
    mapping_rows = read_csv_rows(mapping_csv)
    pmumt_code_lines = load_pmumt_code_lines(pmumt_code_file)

    dev_lines = read_file(pfsmb_dev_file)
    test_lines = read_file(pfsmb_test_file)

    dev_annotations = initialise_empty_annotations(len(dev_lines))
    test_annotations = initialise_empty_annotations(len(test_lines))

    for row in mapping_rows:
        pmumt_idx = int(row["pmumt_line_id_1based"]) - 1

        if pmumt_idx >= len(pmumt_code_lines):
            raise IndexError(
                f"PMUMT line {pmumt_idx + 1} exists in mapping CSV but not in {pmumt_code_file}"
            )

        pmumt_codes = pmumt_code_lines[pmumt_idx]
        ugc_codes = map_pmumt_codes_to_ugc(pmumt_codes)

        if not ugc_codes:
            continue

        add_codes(
            target=dev_annotations,
            indices_1based=row.get("pfsmb_dev_line_ids_1based", ""),
            ugc_codes=ugc_codes,
        )
        add_codes(
            target=test_annotations,
            indices_1based=row.get("pfsmb_test_line_ids_1based", ""),
            ugc_codes=ugc_codes,
        )

    output_dir = Path(output_dir)
    dev_output = output_dir / f"{output_prefix}.dev"
    test_output = output_dir / f"{output_prefix}.test"

    write_annotation_file(dev_output, dev_annotations)
    write_annotation_file(test_output, test_annotations)

    return dev_output, test_output


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Map PMUMT specificity annotations to the project UGC taxonomy and "
            "write PFSMB dev/test line-level annotation files."
        )
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--corpora-config", type=str, default=CORPORA_CONFIG)

    parser.add_argument("--pmumt-annotations-corpus", type=str, default="pmumt-annotations")
    parser.add_argument("--pfsmb-dev-corpus", type=str, default="pfsmb-dev")
    parser.add_argument("--pfsmb-test-corpus", type=str, default="pfsmb")

    parser.add_argument(
        "--mapping-csv",
        type=str,
        default=None,
        help=(
            "CSV produced by map_pmumt_to_pfsmb.py. "
            "Defaults to pmumt_pfsmb_mapping.csv in the PMUMT annotations directory."
        ),
    )
    parser.add_argument(
        "--pmumt-code-file",
        type=str,
        default=None,
        help=(
            "File produced by extract_pmumt_codes.py, e.g. annotations_codes.fr. "
            "Defaults to annotations_codes.fr next to annotations.fr."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory where mapped PFSMB annotation files are saved. "
            "Defaults to the PFSMB data directory inferred from dev file."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="ugc_annotations",
        help="Prefix for output files. Produces <prefix>.dev and <prefix>.test.",
    )

    args = parser.parse_args()

    config = read_config(args.corpora_config, args.data_dir)

    pmumt_ann_fr = Path(config[args.pmumt_annotations_corpus]["src_file_path"])
    pfsmb_dev_file = Path(config[args.pfsmb_dev_corpus]["src_file_path"])
    pfsmb_test_file = Path(config[args.pfsmb_test_corpus]["src_file_path"])

    mapping_csv = (
        Path(args.mapping_csv)
        if args.mapping_csv is not None
        else pmumt_ann_fr.parent / "pmumt_pfsmb_mapping.csv"
    )

    pmumt_code_file = (
        Path(args.pmumt_code_file)
        if args.pmumt_code_file is not None
        else pmumt_ann_fr.with_name(f"{pmumt_ann_fr.stem}_codes{pmumt_ann_fr.suffix}")
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else pfsmb_dev_file.parent
    )

    print(f"Mapping CSV: {mapping_csv}")
    print(f"PMUMT code file: {pmumt_code_file}")
    print(f"PFSMB dev file: {pfsmb_dev_file}")
    print(f"PFSMB test file: {pfsmb_test_file}")
    print(f"Output dir: {output_dir}")

    dev_output, test_output = build_pfsmb_ugc_annotations(
        mapping_csv=mapping_csv,
        pmumt_code_file=pmumt_code_file,
        pfsmb_dev_file=pfsmb_dev_file,
        pfsmb_test_file=pfsmb_test_file,
        output_dir=output_dir,
        output_prefix=args.output_prefix,
    )

    print(f"Saved dev annotations to: {dev_output}")
    print(f"Saved test annotations to: {test_output}")


if __name__ == "__main__":
    main()