#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

from .constants import CORPORA_CONFIG
from .utils import read_config


ANNOTATION_PATTERN = re.compile(
    r"(?P<codes>[0-9]+(?:e[0-9]+)*)-(?P<span>[0-9]+(?::[0-9]+)?)\[[^\]]*\]"
)


def split_annotation_items(line: str) -> list[str]:
    """Split on commas that are outside brackets."""
    items = []
    current = []
    bracket_depth = 0

    for char in line.strip():
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1

        if char == "," and bracket_depth == 0:
            items.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        items.append("".join(current).strip())

    return items


def extract_codes_from_item(item: str) -> set[str]:
    item = item.strip()

    if not item or item == "N":
        return set()

    match = ANNOTATION_PATTERN.fullmatch(item)
    if not match:
        raise ValueError(f"Could not parse annotation item: {item}")

    return set(match.group("codes").split("e"))


def extract_codes_from_line(line: str) -> list[str]:
    codes = set()

    for item in split_annotation_items(line):
        codes.update(extract_codes_from_item(item))

    return sorted(codes, key=lambda x: int(x))


def output_path_for_annotation_file(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_codes{input_path.suffix}")


def process_annotation_file(input_path: str | Path) -> Path:
    input_path = Path(input_path)
    output_path = output_path_for_annotation_file(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_lines = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            try:
                codes = extract_codes_from_line(line)
            except ValueError as exc:
                raise ValueError(f"Error on line {line_id} of {input_path}: {exc}") from exc

            output_lines.append(" ".join(codes))

    with open(output_path, "w", encoding="utf-8") as f:
        for line in output_lines:
            f.write(f"{line}\n")

    return output_path


def get_annotation_files_from_config(
    corpora_config: str,
    data_dir: str | None,
    annotation_corpus: str,
) -> list[Path]:
    config = read_config(corpora_config, data_dir)

    if annotation_corpus not in config:
        raise KeyError(f"Corpus entry not found in config: {annotation_corpus}")

    entry = config[annotation_corpus]

    return [
        Path(entry["src_file_path"]),
        Path(entry["ref_file_path"]),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Extract unique UGC phenomenon codes from PMUMT annotation files."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Parent directory containing files referenced in corpora.yaml.",
    )
    parser.add_argument(
        "--corpora-config",
        type=str,
        default=CORPORA_CONFIG,
        help="Path to corpora config YAML.",
    )
    parser.add_argument(
        "--annotation-corpus",
        type=str,
        default="pmumt-annotations",
        help="Config entry containing annotation file paths.",
    )
    parser.add_argument(
        "--annotation-files",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional explicit annotation files. "
            "If omitted, files are read from --annotation-corpus src/ref paths."
        ),
    )
    args = parser.parse_args()

    if args.annotation_files is not None:
        annotation_files = [Path(path) for path in args.annotation_files]
    else:
        annotation_files = get_annotation_files_from_config(
            corpora_config=args.corpora_config,
            data_dir=args.data_dir,
            annotation_corpus=args.annotation_corpus,
        )

    for annotation_file in annotation_files:
        output_path = process_annotation_file(annotation_file)
        print(f"Saved extracted codes to: {output_path}")


if __name__ == "__main__":
    main()