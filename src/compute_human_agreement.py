#!/usr/bin/env python3
import argparse
import csv
import itertools
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.metrics import cohen_kappa_score


VALID_LABELS = {"default", "guided", "tie"}


def read_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def average_pairwise_percent_agreement(item_to_labels: dict[int, dict[str, str]]) -> float:
    scores = []

    for labels_by_ann in item_to_labels.values():
        labels = list(labels_by_ann.values())
        for a, b in itertools.combinations(labels, 2):
            scores.append(float(a == b))

    return sum(scores) / len(scores) if scores else 0.0


def majority_vote(labels: list[str]) -> str:
    labels = [x for x in labels if x in VALID_LABELS]
    if not labels:
        return ""

    counts = Counter(labels)
    top = counts.most_common()

    if len(top) > 1 and top[0][1] == top[1][1]:
        return "tie"
    return top[0][0]


def compute_pairwise_kappas(rows: list[dict], pref_col: str) -> list[dict]:
    annotators = sorted({r["annotator"] for r in rows})

    by_item_ann = defaultdict(dict)
    for r in rows:
        label = r[pref_col]
        if label not in VALID_LABELS:
            continue
        by_item_ann[int(r["item_id"])][r["annotator"]] = label

    results = []

    for ann1, ann2 in itertools.combinations(annotators, 2):
        y1, y2 = [], []

        for item_id, labels in by_item_ann.items():
            if ann1 in labels and ann2 in labels:
                y1.append(labels[ann1])
                y2.append(labels[ann2])

        kappa = cohen_kappa_score(y1, y2) if y1 else float("nan")
        agreement = sum(a == b for a, b in zip(y1, y2)) / len(y1) if y1 else float("nan")

        results.append({
            "question": pref_col,
            "annotator_1": ann1,
            "annotator_2": ann2,
            "n_items": len(y1),
            "cohen_kappa": kappa,
            "percent_agreement": agreement,
        })

    return results


def compute_majority_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    meta_by_item = {}

    for r in rows:
        item_id = int(r["item_id"])
        grouped[item_id].append(r)
        meta_by_item[item_id] = r

    majority_rows = []

    for item_id, item_rows in sorted(grouped.items()):
        meta = meta_by_item[item_id]

        overall_labels = [r["overall_pref"] for r in item_rows]
        guideline_labels = [r["guideline_pref"] for r in item_rows]

        majority_rows.append({
            "item_id": item_id,
            "sample_id": meta["sample_id"],
            "corpus": meta["corpus"],
            "sentence_id_0based": meta["sentence_id_0based"],
            "model": meta["model"],
            "labels": meta.get("labels", ""),
            "majority_overall_pref": majority_vote(overall_labels),
            "majority_guideline_pref": majority_vote(guideline_labels),
            "n_annotators": len(item_rows),
            "n_valid_overall": sum(x in VALID_LABELS for x in overall_labels),
            "n_valid_guideline": sum(x in VALID_LABELS for x in guideline_labels),
        })

    return majority_rows


def compute_agreement(input_csv: Path, output_dir: Path) -> None:
    rows = read_csv(input_csv)

    pairwise_rows = []
    pairwise_rows.extend(compute_pairwise_kappas(rows, "overall_pref"))
    pairwise_rows.extend(compute_pairwise_kappas(rows, "guideline_pref"))

    write_csv(
        output_dir / "human_pairwise_agreement.csv",
        pairwise_rows,
        [
            "question",
            "annotator_1",
            "annotator_2",
            "n_items",
            "cohen_kappa",
            "percent_agreement",
        ],
    )

    majority_rows = compute_majority_rows(rows)

    write_csv(
        output_dir / "human_majority_votes.csv",
        majority_rows,
        [
            "item_id",
            "sample_id",
            "corpus",
            "sentence_id_0based",
            "model",
            "labels",
            "majority_overall_pref",
            "majority_guideline_pref",
            "n_annotators",
            "n_valid_overall",
            "n_valid_guideline",
        ],
    )

    print(f"Saved pairwise agreement to: {output_dir / 'human_pairwise_agreement.csv'}")
    print(f"Saved majority votes to: {output_dir / 'human_majority_votes.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-eval-dir", type=str, required=True)
    parser.add_argument("--input-csv", type=str, default=None)
    args = parser.parse_args()

    human_eval_dir = Path(args.human_eval_dir)
    input_csv = (
        Path(args.input_csv)
        if args.input_csv
        else human_eval_dir / "human_annotations_long.csv"
    )

    compute_agreement(input_csv=input_csv, output_dir=human_eval_dir)


if __name__ == "__main__":
    main()