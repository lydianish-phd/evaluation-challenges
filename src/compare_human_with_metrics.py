#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

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


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metric_preference(default_score: float, guided_score: float, eps: float = 1e-12) -> str:
    delta = guided_score - default_score

    if delta > eps:
        return "guided"
    if delta < -eps:
        return "default"
    return "tie"


def get_metric_files(
    model_outputs_dir: Path,
    model: str,
    corpus: str,
    guided_guideline: str,
) -> tuple[Path, Path]:
    default_file = model_outputs_dir / model / corpus / "default.comet.json"
    guided_file = model_outputs_dir / model / corpus / f"{guided_guideline}.comet.json"
    return default_file, guided_file


def add_metric_preferences(
    majority_rows: list[dict],
    model_outputs_dir: Path,
    metrics: list[str],
) -> list[dict]:
    cache = {}
    output_rows = []

    for row in majority_rows:
        model = row["model"]
        corpus = row["corpus"]
        sent_id = int(row["sentence_id_0based"])

        # If your key has guided_guideline, add it to majority vote script too.
        guided_guideline = row.get("guided_guideline", "")
        if not guided_guideline:
            # fallback: corpus-specific common case
            guided_guideline = corpus.replace("-dev", "")

        cache_key = (model, corpus, guided_guideline)

        if cache_key not in cache:
            default_file, guided_file = get_metric_files(
                model_outputs_dir=model_outputs_dir,
                model=model,
                corpus=corpus,
                guided_guideline=guided_guideline,
            )

            if not default_file.exists():
                raise FileNotFoundError(f"Missing metric file: {default_file}")
            if not guided_file.exists():
                raise FileNotFoundError(f"Missing metric file: {guided_file}")

            cache[cache_key] = {
                "default": read_json(default_file),
                "guided": read_json(guided_file),
            }

        metric_data = cache[cache_key]

        out = dict(row)

        for metric in metrics:
            default_score = metric_data["default"][metric][sent_id]
            guided_score = metric_data["guided"][metric][sent_id]
            delta = guided_score - default_score

            out[f"{metric}_default"] = default_score
            out[f"{metric}_guided"] = guided_score
            out[f"{metric}_delta"] = delta
            out[f"{metric}_pref"] = metric_preference(default_score, guided_score)

        output_rows.append(out)

    return output_rows


def compute_metric_agreement(rows: list[dict], metrics: list[str]) -> list[dict]:
    results = []

    for human_col in ["majority_overall_pref", "majority_guideline_pref"]:
        for metric in metrics:
            y_human = []
            y_metric = []

            for row in rows:
                human = row.get(human_col, "")
                auto = row.get(f"{metric}_pref", "")

                if human not in VALID_LABELS or auto not in VALID_LABELS:
                    continue

                y_human.append(human)
                y_metric.append(auto)

            if not y_human:
                kappa = float("nan")
                agreement = float("nan")
            else:
                kappa = cohen_kappa_score(y_human, y_metric)
                agreement = sum(h == a for h, a in zip(y_human, y_metric)) / len(y_human)

            results.append({
                "human_question": human_col,
                "metric": metric,
                "n_items": len(y_human),
                "cohen_kappa": kappa,
                "percent_agreement": agreement,
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-eval-dir", type=str, required=True)
    parser.add_argument("--model-outputs-dir", type=str, required=True)
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["comet", "cometkiwi"],
    )
    args = parser.parse_args()

    human_eval_dir = Path(args.human_eval_dir)
    model_outputs_dir = Path(args.model_outputs_dir)

    majority_file = human_eval_dir / "human_majority_votes.csv"
    majority_rows = read_csv(majority_file)

    rows_with_metrics = add_metric_preferences(
        majority_rows=majority_rows,
        model_outputs_dir=model_outputs_dir,
        metrics=args.metrics,
    )

    metric_rows_path = human_eval_dir / "human_majority_with_metrics.csv"
    fieldnames = list(rows_with_metrics[0].keys()) if rows_with_metrics else []
    write_csv(metric_rows_path, rows_with_metrics, fieldnames)

    agreement_rows = compute_metric_agreement(rows_with_metrics, args.metrics)

    agreement_path = human_eval_dir / "human_metric_agreement.csv"
    write_csv(
        agreement_path,
        agreement_rows,
        [
            "human_question",
            "metric",
            "n_items",
            "cohen_kappa",
            "percent_agreement",
        ],
    )

    print(f"Saved majority + metrics to: {metric_rows_path}")
    print(f"Saved human/metric agreement to: {agreement_path}")


if __name__ == "__main__":
    main()