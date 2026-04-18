import os, argparse
import pandas as pd
from utils import (
    read_json,
    LLAMA,
    GEMMA,
    NLLB,
    TOWER,
    CORPORA
)


def flatten_scores_ci(scores_ci, prefix=""):
    flat = {}

    for key, value in scores_ci.items():
        new_key = f"{prefix}_{key}" if prefix else key

        if isinstance(value, dict):
            flat.update(flatten_scores_ci(value, new_key))
        else:
            flat[new_key] = value

    return flat

def scale_comet_scores(scores: dict) -> dict:
    """
    Scale top-level COMET / COMET-Kiwi scores by 100.
    """
    scaled = scores.copy()

    for key in ["comet", "cometkiwi"]:
        if key in scaled:
            scaled[key] = scaled[key] * 100

    return scaled

def scale_comet_scores_ci(scores_ci: dict) -> dict:
    """
    Scale COMET / COMET-Kiwi score-like fields by 100
    before flattening.
    """
    score_fields = {
        "baseline_mean",
        "system_mean",
        "delta_mean",
        "delta_std",
        "delta_ci_low",
        "delta_ci_high",
        "delta_median",
    }

    scaled = {}

    for metric, values in scores_ci.items():
        if not isinstance(values, dict):
            scaled[metric] = values
            continue

        scaled_metric = {}
        for key, val in values.items():
            if metric in {"comet", "cometkiwi"} and key in score_fields:
                scaled_metric[key] = val * 100
            else:
                scaled_metric[key] = val

        scaled[metric] = scaled_metric

    return scaled


def aggregate_scores(input_dir, corpus, models):
    all_scores = []

    for model in models:
        model_output_dir = os.path.join(input_dir, "outputs", model, corpus)
        if os.path.isdir(model_output_dir):
            scores_files = [
                f.path for f in os.scandir(model_output_dir)
                if f.name.endswith("postproc.scores.json")
            ]

            for score_file in scores_files:
                scores = {
                    "model": model,
                    "file": os.path.basename(score_file).removesuffix(".scores.json"),
                }

                scores.update(read_json(score_file))
                scores = scale_comet_scores(scores)

                count_file = score_file.replace(".scores.json", ".counts.json")
                if os.path.exists(count_file):
                    counts = read_json(count_file)

                    # remove any keys where the type is not int (some are lists)
                    counts = {k: v for k, v in counts.items() if isinstance(v, int)}
                    scores.update(counts)

                scores_ci_file = score_file.replace(".scores.json", ".scores_ci.json")
                if os.path.exists(scores_ci_file):
                    scores_ci = read_json(scores_ci_file)
                    scores_ci = scale_comet_scores_ci(scores_ci)
                    flat_scores_ci = flatten_scores_ci(scores_ci)
                    scores.update(flat_scores_ci)

                all_scores.append(scores)

    return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to experiment directory", type=str)
    parser.add_argument("-c", "--corpora", type=str, nargs="+", default=CORPORA)
    parser.add_argument("-m", "--models", type=str, nargs="+", default=[NLLB, LLAMA, GEMMA, TOWER])
    args = parser.parse_args()

    scores_dir = os.path.join(args.input_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)

    print("Aggregating scores for:")
    for corpus in args.corpora:
        print(f" - {corpus}")
        scores = aggregate_scores(args.input_dir, corpus, args.models)
        scores_file = os.path.join(scores_dir, f"scores_ci_{corpus}.csv")
        scores_df = pd.DataFrame(scores)
        scores_df.to_csv(scores_file, index=False)