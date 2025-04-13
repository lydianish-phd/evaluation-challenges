import os, argparse, json
import pandas as pd
from utils import read_json

def aggregate_scores(input_dir, corpus, models):
    all_scores = []

    for model in models:
        model_output_dir = os.path.join(input_dir, "outputs", model, corpus)
        if os.path.isdir(model_output_dir):
            scores_files = [f.path for f in os.scandir(model_output_dir) if f.name.endswith(".scores.json")]
            for score_file in scores_files:
                scores = {
                    "model": model,
                    "file": os.path.basename(score_file).removesuffix(".scores.json"),
                }

                scores.update(read_json(score_file))
                
                scores["comet"] *= 100
                if "xcomet" in scores:
                    scores["xcomet"] *= 100

                count_file = score_file.replace(".scores.json", ".counts.json")                
                if os.path.exists(count_file):
                    counts = read_json(count_file)
                
                    # remove any keys where the type is not int (some are lists)
                    counts = {k: v for k, v in counts.items() if isinstance(v, int)}
                    scores.update(counts)
                
                all_scores.append(scores)
    
    return all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to experiment directory", type=str)
    parser.add_argument("-c", "--corpora", type=str, nargs="+")
    parser.add_argument("-m", "--models", type=str, nargs="+")
    args = parser.parse_args()

    scores_dir = os.path.join(args.input_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)

    print(f"Aggregating scores for:")
    for corpus in args.corpora:
        print(f" - {corpus}")
        scores = aggregate_scores(args.input_dir, corpus, args.models)
        scores_file = os.path.join(scores_dir, f"scores_{corpus}.csv")
        scores_df = pd.DataFrame(scores)
        scores_df.to_csv(scores_file, index=False)