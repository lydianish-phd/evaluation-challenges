#!/usr/bin/env python3
import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sacrebleu.metrics import BLEU

try:
    from scipy.stats import ttest_rel
except Exception:  # pragma: no cover
    ttest_rel = None


TOWER = "Unbabel/TowerInstruct-7B-v0.2"
LLAMA = "meta-llama/Llama-3.1-8B-Instruct"
GEMMA = "google/gemma-2-9b-it"
NLLB = "facebook/nllb-200-3.3B"
CORPORA = ["rocsmt", "footweets", "mmtc", "pfsmb"]
DEFAULT_GUIDELINES = ["default", "rocsmt", "footweets", "mmtc", "pfsmb"]
DEFAULT_CORPORA_CONFIG = os.path.join(
    os.environ.get("HOME", ""), "evaluation-challenges/src/llm/config/corpora.yaml"
)


def read_file(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle]


def read_json(path: str):
    import json

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str, data) -> None:
    import json

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)


def read_yaml(path: str):
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass(frozen=True)
class MetricSpec:
    name: str
    corpus_score_fn: Callable[[Sequence[int]], float]


class BleuMetric:
    def __init__(self, ref_lines: Sequence[str], baseline_lines: Sequence[str], system_lines: Sequence[str]):
        self.ref_lines = ref_lines
        self.baseline_lines = baseline_lines
        self.system_lines = system_lines
        self.metric = BLEU(effective_order=True)

    def corpus_score(self, system_lines: Sequence[str], indices: Sequence[int]) -> float:
        sampled_sys = [system_lines[i] for i in indices]
        sampled_ref = [self.ref_lines[i] for i in indices]
        return float(self.metric.corpus_score(sampled_sys, [sampled_ref]).score)

    def baseline_score(self, indices: Sequence[int]) -> float:
        return self.corpus_score(self.baseline_lines, indices)

    def system_score(self, indices: Sequence[int]) -> float:
        return self.corpus_score(self.system_lines, indices)


class MeanArrayMetric:
    def __init__(self, baseline_scores: Sequence[float], system_scores: Sequence[float]):
        self.baseline_scores = np.asarray(baseline_scores, dtype=float)
        self.system_scores = np.asarray(system_scores, dtype=float)

    def baseline_score(self, indices: Sequence[int]) -> float:
        return float(np.mean(self.baseline_scores[list(indices)]))

    def system_score(self, indices: Sequence[int]) -> float:
        return float(np.mean(self.system_scores[list(indices)]))


def paired_ttest(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if ttest_rel is not None:
        result = ttest_rel(x, y)
        return float(result.statistic), float(result.pvalue)

    diffs = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    n = diffs.size
    if n < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(diffs))
    std = float(np.std(diffs, ddof=1))
    if std == 0.0:
        return float("inf") if mean != 0 else 0.0, 0.0 if mean != 0 else 1.0
    t_stat = mean / (std / math.sqrt(n))
    # Large-sample normal approximation as fallback.
    pvalue = math.erfc(abs(t_stat) / math.sqrt(2.0))
    return float(t_stat), float(pvalue)


def bootstrap_compare(
    baseline_score_fn: Callable[[Sequence[int]], float],
    system_score_fn: Callable[[Sequence[int]], float],
    n_items: int,
    n_splits: int = 300,
    sample_ratio: float = 0.4,
    seed: int = 13,
) -> Dict[str, float]:
    if not (0 < sample_ratio <= 1):
        raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
    if n_items < 2:
        raise ValueError("At least 2 examples are required for significance testing.")

    sample_size = max(2, int(round(n_items * sample_ratio)))
    rng = np.random.default_rng(seed)

    baseline_samples = np.zeros(n_splits, dtype=float)
    system_samples = np.zeros(n_splits, dtype=float)
    deltas = np.zeros(n_splits, dtype=float)

    for split_id in range(n_splits):
        indices = rng.choice(n_items, size=sample_size, replace=True)
        baseline_value = baseline_score_fn(indices)
        system_value = system_score_fn(indices)
        baseline_samples[split_id] = baseline_value
        system_samples[split_id] = system_value
        deltas[split_id] = system_value - baseline_value

    t_stat, t_pvalue = paired_ttest(system_samples, baseline_samples)

    p_ge_zero = float(np.mean(deltas >= 0.0))
    p_le_zero = float(np.mean(deltas <= 0.0))
    two_sided_pvalue = float(min(1.0, 2.0 * min(p_ge_zero, p_le_zero)))
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])

    return {
        "n_splits": int(n_splits),
        "sample_ratio": float(sample_ratio),
        "sample_size": int(sample_size),
        "baseline_mean": float(np.mean(baseline_samples)),
        "system_mean": float(np.mean(system_samples)),
        "delta_mean": float(np.mean(deltas)),
        "delta_std": float(np.std(deltas, ddof=1)) if n_splits > 1 else 0.0,
        "delta_ci_low": float(ci_low),
        "delta_ci_high": float(ci_high),
        "delta_median": float(np.median(deltas)),
        "wins": int(np.sum(deltas > 0.0)),
        "losses": int(np.sum(deltas < 0.0)),
        "ties": int(np.sum(deltas == 0.0)),
        "paired_ttest": {
            "statistic": float(t_stat),
            "pvalue": float(t_pvalue),
        },
        "bootstrap": {
            "pvalue_two_sided": float(two_sided_pvalue),
            "prob_system_ge_baseline": float(p_ge_zero),
            "prob_system_le_baseline": float(p_le_zero),
        },
        "significant_95_ci": bool(ci_low > 0.0 or ci_high < 0.0),
    }


def get_output_files(
    corpora: Iterable[str],
    models: Iterable[str],
    guidelines: Iterable[str],
    input_dir: str,
    corpora_config: str,
) -> List[Dict[str, str]]:
    config = read_yaml(corpora_config)
    items: List[Dict[str, str]] = []

    for corpus in corpora:
        src_file = os.path.expandvars(config[corpus]["src_file_path"])
        ref_file = os.path.expandvars(config[corpus]["ref_file_path"])
        src_file_name = os.path.basename(src_file)
        baseline_file = os.path.join(
            input_dir, "outputs", NLLB, corpus, f"{src_file_name}.out.postproc"
        )
        for model in models:
            for guideline in guidelines:
                system_file = os.path.join(
                    input_dir,
                    "outputs",
                    model,
                    corpus,
                    f"{src_file_name}.{guideline}.out.postproc",
                )
                items.append(
                    {
                        "corpus": corpus,
                        "model": model,
                        "guideline": guideline,
                        "ref_file": ref_file,
                        "baseline_file": baseline_file,
                        "system_file": system_file,
                    }
                )
    return items


def build_metric(metric_name: str, ref_file: str, baseline_file: str, system_file: str):
    metric_name = metric_name.lower()

    if metric_name == "bleu":
        ref_lines = read_file(ref_file)
        baseline_lines = read_file(baseline_file)
        system_lines = read_file(system_file)
        if not (len(ref_lines) == len(baseline_lines) == len(system_lines)):
            raise ValueError(
                f"Mismatched line counts for BLEU: ref={len(ref_lines)}, "
                f"baseline={len(baseline_lines)}, system={len(system_lines)}"
            )
        metric = BleuMetric(ref_lines, baseline_lines, system_lines)
        return metric, len(ref_lines)

    if metric_name in {"comet", "cometkiwi"}:
        baseline_comet_file = f"{baseline_file}.comet.json"
        system_comet_file = f"{system_file}.comet.json"
        baseline_scores = read_json(baseline_comet_file)[metric_name]
        system_scores = read_json(system_comet_file)[metric_name]
        if len(baseline_scores) != len(system_scores):
            raise ValueError(
                f"Mismatched sentence-level scores for {metric_name}: "
                f"baseline={len(baseline_scores)}, system={len(system_scores)}"
            )
        metric = MeanArrayMetric(baseline_scores, system_scores)
        return metric, len(baseline_scores)

    raise ValueError(f"Unsupported metric: {metric_name}")


def update_scores_ci_file(output_file: str, metric_results: Dict[str, Dict]) -> None:
    scores_ci_file = f"{output_file}.scores_ci.json"
    if os.path.exists(scores_ci_file):
        scores_ci = read_json(scores_ci_file)
    else:
        scores_ci = {}
    scores_ci.update(metric_results)
    write_json(scores_ci_file, scores_ci)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired significance testing against the NLLB baseline using bootstrap resampling."
    )
    parser.add_argument("-i", "--input-dir", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("-c", "--corpora", type=str, nargs="+", default=CORPORA)
    parser.add_argument("-m", "--models", type=str, nargs="+", default=[TOWER, LLAMA, GEMMA])
    parser.add_argument("-g", "--guidelines", type=str, nargs="+", default=DEFAULT_GUIDELINES)
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["bleu"],
        help="Metrics to test. Supported: bleu, comet, cometkiwi",
    )
    parser.add_argument("--n-splits", type=int, default=300)
    parser.add_argument("--sample-ratio", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--corpora-config", type=str, default=DEFAULT_CORPORA_CONFIG)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing metric-specific entries in .scores_ci.json",
    )
    args = parser.parse_args()

    outputs = get_output_files(
        corpora=args.corpora,
        models=args.models,
        guidelines=args.guidelines,
        input_dir=args.input_dir,
        corpora_config=args.corpora_config,
    )

    for item in outputs:
        system_file = item["system_file"]
        baseline_file = item["baseline_file"]
        ref_file = item["ref_file"]
        descriptor = f"{item['model']} | {item['corpus']} | {item['guideline']}"

        if not os.path.exists(system_file):
            print(f"Skipping missing system output: {descriptor}")
            continue
        if not os.path.exists(baseline_file):
            print(f"Skipping missing baseline output: {descriptor}")
            continue

        scores_ci_file = f"{system_file}.scores_ci.json"
        existing_scores_ci = read_json(scores_ci_file) if os.path.exists(scores_ci_file) else {}

        metric_results: Dict[str, Dict] = {}
        for metric_name in args.metrics:
            metric_key = metric_name.lower()
            metric_already_present = metric_key in existing_scores_ci
            if metric_already_present and not args.overwrite:
                print(f"Skipping existing {metric_name} stats for {descriptor}")
                continue

            try:
                metric, n_items = build_metric(metric_name, ref_file, baseline_file, system_file)
                stats = bootstrap_compare(
                    baseline_score_fn=metric.baseline_score,
                    system_score_fn=metric.system_score,
                    n_items=n_items,
                    n_splits=args.n_splits,
                    sample_ratio=args.sample_ratio,
                    seed=args.seed,
                )
                metric_results[metric_key] = stats
                print(
                    f"Computed {metric_name} stats for {descriptor}: "
                    f"Δ={stats['delta_mean']:.4f}, "
                    f"t-test p={stats['paired_ttest']['pvalue']:.4g}, "
                    f"bootstrap p={stats['bootstrap']['pvalue_two_sided']:.4g}"
                )
            except FileNotFoundError as exc:
                print(f"Skipping {metric_name} for {descriptor}: missing file ({exc})")
            except Exception as exc:
                print(f"Failed on {metric_name} for {descriptor}: {exc}")

        if metric_results:
            update_scores_ci_file(system_file, metric_results)


if __name__ == "__main__":
    main()
