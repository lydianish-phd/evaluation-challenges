#!/usr/bin/env python3
import argparse
import os
import re

import pandas as pd
import matplotlib.pyplot as plt

from .utils import (
    NLLB,
    LLAMA,
    GEMMA,
    TOWER,
    MODEL_LABELS,
    ROCSMT,
    FOOTWEETS,
    MMTC,
    PFSMB,
    CORPUS_LABELS,
    GUIDELINE_LABELS,
    BLEU,
    COMET,
    COMETKIWI,
    METRIC_LABELS

)

# Camera-ready matplotlib defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

GUIDELINE_ORDER_ALL = ["default", ROCSMT, FOOTWEETS, MMTC, PFSMB]
GUIDELINE_ORDER_NO_DEFAULT = GUIDELINE_ORDER_ALL[1:]

CORPUS_ORDER = [ROCSMT, FOOTWEETS, MMTC, PFSMB]
MODEL_ORDER = [NLLB, LLAMA, GEMMA, TOWER]


def extract_guideline(file_name: str) -> str:
    match = re.search(r"\.(default|footweets|mmtc|pfsmb|rocsmt)\.out\.postproc$", file_name)
    if match:
        return match.group(1)
    return "baseline"


def get_csv_path(score_dir: str, corpus: str, comparison_mode: str) -> str:
    if comparison_mode == "vs_nllb":
        return os.path.join(score_dir, f"scores_ci_{corpus}.csv")
    if comparison_mode == "vs_default":
        return os.path.join(score_dir, f"scores_ci_default_{corpus}.csv")
    raise ValueError(f"Unsupported comparison_mode: {comparison_mode}")


def get_guideline_order(comparison_mode: str):
    if comparison_mode == "vs_nllb":
        return GUIDELINE_ORDER_ALL
    if comparison_mode == "vs_default":
        return GUIDELINE_ORDER_NO_DEFAULT
    raise ValueError(f"Unsupported comparison_mode: {comparison_mode}")


def prepare_delta_df(csv_path: str, metric: str, comparison_mode: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["guideline"] = df["file"].apply(extract_guideline)

    guideline_order = get_guideline_order(comparison_mode)
    df = df[df["guideline"].isin(guideline_order)].copy()

    needed_cols = [
        "model",
        "file",
        "guideline",
        f"{metric}_delta_mean",
        f"{metric}_delta_ci_low",
        f"{metric}_delta_ci_high",
        f"{metric}_significant_95_ci",
    ]
    df = df[needed_cols].dropna(subset=[f"{metric}_delta_mean"])

    df["guideline"] = pd.Categorical(df["guideline"], guideline_order, ordered=True)
    df = df.sort_values(["model", "guideline"])
    return df


def get_offsets(n_models: int):
    if n_models == 1:
        return [0.0]
    if n_models == 2:
        return [-0.12, 0.12]
    if n_models == 3:
        return [-0.2, 0.0, 0.2]
    if n_models == 4:
        return [-0.27, -0.09, 0.09, 0.27]
    raise ValueError(f"Unsupported number of models for plotting: {n_models}")


def sanitize_model_name(model: str) -> str:
    return MODEL_LABELS.get(model, model).replace(" ", "_").replace("/", "_")


def default_output_filename(metric: str, comparison_mode: str, model: str = None) -> str:
    if comparison_mode == "vs_nllb":
        if model is None:
            return f"delta_1x4_{metric}_vs_nllb.pdf"
        return f"delta_1x4_{metric}_vs_nllb_{sanitize_model_name(model)}.pdf"

    return f"delta_1x4_{metric}_vs_default_{sanitize_model_name(model)}.pdf"


def plot_delta_1x4(
    score_dir: str,
    metric: str,
    comparison_mode: str,
    output_path: str,
    model: str = None,
):
    guideline_order = get_guideline_order(comparison_mode)

    if comparison_mode == "vs_default" and model is None:
        raise ValueError("--model is required when comparison_mode is 'vs_default'.")

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 4.2), sharey=True)
    axes = axes.flatten()

    legend_handles = None
    legend_labels = None

    for ax, corpus in zip(axes, CORPUS_ORDER):
        csv_path = get_csv_path(score_dir, corpus, comparison_mode)
        df = prepare_delta_df(csv_path, metric, comparison_mode)

        if comparison_mode == "vs_nllb":
            if model is not None:
                df = df[df["model"] == model].copy()
                models = [model] if not df.empty else []
            else:
                models = [m for m in MODEL_ORDER if m != NLLB and m in set(df["model"])]
        else:
            df = df[df["model"] == model].copy()
            models = [model] if not df.empty else []

        if not models:
            ax.set_title(CORPUS_LABELS.get(corpus, corpus), fontsize=12, pad=6)
            ax.axhline(0, linestyle="--", linewidth=1)
            ax.set_xticks(range(len(guideline_order)))
            ax.set_xticklabels([GUIDELINE_LABELS.get(g, g) for g in guideline_order], rotation=20)
            ax.set_xlabel("Guideline", fontsize=11)
            continue

        offsets = get_offsets(len(models))
        x_base = list(range(len(guideline_order)))

        for offset, current_model in zip(offsets, models):
            sub = df[df["model"] == current_model].copy()
            sub = sub.set_index("guideline").reindex(guideline_order).reset_index()

            x = [i + offset for i in x_base]
            y = sub[f"{metric}_delta_mean"].to_numpy()
            yerr_low = y - sub[f"{metric}_delta_ci_low"].to_numpy()
            yerr_high = sub[f"{metric}_delta_ci_high"].to_numpy() - y

            ax.errorbar(
                x,
                y,
                yerr=[yerr_low, yerr_high],
                fmt="o",
                linewidth=1.5,
                markersize=5,
                capsize=3,
                label=MODEL_LABELS.get(current_model, current_model),
            )

            sig = sub[f"{metric}_significant_95_ci"].fillna(False).to_numpy()
            for xi, yi, is_sig in zip(x, y, sig):
                if is_sig:
                    ax.text(xi, yi, "*", ha="center", va="bottom", fontsize=11)

        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_xticks(x_base)
        ax.set_xticklabels([GUIDELINE_LABELS.get(g, g) for g in guideline_order], rotation=20)
        ax.set_title(CORPUS_LABELS.get(corpus, corpus), fontsize=12, pad=6)
        ax.set_xlabel("Guideline", fontsize=11)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if comparison_mode == "vs_nllb":
        ylabel = f"Δ {METRIC_LABELS.get(metric, metric)} vs {METRIC_LABELS.get(NLLB, NLLB)}"
        if model is None and legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                ncol=len(legend_labels),
                fontsize=12,
                frameon=False,
                bbox_to_anchor=(0.5, 0.99),
            )
            tight_rect = [0.04, 0.04, 1.0, 0.90]
        else:
            title = MODEL_LABELS.get(model, model) if model else None
            if title:
                fig.suptitle(title, fontsize=13, y=0.98)
            tight_rect = [0.04, 0.04, 1.0, 0.92]
    else:
        ylabel = f"Δ {METRIC_LABELS.get(metric, metric)} vs default"
        fig.suptitle(MODEL_LABELS.get(model, model), fontsize=13, y=0.98)
        tight_rect = [0.04, 0.04, 1.0, 0.92]

    fig.supylabel(ylabel, fontsize=12, x=0.04)
    fig.tight_layout(rect=tight_rect, pad=0.6, w_pad=0.8, h_pad=1.0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot 1x4 delta plots from aggregated CSV files.")
    parser.add_argument(
        "--score-dir",
        type=str,
        required=True,
        help="Directory containing aggregated score CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the PDF plot(s) will be saved.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=[BLEU, COMET, COMETKIWI],
        default=[COMET, COMETKIWI],
        help="One or more metrics to plot.",
    )
    parser.add_argument(
        "--comparison-mode",
        type=str,
        choices=["vs_nllb", "vs_default"],
        default="vs_nllb",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[LLAMA, GEMMA, TOWER],
        help=(
            "One or more models to plot. "
            "Required for vs_default. Optional for vs_nllb. "
            "If omitted in vs_nllb mode, all non-NLLB models are shown together."
        ),
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Optional explicit output PDF filename. "
            "Only valid when plotting a single metric and a single model "
            "(or no model in vs_nllb mode)."
        ),
    )
    args = parser.parse_args()

    if args.comparison_mode == "vs_default" and not args.models:
        raise ValueError("--models is required when --comparison-mode vs_default is used.")

    if args.output_name is not None:
        single_metric = len(args.metrics) == 1
        single_model_or_combined = (args.models is None) or (len(args.models) == 1)
        if not (single_metric and single_model_or_combined):
            raise ValueError(
                "--output-name can only be used with a single metric and a single model "
                "(or no model in vs_nllb mode)."
            )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.comparison_mode == "vs_nllb" and not args.models:
        # Combined plot with all non-NLLB models, one output per metric
        for metric in args.metrics:
            output_name = args.output_name or default_output_filename(
                metric=metric,
                comparison_mode=args.comparison_mode,
                model=None,
            )
            output_path = os.path.join(args.output_dir, output_name)

            plot_delta_1x4(
                score_dir=args.score_dir,
                metric=metric,
                comparison_mode=args.comparison_mode,
                output_path=output_path,
                model=None,
            )
            print(f"Saved plot to: {output_path}")
    else:
        # One output per (metric, model)
        models_to_plot = args.models if args.models else [m for m in MODEL_ORDER if m != NLLB]

        for model in models_to_plot:
            if args.comparison_mode == "vs_default" and model == NLLB:
                print(f"Skipping {model}: NLLB is not valid for vs_default.")
                continue

            for metric in args.metrics:
                output_name = (
                    args.output_name
                    if args.output_name is not None
                    else default_output_filename(
                        metric=metric,
                        comparison_mode=args.comparison_mode,
                        model=model,
                    )
                )
                output_path = os.path.join(args.output_dir, output_name)

                plot_delta_1x4(
                    score_dir=args.score_dir,
                    metric=metric,
                    comparison_mode=args.comparison_mode,
                    output_path=output_path,
                    model=model,
                )
                print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    main()