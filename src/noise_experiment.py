"""Experiment s sumem. Robustnost ML modelu vs pravidla."""

import logging
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from src.models import (
    split_data,
    build_model_pipeline,
    get_param_grid,
    train_with_gridsearch,
    evaluate_model,
)

logger = logging.getLogger(__name__)
CLASS_ORDER = ["shelf_picking", "front_zone_bin", "special_zone", "floor_block", "pallet_rack"]


def inject_label_noise(
    y: pd.Series,
    noise_level: float,
    random_state: int = 42,
) -> tuple[pd.Series, dict]:
    rng = np.random.RandomState(random_state)
    y_noisy = y.copy()
    classes = sorted(y.unique())
    flip_stats: dict[str, int] = {}

    for cls in classes:
        cls_mask = y == cls
        cls_indices = y.index[cls_mask]
        n_cls = len(cls_indices)
        n_flip = int(round(n_cls * noise_level))

        if n_flip == 0:
            flip_stats[cls] = 0
            continue

        flip_idx = rng.choice(cls_indices, size=n_flip, replace=False)

        # Flip vzdy do jine tridy
        other_classes = [c for c in classes if c != cls]
        new_labels = rng.choice(other_classes, size=n_flip)

        y_noisy.loc[flip_idx] = new_labels
        flip_stats[cls] = n_flip

    assert y_noisy.index.equals(y.index), "Index se neshoduje"
    assert set(y_noisy.unique()) <= set(classes), "Nove kategorie ve vystupu"
    y_noisy = y_noisy.astype(y.dtype)

    return y_noisy, flip_stats


def run_single_experiment(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_noisy: pd.Series,
    y_test: pd.Series,
    model_type: str,
) -> dict:
    if model_type == "xgb":
        # Union train+test zajisti, ze encoder zna vsechny tridy i kdyz noise nahodou nejakou v trainu vyluci
        all_classes = sorted(set(y_train_noisy.unique()) | set(y_test.unique()))
        le = LabelEncoder()
        le.fit(all_classes)

        y_train_enc = pd.Series(le.transform(y_train_noisy), index=y_train_noisy.index)
        y_test_enc = pd.Series(le.transform(y_test), index=y_test.index)

        yt_train = y_train_enc
        yt_test = y_test_enc
    else:
        yt_train = y_train_noisy
        yt_test = y_test

    pipeline = build_model_pipeline(model_type)
    param_grid = get_param_grid(model_type)
    grid, train_time = train_with_gridsearch(
        pipeline, param_grid, X_train, yt_train, model_type=model_type
    )
    best_est = grid.best_estimator_

    eval_result = evaluate_model(best_est, X_test, yt_test)

    f1_per_class = eval_result["f1_per_class"]
    if model_type == "xgb":
        f1_per_class = {
            le.inverse_transform([k])[0]: v
            for k, v in f1_per_class.items()
        }

    return {
        "accuracy": eval_result["accuracy"],
        "f1_macro": eval_result["f1_macro"],
        "f1_per_class": f1_per_class,
        "train_time_s": train_time,
        "best_params": grid.best_params_,
    }


# split -> pro kazdy (noise_level, seed, model) inject noise, train, evaluate; crash recovery pres CSV.
def run_noise_experiment(
    input_path: str = "data/processed/products_labeled.csv",
    noise_levels: Optional[list[float]] = None,
    seeds: Optional[list[int]] = None,
    model_types: Optional[list[str]] = None,
    results_dir: str = "results/phase5b_noise",
) -> pd.DataFrame:
    if noise_levels is None:
        noise_levels = [0.10, 0.15, 0.20]
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]
    if model_types is None:
        model_types = ["dt", "rf", "xgb"]

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = results_dir / "noise_experiment_raw.csv"

    X_train, X_test, y_train, y_test = split_data(input_path)

    completed: set[tuple] = set()
    if raw_csv_path.exists():
        existing_df = pd.read_csv(raw_csv_path)
        for _, row in existing_df.iterrows():
            completed.add((row["noise_level"], row["seed"], row["model"]))
        logger.info("Crash recovery: nalezeno %d dokonceny behu", len(completed))
    else:
        existing_df = pd.DataFrame()

    total_runs = len(noise_levels) * len(seeds) * len(model_types)
    run_idx = 0

    rows: list[dict] = []

    for noise_level in noise_levels:
        for seed in seeds:
            # Stejny noisy train set pouzivame pro vsechny modely daneho (noise_level, seed)
            y_train_noisy, flip_stats = inject_label_noise(y_train, noise_level, random_state=seed)
            actual_flip_rate = sum(flip_stats.values()) / len(y_train)
            logger.info(
                "Noise %.0f%%, seed=%d: flipnuto %d vzorku (%.2f%%), per-class: %s",
                noise_level * 100,
                seed,
                sum(flip_stats.values()),
                actual_flip_rate * 100,
                flip_stats,
            )

            for mt in model_types:
                run_idx += 1
                combo = (noise_level, seed, mt)

                if combo in completed:
                    logger.info("Run %d/%d: %s @ noise=%.2f, seed=%d — PRESKOCENO (jiz hotovo)", run_idx, total_runs, mt, noise_level, seed)
                    continue

                logger.info("Run %d/%d: %s @ noise=%.2f, seed=%d", run_idx, total_runs, mt, noise_level, seed)

                result = run_single_experiment(X_train, X_test, y_train_noisy, y_test, mt)

                row = {
                    "noise_level": noise_level,
                    "seed": seed,
                    "model": mt,
                    "accuracy": result["accuracy"],
                    "f1_macro": result["f1_macro"],
                    "train_time_s": result["train_time_s"],
                    "best_params": str(result["best_params"]),
                }
                for cls in CLASS_ORDER:
                    row[f"f1_{cls}"] = result["f1_per_class"].get(cls, 0.0)

                rows.append(row)

                row_df = pd.DataFrame([row])
                if raw_csv_path.exists():
                    row_df.to_csv(raw_csv_path, mode="a", header=False, index=False)
                else:
                    row_df.to_csv(raw_csv_path, index=False)

    if raw_csv_path.exists():
        results_df = pd.read_csv(raw_csv_path)
    else:
        results_df = pd.DataFrame(rows)

    results_df = compute_denoising_gain(results_df)
    results_df.to_csv(raw_csv_path, index=False)

    phase5_comparison_path = results_dir.parent / "phase5_modeling" / "model_comparison_table.csv"
    summary_df = generate_summary_table(results_df, phase5_comparison_path)
    summary_df.to_csv(results_dir / "noise_experiment_summary.csv", index=False)
    logger.info("Ulozena noise_experiment_summary.csv")

    plot_accuracy_vs_noise(summary_df, str(results_dir / "noise_accuracy_vs_noise_level.png"))
    plot_denoising_gain(summary_df, str(results_dir / "noise_denoising_gain.png"))
    plot_f1_degradation(summary_df, str(results_dir / "noise_f1_degradation.png"))

    logger.info("Experiment dokoncen. Celkem %d behu.", len(results_df))
    return results_df


# Prida sloupec denoising_gain = accuracy - (1 - noise_level)
def compute_denoising_gain(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()
    df["denoising_gain"] = df["accuracy"] - (1.0 - df["noise_level"])
    return df


def generate_summary_table(
    results_df: pd.DataFrame,
    phase5_comparison_path: str | Path = "results/phase5_modeling/model_comparison_table.csv",
) -> pd.DataFrame:
    phase5_comparison_path = Path(phase5_comparison_path)

    agg = (
        results_df
        .groupby(["noise_level", "model"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            denoising_gain_mean=("denoising_gain", "mean"),
            denoising_gain_std=("denoising_gain", "std"),
        )
        .reset_index()
    )

    if phase5_comparison_path.exists():
        phase5_df = pd.read_csv(phase5_comparison_path)
        phase5_ml = phase5_df[phase5_df["model"].isin(["dt", "rf", "xgb"])].copy()

        zero_rows = []
        for _, row in phase5_ml.iterrows():
            zero_rows.append({
                "noise_level": 0.0,
                "model": row["model"],
                "accuracy_mean": row["accuracy"],
                "accuracy_std": 0.0,
                "f1_macro_mean": row["f1_macro"],
                "f1_macro_std": 0.0,
                "denoising_gain_mean": row["accuracy"] - 1.0,
                "denoising_gain_std": 0.0,
            })

        if zero_rows:
            zero_df = pd.DataFrame(zero_rows)
            agg = pd.concat([zero_df, agg], ignore_index=True)
            logger.info("Pridana 0%% data z Faze 5 (%s)", phase5_comparison_path)
    else:
        logger.warning(
            "Soubor %s nenalezen — 0%% noise nebude zahrnut v summary.",
            phase5_comparison_path,
        )

    agg = agg.sort_values(["model", "noise_level"]).reset_index(drop=True)

    return agg

##
def plot_accuracy_vs_noise(
    summary_df: pd.DataFrame,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    model_styles = {
        "dt": {"label": "Decision Tree", "marker": "s", "color": "#1f77b4"},
        "rf": {"label": "Random Forest", "marker": "o", "color": "#2ca02c"},
        "xgb": {"label": "XGBoost", "marker": "^", "color": "#d62728"},
    }

    all_noise_levels = sorted(summary_df["noise_level"].unique())
    noise_floor = [1.0 - nl for nl in all_noise_levels]
    ax.plot(
        [nl * 100 for nl in all_noise_levels],
        noise_floor,
        "--",
        color="grey",
        linewidth=1.5,
        label="Noise floor (1 - noise)",
        zorder=1,
    )

    best_acc_per_noise = {}
    for nl in all_noise_levels:
        subset = summary_df[summary_df["noise_level"] == nl]
        if not subset.empty:
            best_acc_per_noise[nl] = subset["accuracy_mean"].max()

    best_accs = [best_acc_per_noise.get(nl, np.nan) for nl in all_noise_levels]
    ax.fill_between(
        [nl * 100 for nl in all_noise_levels],
        noise_floor,
        best_accs,
        alpha=0.15,
        color="grey",
        label="Area of Learned Resilience",
        zorder=0,
    )

    for mt in ["dt", "rf", "xgb"]:
        mt_data = summary_df[summary_df["model"] == mt].sort_values("noise_level")
        if mt_data.empty:
            continue
        style = model_styles[mt]
        ax.errorbar(
            mt_data["noise_level"] * 100,
            mt_data["accuracy_mean"],
            yerr=mt_data["accuracy_std"],
            label=style["label"],
            marker=style["marker"],
            color=style["color"],
            linewidth=2,
            capsize=4,
            zorder=2,
        )

    ax.set_xlabel("Label Noise Level (%)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Model Accuracy vs. Label Noise Level", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xticks([nl * 100 for nl in all_noise_levels])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Ulozen graf: %s", output_path)


def plot_denoising_gain(
    summary_df: pd.DataFrame,
    output_path: str,
) -> None:
    df_plot = summary_df[summary_df["noise_level"] > 0].copy()
    if df_plot.empty:
        logger.warning("Zadna data s noise_level > 0 pro plot_denoising_gain.")
        return

    noise_levels = sorted(df_plot["noise_level"].unique())
    models = ["dt", "rf", "xgb"]
    model_labels = {"dt": "Decision Tree", "rf": "Random Forest", "xgb": "XGBoost"}

    x = np.arange(len(noise_levels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    model_colors = {"dt": "#1f77b4", "rf": "#2ca02c", "xgb": "#d62728"}

    for i, mt in enumerate(models):
        mt_data = df_plot[df_plot["model"] == mt].sort_values("noise_level")
        gains = []
        for nl in noise_levels:
            row = mt_data[mt_data["noise_level"] == nl]
            if not row.empty:
                gains.append(row["denoising_gain_mean"].values[0])
            else:
                gains.append(0.0)

        gains = np.array(gains)

        ax.bar(
            x + i * width,
            gains,
            width,
            label=model_labels.get(mt, mt),
            color=model_colors[mt],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1.0)

    ax.set_xlabel("Label Noise Level (%)", fontsize=12)
    ax.set_ylabel("Denoising Gain (accuracy - noise floor)", fontsize=12)
    ax.set_title("Denoising Gain by Model and Noise Level", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{nl * 100:.0f}%" for nl in noise_levels])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Ulozen graf: %s", output_path)


def plot_f1_degradation(
    summary_df: pd.DataFrame,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    model_styles = {
        "dt": {"label": "Decision Tree", "marker": "s", "color": "#1f77b4"},
        "rf": {"label": "Random Forest", "marker": "o", "color": "#2ca02c"},
        "xgb": {"label": "XGBoost", "marker": "^", "color": "#d62728"},
    }

    for mt in ["dt", "rf", "xgb"]:
        mt_data = summary_df[summary_df["model"] == mt].sort_values("noise_level")
        if mt_data.empty:
            continue
        style = model_styles[mt]
        ax.errorbar(
            mt_data["noise_level"] * 100,
            mt_data["f1_macro_mean"],
            yerr=mt_data["f1_macro_std"],
            label=style["label"],
            marker=style["marker"],
            color=style["color"],
            linewidth=2,
            capsize=4,
        )

    ax.set_xlabel("Label Noise Level (%)", fontsize=12)
    ax.set_ylabel("F1 Macro Score", fontsize=12)
    ax.set_title("F1 Macro Degradation with Increasing Label Noise", fontsize=14)
    ax.legend(loc="lower left", fontsize=10)

    all_noise_levels = sorted(summary_df["noise_level"].unique())
    ax.set_xticks([nl * 100 for nl in all_noise_levels])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Ulozen graf: %s", output_path)
