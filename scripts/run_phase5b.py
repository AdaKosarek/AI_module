"""Faze 5b: Experiment s sumem"""

import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
os.chdir(_PROJECT_ROOT)
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    from src.noise_experiment import run_noise_experiment

    logger.info("Faze 5b: Experiment s sumem")

    results_df = run_noise_experiment(
        input_path="data/processed/products_labeled.csv",
        noise_levels=[0.10, 0.15, 0.20],
        seeds=[42, 123, 456, 789, 1024],
        model_types=["dt", "rf", "xgb"],
        results_dir="results/phase5b_noise",
    )

    print("\n" + "=" * 80)
    print("VYSLEDKY EXPERIMENTU S SUMEM")
    print("=" * 80)

    summary = results_df.groupby(["noise_level", "model"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        denoising_gain_mean=("denoising_gain", "mean"),
    ).reset_index()

    for _, row in summary.iterrows():
        print(
            f"  noise={row['noise_level']:.0%}, model={row['model']:<4} | "
            f"accuracy={row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f} | "
            f"denoising_gain={row['denoising_gain_mean']:+.4f}"
        )

    print("=" * 80)
    print("Hotovo.")

if __name__ == "__main__":
    main()
