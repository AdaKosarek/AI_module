"""Faze 8: Cold-start simulace.."""

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
    from src.cold_start import run_cold_start_experiment

    logger.info("=== Faze 8: Cold-start simulace ===")

    result = run_cold_start_experiment(
        input_path="data/processed/products_labeled.csv",
        results_dir="results/phase8_cold_start",
    )

    print("\n" + "=" * 70)
    print("COLD-START SIMULACE — VYSLEDKY")
    print("=" * 70)

    import pandas as pd
    df = pd.read_csv("results/phase8_cold_start/cold_start_results.csv")
    for _, row in df.iterrows():
        front_f1 = row.get("f1_front_zone_bin", 0)
        print(
            f"  {row['variant']:<25} {row['model']:<5} | "
            f"F1={row['f1_macro']:.4f} | "
            f"front_zone_bin F1={front_f1:.4f}"
        )
    print(f"\n  ML advantage nad pravidly: +{result['ml_advantage_f1']:.4f} F1 macro")

    print("=" * 70)
    print("Hotovo.")


if __name__ == "__main__":
    main()
