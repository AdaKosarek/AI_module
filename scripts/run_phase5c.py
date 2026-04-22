"""Faze 5c: Analyza zavaznosti chyb"""

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
    from src.error_severity import run_severity_analysis
    logger.info("=== Faze 5c: Analyza zavaznosti chyb ===")

    result = run_severity_analysis(
        input_path="data/processed/products_labeled.csv",
        models_dir="models",
        results_dir="results/phase5c_severity",
        phase5_results_dir="results/phase5_modeling",
    )

    print("\n" + "=" * 70)
    print("ANALYZA ZAVAZNOSTI CHYB")
    print("=" * 70)

    import pandas as pd
    scores_df = pd.DataFrame(result["scores"])
    for _, row in scores_df.iterrows():
        print(
            f"  {row['model']:<4} | TWE={row['total_weighted_error']:>6.0f} | "
            f"WER={row['weighted_error_rate']:.4f} | "
            f"Chyb (nevazene)={row['unweighted_error_count']:>4.0f}"
        )

    print("=" * 70)
    print("Hotovo.")


if __name__ == "__main__":
    main()
