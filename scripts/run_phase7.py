"""Faze 7: SHAP analyza"""

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
    from src.shap_analysis import run_shap_analysis

    logger.info("Faze 7: SHAP interpretace")

    result = run_shap_analysis(
        input_path="data/processed/products_labeled.csv",
        models_dir="models",
        results_dir="results/phase7_shap",
    )

    print("\n" + "=" * 70)
    print("SHAP INTERPRETACE — VYSLEDKY")
    print("=" * 70)
    print(f"  SHAP values shape: {result.get('shap_shape', 'N/A')}")
    print(f"  Pocet vygenerovanych grafu: {result.get('n_plots', 0)}")
    print(f"  Reprezentativni vzorky: {result.get('representative_samples', {})}")
    print("=" * 70)
    print("Hotovo.")


if __name__ == "__main__":
    main()
