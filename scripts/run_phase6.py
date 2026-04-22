"""KNN podobnost"""

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
    from src.similarity import run_knn_analysis

    logger.info("Faze 6: KNN podobnost")

    result = run_knn_analysis(
        input_path="data/processed/products_labeled.csv",
        models_dir="models",
        results_dir="results/phase6_knn",
    )

    print("\n" + "=" * 70)
    print("KNN PODOBNOST — VYSLEDKY")
    print("=" * 70)
    print(f"  Nejlepsi K: {result['best_k']}")
    print(f"  KNN accuracy: {result['knn_accuracy']:.4f}")
    print(f"  XGB accuracy: {result['xgb_accuracy']:.4f}")
    print(f"  Celkova shoda KNN vs XGB: {result['overall_agreement']:.4f}")
    print("\n  Shoda per trida:")
    for cls, rate in result['per_class_agreement'].items():
        print(f"    {cls:<20} {rate:.4f}")
    print("=" * 70)
    print("Hotovo.")


if __name__ == "__main__":
    main()
