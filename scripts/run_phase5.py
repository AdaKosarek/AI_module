"""Faze 5: Trenovani modelu"""

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
    from src.models import train_all_models

    logger.info("Faze 5: Trenovani modelu")

    stats = train_all_models(
        input_path="data/processed/products_labeled.csv",
        models_dir="models",
        results_dir="results/phase5_modeling",
    )

    print("\n" + "=" * 80)
    print("SROVNANI MODELU")
    print("=" * 80)

    header = (
        f"{'Model':<12} {'Accuracy':>10} {'F1 macro':>10} "
        f"{'F1 shelf':>10} {'F1 front':>10} {'F1 special':>10} "
        f"{'F1 floor':>10} {'F1 pallet':>10} {'Train(s)':>10} {'Pred(ms)':>10}"
    )
    print(header)
    print("-" * len(header))

    for row in stats["comparison_table"]:
        line = (
            f"{row['model']:<12} {row['accuracy']:>10.4f} {row['f1_macro']:>10.4f} "
            f"{row['f1_shelf_picking']:>10.4f} {row['f1_front_zone_bin']:>10.4f} "
            f"{row['f1_special_zone']:>10.4f} {row['f1_floor_block']:>10.4f} "
            f"{row['f1_pallet_rack']:>10.4f} {row['train_time_s']:>10.2f} "
            f"{row['predict_time_ms']:>10.2f}"
        )
        print(line)

    print("=" * 80)
    print(f"\nNejlepsi model: {stats['best_model']} (F1 macro = {stats['best_f1_macro']:.4f})")
    print("Hotovo.")


if __name__ == "__main__":
    main()
