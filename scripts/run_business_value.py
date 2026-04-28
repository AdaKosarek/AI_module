"""Spustí analýzu vypoctu metrik vydalenosti cold start modelu"""

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
    from src.business_value import run_business_value_analysis

    logger.info("=== Doplněk: Business value cold-start ===")

    summary = run_business_value_analysis(
        input_path="data/processed/products_labeled.csv",
        models_dir="models",
        results_dir="results/phase_nadstavba_business_value",
    )

    import pandas as pd
    df_global = pd.read_csv("results/phase_nadstavba_business_value/global_savings_summary.csv")
    g = df_global.iloc[0]

    print("\n" + "=" * 70)
    print("BUSINESS VALUE - VÝSLEDKY")
    print(f"Front_zone_bin produkty v testu:     {int(g['n_front'])}")
    print(f"Pravidla - průměrná vzdálenost:  {g['avg_rules_dist']:.2f} m")
    print(f"ML model - průměrná vzdálenost:  {g['avg_ml_dist']:.2f} m")
    print(f"Úspora pro rychloobrátkové zboží:    {g['savings_pct']:.1f} %  ({g['savings_m']:.1f} m / vychystání)")
    print(f"Úspora globálně (celý test set):     {g['total_savings_pct']:.1f} %")
    print(f"Výstupy v: results/phase_nadstavba_business_value/")
    print("=" * 70)


if __name__ == "__main__":
    main()
