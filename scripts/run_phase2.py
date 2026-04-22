import os, sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from src.cleaning import clean_products

def main() -> None:
    df, stats = clean_products()

    print("\n" + "=" * 70)
    print("SOUHRN FÁZE 2")
    print("=" * 70)
    print(f"Vstupních produktů:   {stats['input_rows']}")
    print(f"Odstraněno (invalid): {stats['removed_invalid']}")
    print(f"  Opraveno (hustota):   {stats['density_fixed']}")
    print(f"  Imputace:             {stats['imputed']}")
    print(f"  Výstupních produktů:  {stats['output_rows']}")
    print(f"  Zachováno:            {stats['retention_pct']}%")
    print("=" * 70)
    print(f"\nSloupce: {list(df.columns)}")
    print(f"Shape:   {df.shape}")
    print(f"\nCategory groups:\n{df['category_group'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
