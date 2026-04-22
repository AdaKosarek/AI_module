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

from src.labeling import label_products_pipeline

def main() -> None:
    df, stats = label_products_pipeline()

    print("\n" + "=" * 60)
    print("SOUHRN FAZE 3 - LABELING")
    print("=" * 60)
    print(f"Celkem produktu: {len(df)}")
    print()

    dist = df["storage_class"].value_counts()
    total = len(df)
    for cls_name, count in dist.items():
        pct = count / total * 100
        print(f"  {cls_name}: {count} ({pct:.1f}%)")



if __name__ == "__main__":
    main()
