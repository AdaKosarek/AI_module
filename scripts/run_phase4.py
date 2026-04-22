
import os, sys, logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from src.features import feature_engineering_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main() -> None:
    print("=" * 50)
    print("Faze 4: Feature Engineering")
    print("=" * 50)

    stats = feature_engineering_pipeline()

    print()
    print("Statistiky:")
    print(f"Radku na vstupu:  {stats['n_rows_input']}")
    print(f"Radku na vystupu: {stats['n_rows_output']}")
    print(f"Pocet features:   {stats['n_features']}")
    print(f"numeric:      {stats['n_numeric']}")
    print(f"OHE:          {stats['n_ohe']}")
    print(f"Vystupni CSV:     {stats['output_path']}")
    print(f"Pipeline soubory:")
    for p in stats["pipelines_saved"]:
        print(f"- {p}")
    print()
    print("Faze 4 dokoncena.")


if __name__ == "__main__":
    main()
