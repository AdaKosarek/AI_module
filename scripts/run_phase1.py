"""Fáze 1: načtení dat, propojení tabulek, výpočet obrátkovosti."""

import logging
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from src.data_preparation import build_products_clean

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")


def main() -> None:
    df = build_products_clean()
    print(f"\nHotovo: {df.shape[0]:,} produktů, {df.shape[1]} sloupců")
    print(f"Uloženo do: data/processed/products_clean.csv")


if __name__ == "__main__":
    main()
