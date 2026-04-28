"""Konstanty API modulu."""

from src.features import NUMERIC_FEATURES

CATEGORY_GROUPS = [
    "electronics",
    "furniture",
    "home_appliances",
    "beauty_health",
    "sports_leisure",
    "fashion",
    "toys_baby",
    "books_media",
    "home_garden",
    "food_drinks",
    "housewares",
    "stationery",
    "other",
]

STORAGE_CLASS_CZ = {
    "shelf_picking": "Police na ruční vychystávání",
    "front_zone_bin": "Přihrádka v přední zóně",
    "special_zone": "Speciální zóna",
    "floor_block": "Bloková zóna na zemi",
    "pallet_rack": "Paletová pozice v regálu",
}

CLASS_ORDER_LE = [
    "floor_block",
    "front_zone_bin",
    "pallet_rack",
    "shelf_picking",
    "special_zone",
]

NUMERIC_NO_TURNOVER = [f for f in NUMERIC_FEATURES if f != "daily_turnover"]

MAX_BATCH_SIZE = 100
