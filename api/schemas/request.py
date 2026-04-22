
from pydantic import BaseModel, Field, field_validator, model_validator

from api.constants import CATEGORY_GROUPS

# Limity odvozene z trenovacich dat
MAX_VOLUME_CM3 = 400_000
MAX_WEIGHT_G = 50_000
MAX_PRICE_BRL = 10_000
MAX_DIMENSION_CM = 150


class ProductInput(BaseModel):
    """Vstupni data produktu pro predikci skladove lokace.

    Limity jsou odvozene z trenovacich dat — vstupy mimo tyto rozsahy
    by zpusobily OOD (out-of-distribution) extrapolaci modelu.
    """

    product_weight_g: float = Field(
        gt=0, le=MAX_WEIGHT_G,
        description="Hmotnost v gramech (trénovací max 40 425 g)",
    )
    product_length_cm: float = Field(
        gt=0, le=MAX_DIMENSION_CM,
        description="Délka v cm (trénovací max ~105 cm)",
    )
    product_height_cm: float = Field(
        gt=0, le=MAX_DIMENSION_CM,
        description="Výška v cm (trénovací max ~105 cm)",
    )
    product_width_cm: float = Field(
        gt=0, le=MAX_DIMENSION_CM,
        description="Šířka v cm (trénovací max ~118 cm)",
    )
    category_group: str = Field(description="Kategorie produktu")
    avg_price: float = Field(
        gt=0, le=MAX_PRICE_BRL,
        description="Cena produktu v BRL (trénovací max 6 735 BRL)",
    )
    daily_turnover: float = Field(
        ge=0, default=0.0,
        description="Denní obrátkovost",
    )
    cold_start: bool = Field(default=True, description="Režim studeného startu")

    @field_validator("category_group")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if v not in CATEGORY_GROUPS:
            raise ValueError(
                f"Neplatná kategorie: {v}. Povolené: {CATEGORY_GROUPS}"
            )
        return v

    @model_validator(mode="after")
    def validate_combined_volume(self) -> "ProductInput":
        volume = (
            self.product_length_cm
            * self.product_height_cm
            * self.product_width_cm
        )
        if volume > MAX_VOLUME_CM3:
            raise ValueError(
                f"Objem {volume:.0f} cm3 (D x V x S = "
                f"{self.product_length_cm} x {self.product_height_cm} x "
                f"{self.product_width_cm}) přesahuje povolený limit "
                f"{MAX_VOLUME_CM3} cm3. Trénovací data neobsahují "
                f"produkty s takovými rozměry."
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "product_weight_g": 500,
                    "product_length_cm": 20,
                    "product_height_cm": 10,
                    "product_width_cm": 15,
                    "category_group": "electronics",
                    "avg_price": 150.0,
                    "daily_turnover": 0.0,
                    "cold_start": True,
                }
            ]
        }
    }
