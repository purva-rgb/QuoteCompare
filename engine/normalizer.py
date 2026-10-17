
import re
from decimal import Decimal

from engine.schemas import Quotation


# Canonical units are application conventions, not assumptions
# about product equivalence.
UNIT_ALIASES = {
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",

    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",

    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "kgs": "kg",

    "g": "g",
    "gram": "g",
    "grams": "g",

    "l": "L",
    "liter": "L",
    "liters": "L",
    "litre": "L",
    "litres": "L",

    "ml": "mL",
    "pcs": "piece",
    "pc": "piece",
    "piece": "piece",
    "pieces": "piece",
    "nos": "piece",
    "no.": "piece",
    "each": "piece",
    "unit": "piece",
    "units": "piece",
}


def normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None

    key = re.sub(r"\s+", " ", unit.strip().lower())

    if not key:
        return None

    # Unknown units must remain visible for manual review.
    return UNIT_ALIASES.get(key, unit.strip())


# Whole-word wording variants only. Specifications, grades, and
# dimensions are left in place.
_EQUIVALENT_WORDING = (
    (re.compile(r"\bset[\s-]+up\b"), "setup"),
    (re.compile(r"\ba\s*/\s*v\b"), "audiovisual"),
    (re.compile(r"\baudio\s*/\s*visual\b"), "audiovisual"),
    (re.compile(r"\baudio\s+visual\b"), "audiovisual"),
)


def prepare_comparison_text(description: str) -> str:
    """Case, spacing, and a few equivalent phrases. Specs stay intact."""
    text = description.casefold().replace("&", " and ")
    text = text.replace("×", " x ")
    text = re.sub(r"[\t\r\n]+", " ", text)
    for pattern, replacement in _EQUIVALENT_WORDING:
        text = pattern.sub(replacement, text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_name(description: str) -> str:
    """
    Conservative text normalization.

    Do not remove grades, model numbers, dimensions or material
    names: those details may determine product equivalence.
    """
    return prepare_comparison_text(description)


def normalize_quotation(quotation: Quotation) -> Quotation:
    """
    Apply non-destructive normalization.

    The original description and original numeric values are kept.
    """
    for item in quotation.items:
        item.normalized_name = normalize_name(
            item.original_description
        )

        item.unit = normalize_unit(item.unit)

        # Normalize known specification units, without converting
        # values unless a validated conversion rule is defined.
        for specification in item.specifications.values():
            if specification.unit:
                specification.unit = normalize_unit(
                    specification.unit
                )

        if item.quantity is not None:
            item.quantity = Decimal(str(item.quantity))

        if item.unit_price is not None:
            item.unit_price = Decimal(str(item.unit_price))

        if item.line_total is not None:
            item.line_total = Decimal(str(item.line_total))

    return quotation
