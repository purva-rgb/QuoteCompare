from decimal import Decimal
from difflib import SequenceMatcher

from engine.normalizer import normalize_unit, prepare_comparison_text
from engine.schemas import Quotation, QuotationItem


# Specification keys that commonly affect product equivalence.
CRITICAL_SPECS = {
    "material",
    "grade",
    "model",
    "brand",
    "diameter",
    "thickness",
    "cross_section",
    "capacity",
    "voltage",
    "power",
    "size",
    "standard",
}

# Plurals and abbreviations of the SAME unit. Different pricing bases
# (person, guest, hour, day, piece, package, lot) stay distinct.
UNIT_VARIANTS = {
    "guest": "guest",
    "guests": "guest",
    "person": "person",
    "persons": "person",
    "people": "person",
    "hour": "hour",
    "hours": "hour",
    "hr": "hour",
    "hrs": "hour",
    "day": "day",
    "days": "day",
    "piece": "piece",
    "pieces": "piece",
    "pc": "piece",
    "pcs": "piece",
    "package": "package",
    "packages": "package",
    "lot": "lot",
    "lots": "lot",
}


def _normal_value(value) -> str | None:
    if value is None:
        return None

    return " ".join(str(value).casefold().split())


def comparable_unit(unit: str | None) -> str | None:
    """A comparison key for a unit. Unknown text is kept, not dropped."""
    if unit is None or not str(unit).strip():
        return None

    normalized = normalize_unit(str(unit))
    if not normalized:
        return None

    key = normalized.casefold()
    return UNIT_VARIANTS.get(key, key)


def specification_conflicts(
    left: QuotationItem,
    right: QuotationItem,
) -> list[str]:
    conflicts = []

    left_specs = {
        key.casefold(): value
        for key, value in left.specifications.items()
    }
    right_specs = {
        key.casefold(): value
        for key, value in right.specifications.items()
    }

    for key in CRITICAL_SPECS:
        a = left_specs.get(key)
        b = right_specs.get(key)

        # Missing specifications are unknown, not equal.
        if a is None or b is None:
            continue

        a_value = _normal_value(a.value)
        b_value = _normal_value(b.value)
        a_unit = _normal_value(a.unit)
        b_unit = _normal_value(b.unit)

        if a_value is not None and b_value is not None:
            if a_value != b_value or a_unit != b_unit:
                conflicts.append(key)

    return conflicts


def _comparison_text(item: QuotationItem) -> str:
    raw = item.normalized_name or item.original_description
    return prepare_comparison_text(raw)


def name_similarity(
    left: QuotationItem,
    right: QuotationItem,
) -> float:
    return SequenceMatcher(
        None,
        _comparison_text(left),
        _comparison_text(right),
    ).ratio()


def _same_quantity(left, right) -> bool | None:
    if left is None or right is None:
        return None

    try:
        return abs(Decimal(str(left)) - Decimal(str(right))) <= Decimal(
            "0.0001"
        )
    except Exception:
        return str(left).strip() == str(right).strip()


def find_candidates(
    source: QuotationItem,
    other: Quotation,
    threshold: float = 0.65,
) -> list[dict]:
    """
    Return ranked candidate matches from another quotation.

    A high name score is never a confirmed match. Quantity, unit, and
    specification differences keep the candidate in review, and prices
    are comparable only when the quantity and unit are both present
    and the same.
    """
    results = []

    for candidate in other.items:
        score = name_similarity(source, candidate)
        if score < threshold:
            continue

        conflicts = specification_conflicts(source, candidate)
        reasons: list[str] = []

        if conflicts:
            reasons.append(
                "Specification conflict: " + ", ".join(conflicts) + "."
            )

        same_qty = _same_quantity(source.quantity, candidate.quantity)
        if same_qty is None:
            reasons.append(
                "Quantity is missing, so the items are not treated as "
                "equivalent."
            )
        elif not same_qty:
            reasons.append(
                f"Quantities differ ({source.quantity} vs "
                f"{candidate.quantity})."
            )

        left_unit = comparable_unit(source.unit)
        right_unit = comparable_unit(candidate.unit)
        if left_unit is None or right_unit is None:
            reasons.append(
                "Unit is missing, so the pricing basis is not comparable."
            )
        elif left_unit != right_unit:
            reasons.append(
                f"Units differ ({source.unit} vs {candidate.unit})."
            )

        basis_matches = not any(
            reason.startswith("Quantity")
            or reason.startswith("Quantities")
            or reason.startswith("Unit")
            or reason.startswith("Units")
            for reason in reasons
        )

        if conflicts or not basis_matches:
            status = "needs_review"
        elif score >= 0.90:
            status = "candidate_strong"
        else:
            status = "candidate_possible"
            reasons.append(
                f"Name similarity is {score:.2f}. The match is not confirmed."
            )

        prices_comparable = status != "needs_review" and basis_matches

        results.append({
            "source_item_id": source.item_id,
            "candidate_item_id": candidate.item_id,
            "similarity": round(score, 3),
            "conflicting_specifications": conflicts,
            "status": status,
            "review_reasons": reasons,
            "prices_comparable": prices_comparable,
        })

    return sorted(
        results,
        key=lambda result: result["similarity"],
        reverse=True,
    )
