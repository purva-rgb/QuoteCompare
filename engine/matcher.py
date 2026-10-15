
from difflib import SequenceMatcher

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


def _normal_value(value) -> str | None:
    if value is None:
        return None

    return " ".join(str(value).casefold().split())


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


def name_similarity(
    left: QuotationItem,
    right: QuotationItem,
) -> float:
    a = left.normalized_name or left.original_description
    b = right.normalized_name or right.original_description

    return SequenceMatcher(
        None,
        a.casefold(),
        b.casefold(),
    ).ratio()


def find_candidates(
    source: QuotationItem,
    other: Quotation,
    threshold: float = 0.65,
) -> list[dict]:
    """
    Return ranked candidate matches from another quotation.

    This function never confirms a match automatically.
    """
    results = []

    for candidate in other.items:
        score = name_similarity(source, candidate)
        conflicts = specification_conflicts(source, candidate)

        if score < threshold:
            continue

        if conflicts:
            status = "needs_review"
        elif score >= 0.90:
            status = "candidate_strong"
        else:
            status = "candidate_possible"

        results.append({
            "source_item_id": source.item_id,
            "candidate_item_id": candidate.item_id,
            "similarity": round(score, 3),
            "conflicting_specifications": conflicts,
            "status": status,
        })

    return sorted(
        results,
        key=lambda result: result["similarity"],
        reverse=True,
    )
