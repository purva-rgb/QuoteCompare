from decimal import Decimal

from engine.matcher import find_candidates
from engine.schemas import (
    Quotation,
    QuotationItem,
    Specification,
    Vendor,
)


def item(**overrides):
    data = {
        "item_id": "item_1",
        "original_description": "Catering for the evening",
        "quantity": Decimal("100"),
        "unit": "guests",
        "unit_price": Decimal("850"),
        "line_total": Decimal("85000"),
    }
    data.update(overrides)
    return QuotationItem(**data)


def quotation(*items):
    return Quotation(
        vendor=Vendor(name="Other"),
        items=list(items),
    )


def test_same_item_is_a_strong_candidate_not_a_confirmed_match():
    source = item()
    other = item(item_id="item_2", unit="guest")

    candidates = find_candidates(source, quotation(other))

    assert len(candidates) == 1
    assert candidates[0]["status"] == "candidate_strong"
    assert candidates[0]["status"] != "matched"
    assert candidates[0]["prices_comparable"] is True


def test_person_and_guest_are_not_the_same_unit():
    source = item(unit="person")
    other = item(item_id="item_2", unit="guest")

    candidate = find_candidates(source, quotation(other))[0]

    assert candidate["status"] == "needs_review"
    assert candidate["prices_comparable"] is False
    assert any("Units differ" in reason for reason in candidate["review_reasons"])


def test_hour_and_day_are_not_treated_as_equivalent():
    source = item(unit="hour", quantity=Decimal("8"))
    other = item(item_id="item_2", unit="day", quantity=Decimal("1"))

    candidate = find_candidates(source, quotation(other))[0]

    assert candidate["status"] == "needs_review"
    assert candidate["prices_comparable"] is False
    assert any("Quantities differ" in reason for reason in candidate["review_reasons"])
    assert any("Units differ" in reason for reason in candidate["review_reasons"])


def test_missing_quantity_is_not_equivalence():
    source = item(quantity=None)
    other = item(item_id="item_2")

    candidate = find_candidates(source, quotation(other))[0]

    assert candidate["status"] == "needs_review"
    assert candidate["prices_comparable"] is False
    assert any("Quantity is missing" in reason for reason in candidate["review_reasons"])


def test_specification_conflict_stays_unconfirmed():
    source = item(specifications={
        "material": Specification(value="copper"),
    })
    other = item(
        item_id="item_2",
        specifications={"material": Specification(value="aluminium")},
    )

    candidate = find_candidates(source, quotation(other))[0]

    assert candidate["status"] == "needs_review"
    assert "material" in candidate["conflicting_specifications"]
    assert candidate["prices_comparable"] is False


def test_unrelated_description_is_not_a_match():
    source = item(original_description="Photography package")
    other = item(
        item_id="item_2",
        original_description="Delivery and logistics",
    )

    assert find_candidates(source, quotation(other)) == []


def test_equivalent_wording_can_score_as_a_candidate():
    source = item(original_description="Hall set-up")
    other = item(item_id="item_2", original_description="Hall setup")

    candidate = find_candidates(source, quotation(other))[0]

    assert candidate["similarity"] == 1.0
    assert candidate["status"] == "candidate_strong"


def test_possible_match_explains_that_it_is_not_confirmed():
    source = item(original_description="Venue and hall setup")
    other = item(item_id="item_2", original_description="Venue rental and hall")

    candidates = find_candidates(source, quotation(other), threshold=0.55)

    assert len(candidates) == 1
    assert candidates[0]["status"] == "candidate_possible"
    assert any(
        "not confirmed" in reason for reason in candidates[0]["review_reasons"]
    )
    assert candidates[0]["prices_comparable"] is True
