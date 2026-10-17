
from decimal import Decimal

import pytest

from engine.schemas import (
    Quotation,
    QuotationItem,
    QuotationTotals,
    Specification,
    Vendor,
)
from engine.normalizer import normalize_quotation, normalize_unit
from engine.validator import validate_quotation
from engine.matcher import (
    find_candidates,
    specification_conflicts,
)


def make_item(
    item_id="item_1",
    description="Copper cable 2.5 sq mm",
    quantity="10",
    unit="meters",
    unit_price="500",
    line_total="5000",
    specs=None,
):
    return QuotationItem(
        item_id=item_id,
        original_description=description,
        quantity=Decimal(quantity) if quantity is not None else None,
        unit=unit,
        unit_price=(
            Decimal(unit_price) if unit_price is not None else None
        ),
        line_total=(
            Decimal(line_total) if line_total is not None else None
        ),
        specifications=specs or {},
    )


def make_quotation(items, subtotal="5000"):
    return Quotation(
        vendor=Vendor(name="Test Vendor", currency="INR"),
        items=items,
        totals=QuotationTotals(
            subtotal=Decimal(subtotal) if subtotal is not None else None
        ),
    )


def test_unit_aliases():
    assert normalize_unit("meters") == "m"
    assert normalize_unit("KG") == "kg"
    assert normalize_unit("pcs") == "piece"


def test_unknown_unit_is_preserved():
    assert normalize_unit("bundle") == "bundle"


def test_normalizer_preserves_original_description():
    item = make_item()
    quotation = make_quotation([item])

    normalize_quotation(quotation)

    assert item.original_description == "Copper cable 2.5 sq mm"
    assert item.normalized_name == "copper cable 2.5 sq mm"
    assert item.unit == "m"



def test_valid_arithmetic():
    item = make_item()
    item.tax_rate = Decimal("0")  # Explicitly known, for this test.
    quotation = make_quotation([item])
    # The line states zero tax, so the quotation totals must state it too.
    # Grand total is the fixture subtotal plus that known tax, not a guess.
    quotation.totals.tax_total = Decimal("0")
    quotation.totals.grand_total = quotation.totals.subtotal + Decimal("0")

    result = validate_quotation(quotation)

    assert result.status == "valid"


def test_mismatched_subtotal_needs_review():
    quotation = make_quotation(
        [make_item()],
        subtotal="4500",
    )

    result = validate_quotation(quotation)

    assert result.status == "needs_review"
    assert any("subtotal" in issue.lower() for issue in result.issues)


def test_missing_tax_is_not_treated_as_zero():
    quotation = make_quotation([make_item()])

    result = validate_quotation(quotation)

    assert any("tax rate is unknown" in issue for issue in result.issues)


def test_empty_items_rejected():
    with pytest.raises(ValueError):
        make_quotation([])


def test_different_critical_specifications_conflict():
    a = make_item(specs={
        "material": Specification(value="copper")
    })
    b = make_item(
        item_id="item_2",
        specs={"material": Specification(value="aluminium")},
    )

    assert "material" in specification_conflicts(a, b)


def test_similar_item_is_only_a_candidate():
    a = make_item()
    b = make_item(item_id="item_2")

    other = make_quotation([b])

    candidates = find_candidates(a, other)

    assert len(candidates) == 1
    assert candidates[0]["status"] == "candidate_strong"
