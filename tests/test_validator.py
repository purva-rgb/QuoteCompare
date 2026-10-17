from decimal import Decimal

from engine.schemas import Quotation, QuotationItem, QuotationTotals, Vendor
from engine.validator import validate_quotation


def make(**totals):
    return Quotation(
        vendor=Vendor(name="BluePeak", currency="INR"),
        items=[
            QuotationItem(
                item_id="catering",
                original_description="Catering",
                quantity=Decimal("100"),
                unit="guests",
                unit_price=Decimal("850"),
                line_total=Decimal("85000"),
            )
        ],
        totals=QuotationTotals(**totals),
    )


def test_bill_level_tax_rate_clears_per_line_tax_warnings():
    quotation = make(
        subtotal=Decimal("85000"),
        tax_total=Decimal("15300"),
        grand_total=Decimal("100300"),
        tax_rate=Decimal("18"),
    )

    result = validate_quotation(quotation)

    assert result.status == "valid"
    assert result.issues == []


def test_missing_bill_level_rate_still_flags_each_line():
    quotation = make(
        subtotal=Decimal("85000"),
        tax_total=Decimal("15300"),
        grand_total=Decimal("100300"),
    )

    result = validate_quotation(quotation)

    assert any("tax rate is unknown" in issue for issue in result.issues)


def test_missing_grand_total_is_not_valid():
    quotation = make(subtotal=Decimal("85000"), tax_rate=Decimal("18"))

    result = validate_quotation(quotation)

    assert result.status == "needs_review"
    assert "Grand total is unknown." in result.issues
    assert "Tax total is unknown." in result.issues


def test_tax_total_that_disagrees_with_the_rate_is_flagged():
    quotation = make(
        subtotal=Decimal("85000"),
        tax_total=Decimal("1000"),
        grand_total=Decimal("86000"),
        tax_rate=Decimal("18"),
    )

    result = validate_quotation(quotation)

    assert result.status == "needs_review"
    assert any("tax rate of 18%" in issue for issue in result.issues)


def test_grand_total_that_is_not_subtotal_plus_tax_is_flagged():
    quotation = make(
        subtotal=Decimal("85000"),
        tax_total=Decimal("15300"),
        grand_total=Decimal("120000"),
        tax_rate=Decimal("18"),
    )

    result = validate_quotation(quotation)

    assert any(
        "Subtotal plus tax total" in issue for issue in result.issues
    )