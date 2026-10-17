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


def test_explicit_zero_tax_is_valid():
    quotation = make(
        subtotal=Decimal("85000"),
        tax_total=Decimal("0"),
        grand_total=Decimal("85000"),
        tax_rate=Decimal("0"),
    )
    quotation.items[0].tax_rate = Decimal("0")

    result = validate_quotation(quotation)

    assert result.status == "valid"
    assert result.issues == []
    assert quotation.totals.tax_total == Decimal("0")
    assert quotation.totals.grand_total == Decimal("85000")


def _line_matches(quotation, amount):
    quotation.items[0].quantity = Decimal("1")
    quotation.items[0].unit_price = amount
    quotation.items[0].line_total = amount


def test_rounding_within_tolerance_is_valid():
    quotation = make(
        subtotal=Decimal("100"),
        tax_total=Decimal("18.02"),
        grand_total=Decimal("118.02"),
        tax_rate=Decimal("18"),
    )
    _line_matches(quotation, Decimal("100"))

    result = validate_quotation(quotation)

    assert result.status == "valid"


def test_rounding_beyond_tolerance_is_flagged():
    quotation = make(
        subtotal=Decimal("100"),
        tax_total=Decimal("18.03"),
        grand_total=Decimal("118.03"),
        tax_rate=Decimal("18"),
    )
    _line_matches(quotation, Decimal("100"))

    result = validate_quotation(quotation)

    assert result.status == "needs_review"
    assert any("tax rate of 18%" in issue for issue in result.issues)


def test_missing_tax_total_is_not_filled_in():
    quotation = make(
        subtotal=Decimal("85000"),
        tax_rate=Decimal("18"),
        grand_total=Decimal("100300"),
    )

    result = validate_quotation(quotation)

    assert quotation.totals.tax_total is None
    assert result.status == "needs_review"
    assert "Tax total is unknown." in result.issues
    assert any(issue.startswith("Diagnostic only:") for issue in result.issues)
    assert not any(
        "arithmetic" in issue.lower() and "unknown" in issue.lower()
        for issue in result.issues
    )


def test_missing_fields_are_reported_separately():
    quotation = make()

    result = validate_quotation(quotation)

    assert "Subtotal is unknown." in result.issues
    assert "Tax total is unknown." in result.issues
    assert "Grand total is unknown." in result.issues


def test_unreadable_characters_need_review_without_changing_totals():
    from engine.schemas import TextQuality

    quotation = make(
        subtotal=Decimal("85000"),
        tax_total=Decimal("15300"),
        grand_total=Decimal("100300"),
        tax_rate=Decimal("18"),
    )
    quotation.text_quality = TextQuality(
        assessed=True,
        suspicious_characters=3,
        currency_status="identified",
        currency_code="INR",
        currency_source="stated_code",
        message="3 unreadable character(s) were found.",
    )

    result = validate_quotation(quotation)

    assert result.status == "needs_review"
    assert any(issue.startswith("Extraction uncertainty:") for issue in result.issues)
    assert quotation.totals.grand_total == Decimal("100300")
    assert not any("Subtotal plus tax total" in issue for issue in result.issues)


def test_missing_unit_needs_review_for_a_non_arithmetic_reason():
    quotation = make(
        subtotal=Decimal("85000"),
        tax_total=Decimal("15300"),
        grand_total=Decimal("100300"),
        tax_rate=Decimal("18"),
    )
    quotation.items[0].unit = None

    result = validate_quotation(quotation)

    assert result.status == "needs_review"
    assert any("unit is unknown" in issue for issue in result.issues)
    assert not any("Subtotal plus tax total" in issue for issue in result.issues)


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