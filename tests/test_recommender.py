from decimal import Decimal

import pytest

from engine.recommender import (
    STATUS_CLEAR_WINNER,
    STATUS_EFFECTIVELY_TIED,
    STATUS_NONE,
    STATUS_SINGLE_ELIGIBLE,
    QuotationEntry,
    recommend,
)
from engine.schemas import (
    Quotation,
    QuotationItem,
    QuotationTotals,
    ValidationResult,
    Vendor,
)


def make_quotation(
    name,
    items,
    subtotal,
    tax_total,
    grand_total,
    currency="INR",
    status="valid",
):
    return Quotation(
        vendor=Vendor(name=name, currency=currency),
        items=[
            QuotationItem(
                item_id=f"item_{index}",
                original_description=description,
                quantity=1,
                unit="package",
                unit_price=price,
                line_total=price,
            )
            for index, (description, price) in enumerate(items, start=1)
        ],
        totals=QuotationTotals(
            subtotal=subtotal,
            tax_total=tax_total,
            grand_total=grand_total,
        ),
        validation=ValidationResult(status=status),
    )


@pytest.fixture
def bluepeak():
    return make_quotation(
        "BluePeak",
        [
            ("Venue and hall setup", 45000),
            ("Catering", 85000),
            ("Audio/visual equipment", 18000),
            ("Stage and basic decoration", 22000),
            ("Photography", 12000),
        ],
        182000, 32760, 214760,
    )


@pytest.fixture
def crescent():
    return make_quotation(
        "Crescent",
        [
            ("Venue rental", 52000),
            ("Catering", 78000),
            ("Audio/visual equipment", 15000),
            ("Stage and decoration", 18000),
            ("Photography", 10000),
            ("Delivery and logistics", 8500),
        ],
        181500, 32670, 214170,
    )


@pytest.fixture
def silverfern():
    return make_quotation(
        "SilverFern",
        [
            ("Venue and seating", 48000),
            ("Catering", 82000),
            ("Audio/visual equipment", 22000),
            ("Stage and premium decoration", 28000),
            ("Photography", 14000),
        ],
        194000, 34920, 228920,
    )


def entries(**quotations):
    return [QuotationEntry(label, q) for label, q in quotations.items()]


def by_label(recommendation):
    return {a.label: a for a in recommendation.assessments}


def test_sample_quotations_are_a_near_tie(bluepeak, crescent, silverfern):
    result = recommend(
        entries(BluePeak=bluepeak, Crescent=crescent, SilverFern=silverfern)
    )

    assert result.status == STATUS_EFFECTIVELY_TIED
    assessments = by_label(result)
    assert assessments["Crescent"].rank == 1
    assert assessments["BluePeak"].rank == 2
    assert assessments["SilverFern"].rank == 3
    assert assessments["BluePeak"].difference_from_lowest == Decimal("590.00")
    assert assessments["BluePeak"].within_tie_band is True
    assert assessments["SilverFern"].within_tie_band is False
    assert "Crescent" in result.headline and "BluePeak" in result.headline


def test_narrow_margin_gives_clear_winner(bluepeak, crescent, silverfern):
    result = recommend(
        entries(BluePeak=bluepeak, Crescent=crescent, SilverFern=silverfern),
        tie_margin_percent=Decimal("0.1"),
    )

    assert result.status == STATUS_CLEAR_WINNER
    assert result.assessments[0].label == "Crescent"
    assert "lowest total" in result.headline


def test_results_are_sorted_by_total(bluepeak, crescent, silverfern):
    result = recommend(
        entries(SilverFern=silverfern, BluePeak=bluepeak, Crescent=crescent)
    )

    assert [a.label for a in result.assessments] == [
        "Crescent", "BluePeak", "SilverFern",
    ]


def test_scope_differences_are_reported(bluepeak, crescent, silverfern):
    result = recommend(
        entries(BluePeak=bluepeak, Crescent=crescent, SilverFern=silverfern)
    )
    assessments = by_label(result)

    assert assessments["Crescent"].extra_items == ["Delivery and logistics"]
    assert "Delivery and logistics" in assessments["BluePeak"].missing_items
    assert any(
        "Includes items" in caution
        for caution in assessments["Crescent"].cautions
    )


def test_invalid_validation_is_excluded(bluepeak, crescent):
    crescent.validation = ValidationResult(status="invalid")

    result = recommend(entries(BluePeak=bluepeak, Crescent=crescent))

    assessments = by_label(result)
    assert assessments["Crescent"].eligible is False
    assert assessments["Crescent"].rank is None
    assert assessments["BluePeak"].rank == 1
    assert result.status == STATUS_SINGLE_ELIGIBLE
    # Excluded quotations are listed after the ranked ones.
    assert result.assessments[-1].label == "Crescent"


def test_missing_grand_total_is_excluded(bluepeak, crescent):
    crescent.totals.grand_total = None

    result = recommend(entries(BluePeak=bluepeak, Crescent=crescent))

    reasons = by_label(result)["Crescent"].ineligible_reasons
    assert any("grand total" in reason for reason in reasons)


def test_inconsistent_totals_are_excluded(bluepeak, crescent):
    crescent.totals.grand_total = Decimal("200000")

    result = recommend(entries(BluePeak=bluepeak, Crescent=crescent))

    reasons = by_label(result)["Crescent"].ineligible_reasons
    assert any("does not equal" in reason for reason in reasons)


def test_mixed_currencies_give_no_recommendation(bluepeak, crescent):
    crescent.vendor.currency = "USD"

    result = recommend(entries(BluePeak=bluepeak, Crescent=crescent))

    assert result.status == STATUS_NONE
    assert all(not a.eligible for a in result.assessments)
    assert "currencies" in result.headline


def test_no_eligible_quotation(bluepeak, crescent):
    bluepeak.validation = ValidationResult(status="invalid")
    crescent.validation = ValidationResult(status="invalid")

    result = recommend(entries(BluePeak=bluepeak, Crescent=crescent))

    assert result.status == STATUS_NONE
    assert all(a.rank is None for a in result.assessments)


def test_needs_review_is_ranked_with_a_caution(bluepeak, crescent):
    crescent.validation = ValidationResult(
        status="needs_review", issues=["a", "b"]
    )

    result = recommend(entries(BluePeak=bluepeak, Crescent=crescent))

    crescent_assessment = by_label(result)["Crescent"]
    assert crescent_assessment.eligible is True
    assert any(
        "2 issues" in caution for caution in crescent_assessment.cautions
    )


def test_equal_totals_share_a_rank(bluepeak, crescent):
    crescent.totals.subtotal = bluepeak.totals.subtotal
    crescent.totals.tax_total = bluepeak.totals.tax_total
    crescent.totals.grand_total = bluepeak.totals.grand_total

    result = recommend(entries(BluePeak=bluepeak, Crescent=crescent))

    assert [a.rank for a in result.assessments] == [1, 1]
    assert result.status == STATUS_EFFECTIVELY_TIED


def test_single_quotation(bluepeak):
    result = recommend(entries(BluePeak=bluepeak))

    assert result.status == STATUS_SINGLE_ELIGIBLE
    assert result.assessments[0].extra_items == []


def test_missing_currency_adds_caution(bluepeak, crescent):
    bluepeak.vendor.currency = None

    result = recommend(entries(BluePeak=bluepeak, Crescent=crescent))

    assert any(
        "Currency is not stated" in caution
        for caution in by_label(result)["BluePeak"].cautions
    )


def test_input_validation(bluepeak):
    with pytest.raises(ValueError):
        recommend([])

    with pytest.raises(ValueError):
        recommend(entries(BluePeak=bluepeak), tie_margin_percent=-1)

    with pytest.raises(ValueError):
        recommend([
            QuotationEntry("Same", bluepeak),
            QuotationEntry("Same", bluepeak),
        ])

    with pytest.raises(TypeError):
        recommend([QuotationEntry("Bad", "not a quotation")])