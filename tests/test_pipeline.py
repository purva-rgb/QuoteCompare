
from decimal import Decimal

import pytest

from engine.pipeline import (
    process_extracted_quotation,
    process_quotation_text,
)
from engine.schemas import (
    Quotation,
    QuotationItem,
    QuotationTotals,
    Vendor,
)


def sample_quotation():
    # Line total is 10 × 500 = 5000, and the line states a zero tax rate.
    subtotal = Decimal("5000")
    tax_total = Decimal("0")
    return Quotation(
        vendor=Vendor(
            name="ABC Electricals",
            quotation_number="Q-104",
            currency="INR",
        ),
        items=[
            QuotationItem(
                item_id="item_001",
                original_description="Copper cable 2.5 sq mm",
                quantity=Decimal("10"),
                unit="meters",
                unit_price=Decimal("500"),
                discount=Decimal("0"),
                tax_rate=Decimal("0"),
                line_total=subtotal,
            )
        ],
        totals=QuotationTotals(
            subtotal=subtotal,
            tax_total=tax_total,
            grand_total=subtotal + tax_total,
        ),
    )


def test_pipeline_normalizes_and_validates():
    result = process_extracted_quotation(sample_quotation())

    assert result.items[0].unit == "m"
    assert result.items[0].normalized_name == (
        "copper cable 2.5 sq mm"
    )
    assert result.validation.status == "valid"


def test_pipeline_does_not_mutate_input():
    original = sample_quotation()

    process_extracted_quotation(original)

    assert original.items[0].unit == "meters"
    assert original.items[0].normalized_name is None


def test_empty_pdf_text_is_rejected(monkeypatch):
    def should_not_run(_text):
        pytest.fail("Ollama must not run for empty text.")

    monkeypatch.setattr(
        "engine.pipeline.extract_quotation",
        should_not_run,
    )

    with pytest.raises(ValueError, match="no text"):
        process_quotation_text("   ")


def test_pipeline_rejects_wrong_input_type():
    with pytest.raises(TypeError):
        process_extracted_quotation({"items": []})


def test_missing_tax_requires_review():
    quotation = sample_quotation()
    quotation.items[0].tax_rate = None

    result = process_extracted_quotation(quotation)

    assert result.validation.status == "needs_review"
    assert any(
        "tax rate is unknown" in issue
        for issue in result.validation.issues
    )
