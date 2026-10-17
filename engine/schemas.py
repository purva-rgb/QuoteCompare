from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Quantity(StrictModel):
    value: Decimal = Field(ge=0)
    unit: str | None = None


class Specification(StrictModel):
    value: str | Decimal | None = None
    unit: str | None = None


class Vendor(StrictModel):
    name: str | None = None
    quotation_number: str | None = None
    quotation_date: str | None = None
    currency: str | None = None


class QuotationItem(StrictModel):
    item_id: str
    original_description: str = Field(min_length=1)
    normalized_name: str | None = None
    specifications: dict[str, Specification] = Field(
        default_factory=dict
    )

    quantity: Decimal | None = Field(default=None, ge=0)
    unit: str | None = None
    unit_price: Decimal | None = Field(default=None, ge=0)
    discount: Decimal | None = Field(default=None, ge=0)
    tax_rate: Decimal | None = Field(default=None, ge=0, le=100)
    line_total: Decimal | None = Field(default=None, ge=0)

    # Values from the vendor's document, not AI-generated citations.
    source_page: int | None = Field(default=None, ge=1)

    # "extracted" is not the same as "verified".
    match_status: Literal[
        "unreviewed", "matched", "needs_review", "not_matched"
    ] = "unreviewed"

    warnings: list[str] = Field(default_factory=list)


class QuotationTotals(StrictModel):
    subtotal: Decimal | None = Field(default=None, ge=0)
    tax_total: Decimal | None = Field(default=None, ge=0)
    grand_total: Decimal | None = Field(default=None, ge=0)

    # Tax rate stated once for the whole quotation, as a percentage
    # (for example 18 for "GST (18%)"). Line items may leave tax_rate
    # empty when the document states tax only at this level.
    tax_rate: Decimal | None = Field(default=None, ge=0, le=100)


class ValidationResult(StrictModel):
    status: Literal["valid", "needs_review", "invalid"] = "needs_review"
    issues: list[str] = Field(default_factory=list)


class TextQuality(StrictModel):
    """What the extractor could honestly tell about the source text.

    `assessed` stays false until extraction inspects a real document.
    Currency fields record evidence. They are not a guess, and they are
    never written back over a number.
    """

    assessed: bool = False
    suspicious_characters: int = Field(default=0, ge=0)
    currency_status: Literal["identified", "unknown"] = "unknown"
    currency_code: str | None = None
    currency_source: Literal["stated_code", "stated_symbol", "none"] = "none"
    message: str = ""


class Quotation(StrictModel):
    vendor: Vendor
    items: list[QuotationItem] = Field(default_factory=list)
    totals: QuotationTotals = Field(default_factory=QuotationTotals)
    validation: ValidationResult = Field(
        default_factory=ValidationResult
    )
    text_quality: TextQuality = Field(default_factory=TextQuality)

    # Keep the vendor's original stated totals separately if necessary.
    extraction_notes: list[str] = Field(default_factory=list)

    @field_validator("items")
    @classmethod
    def require_items(cls, items):
        if not items:
            raise ValueError("Quotation must contain at least one item.")
        return items