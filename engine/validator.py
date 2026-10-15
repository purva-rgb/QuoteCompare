
from decimal import Decimal, ROUND_HALF_UP

from engine.schemas import Quotation, ValidationResult


MONEY_PLACES = Decimal("0.01")


def money(value: Decimal) -> Decimal:
    return value.quantize(MONEY_PLACES, rounding=ROUND_HALF_UP)


def validate_quotation(
    quotation: Quotation,
    tolerance: Decimal = Decimal("0.02"),
) -> ValidationResult:
    issues: list[str] = []
    has_error = False

    if not quotation.items:
        issues.append("No line items were extracted.")
        has_error = True

    if not quotation.vendor.name:
        issues.append("Vendor name is missing.")

    # Validate individual line totals.
    for item in quotation.items:
        label = item.item_id

        if not item.original_description.strip():
            issues.append(f"{label}: description is empty.")
            has_error = True

        if item.quantity is None:
            issues.append(f"{label}: quantity is unknown.")

        if item.unit_price is None:
            issues.append(f"{label}: unit price is unknown.")

        if item.line_total is None:
            issues.append(f"{label}: line total is unknown.")

        if (
            item.quantity is not None
            and item.unit_price is not None
            and item.line_total is not None
        ):
            # A basic expected value only. Complex tax, discount,
            # pack-size and rounding rules need separate handling.
            expected = money(item.quantity * item.unit_price)

            if abs(expected - money(item.line_total)) > tolerance:
                issues.append(
                    f"{label}: quantity × unit price is "
                    f"{expected}, but extracted line total is "
                    f"{money(item.line_total)}. "
                    "Discounts or other pricing rules may explain "
                    "the difference."
                )

        if item.tax_rate is None:
            issues.append(f"{label}: tax rate is unknown.")

        if not item.unit:
            issues.append(f"{label}: unit is unknown.")

    # Reconcile the extracted line totals with the stated subtotal.
    complete_totals = all(
        item.line_total is not None for item in quotation.items
    )

    if quotation.totals.subtotal is not None and complete_totals:
        calculated = money(
            sum(
                (item.line_total for item in quotation.items),
                start=Decimal("0"),
            )
        )

        if abs(
            calculated - money(quotation.totals.subtotal)
        ) > tolerance:
            issues.append(
                f"Sum of line totals is {calculated}, but "
                f"the stated subtotal is "
                f"{money(quotation.totals.subtotal)}."
            )

    if (
        quotation.totals.grand_total is not None
        and quotation.totals.subtotal is None
    ):
        issues.append(
            "Grand total is present, but subtotal is unknown."
        )

    # Do not compute taxes or grand totals until tax treatment,
    # discounts, freight and other charges are understood.
    if not issues and not has_error:
        result = ValidationResult(
            status="valid",
            issues=[],
        )
    else:
        result = ValidationResult(
            status="invalid" if has_error else "needs_review",
            issues=issues,
        )

    quotation.validation = result
    return result
