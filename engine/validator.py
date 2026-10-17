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

    totals = quotation.totals

    # A tax rate stated once for the whole quotation (for example
    # "GST (18%)") covers line items that do not carry their own rate.
    bill_tax_rate = totals.tax_rate

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

        if item.tax_rate is None and bill_tax_rate is None:
            issues.append(f"{label}: tax rate is unknown.")

        if not item.unit:
            issues.append(f"{label}: unit is unknown.")

    # Reconcile the extracted line totals with the stated subtotal.
    complete_totals = all(
        item.line_total is not None for item in quotation.items
    )

    if totals.subtotal is not None and complete_totals:
        calculated = money(
            sum(
                (item.line_total for item in quotation.items),
                start=Decimal("0"),
            )
        )

        if abs(calculated - money(totals.subtotal)) > tolerance:
            issues.append(
                f"Sum of line totals is {calculated}, but "
                f"the stated subtotal is "
                f"{money(totals.subtotal)}."
            )

    if totals.grand_total is not None and totals.subtotal is None:
        issues.append(
            "Grand total is present, but subtotal is unknown."
        )

    # A quotation that states a subtotal but yields no tax total or grand
    # total is an incomplete extraction, not a valid one.
    if totals.subtotal is not None:
        if totals.tax_total is None:
            issues.append("Tax total is unknown.")

        if totals.grand_total is None:
            issues.append("Grand total is unknown.")

    # Consistency checks between figures the vendor stated. These flag
    # differences for review; they never replace a stated value.
    if (
        bill_tax_rate is not None
        and totals.subtotal is not None
        and totals.tax_total is not None
    ):
        expected_tax = money(
            totals.subtotal * bill_tax_rate / Decimal("100")
        )

        if abs(expected_tax - money(totals.tax_total)) > tolerance:
            issues.append(
                f"A tax rate of {bill_tax_rate}% on the subtotal gives "
                f"{expected_tax}, but the stated tax total is "
                f"{money(totals.tax_total)}. Other charges or taxable-"
                "amount rules may explain the difference."
            )

    if (
        totals.subtotal is not None
        and totals.tax_total is not None
        and totals.grand_total is not None
    ):
        expected_grand = money(totals.subtotal) + money(totals.tax_total)

        if abs(expected_grand - money(totals.grand_total)) > tolerance:
            issues.append(
                f"Subtotal plus tax total is {expected_grand}, but the "
                f"stated grand total is {money(totals.grand_total)}. "
                "Freight, rounding or other charges may explain the "
                "difference."
            )

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