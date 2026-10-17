"""
Rank comparable supplier quotations and explain the result.

The recommender never decides silently. It ranks quotations by their stated
grand total, but only after eligibility checks, and it reports near-ties as
ties instead of naming a winner on a tiny margin. Scope differences between
quotations are reported as cautions so a low total is never shown without
context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from typing import Iterable

from engine.matcher import find_candidates
from engine.schemas import Quotation, QuotationItem

TOTALS_TOLERANCE = Decimal("0.02")
DEFAULT_TIE_MARGIN_PERCENT = Decimal("2")

# Name similarity used to decide whether two quotations cover the same
# item. It is only a heuristic for reporting scope differences as cautions:
# too low and unrelated items look alike, too high and the same item
# worded differently ("Venue rental" and "Venue and hall setup") looks
# missing. Confirmed matches will replace it.
DEFAULT_SCOPE_THRESHOLD = 0.55

STATUS_CLEAR_WINNER = "clear_winner"
STATUS_EFFECTIVELY_TIED = "effectively_tied"
STATUS_SINGLE_ELIGIBLE = "single_eligible"
STATUS_NONE = "none"


@dataclass(frozen=True)
class QuotationEntry:
    """One quotation to compare, with a unique display label."""

    label: str
    quotation: Quotation


@dataclass
class Assessment:
    label: str
    grand_total: Decimal | None
    currency: str | None
    eligible: bool
    rank: int | None = None
    difference_from_lowest: Decimal | None = None
    percent_above_lowest: Decimal | None = None
    within_tie_band: bool = False
    ineligible_reasons: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    extra_items: list[str] = field(default_factory=list)
    missing_items: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    status: str
    headline: str
    tie_margin_percent: Decimal
    assessments: list[Assessment]
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _percent(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _amount(value: Decimal, currency: str | None) -> str:
    prefix = f"{currency} " if currency else ""
    return f"{prefix}{_money(value):,.2f}"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def _ineligible_reasons(quotation: Quotation) -> list[str]:
    reasons: list[str] = []
    totals = quotation.totals

    if quotation.validation.status == "invalid":
        reasons.append("Validation status is invalid.")

    grand = totals.grand_total
    if grand is None:
        reasons.append("No grand total was extracted.")
    elif grand <= 0:
        reasons.append("The grand total is zero.")

    if (
        totals.subtotal is not None
        and totals.tax_total is not None
        and grand is not None
    ):
        expected = _money(totals.subtotal) + _money(totals.tax_total)
        if abs(expected - _money(grand)) > TOTALS_TOLERANCE:
            reasons.append(
                "The stated subtotal plus tax does not equal the stated "
                "grand total."
            )

    return reasons


def _has_candidate(
    item: QuotationItem,
    quotation: Quotation,
    threshold: float,
) -> bool:
    try:
        return bool(find_candidates(item, quotation, threshold))
    except Exception:
        # If matching itself fails, do not claim a scope difference.
        return True


def _scope_differences(
    entries: list[QuotationEntry],
    threshold: float,
) -> dict[str, tuple[list[str], list[str]]]:
    """For each entry: (items only it has, items others have that it lacks)."""
    result: dict[str, tuple[list[str], list[str]]] = {
        entry.label: ([], []) for entry in entries
    }

    if len(entries) < 2:
        return result

    for entry in entries:
        extra, missing = result[entry.label]
        others = [other for other in entries if other is not entry]

        for item in entry.quotation.items:
            if not any(
                _has_candidate(item, other.quotation, threshold)
                for other in others
            ):
                extra.append(item.original_description)

        seen: set[str] = set()
        for other in others:
            for item in other.quotation.items:
                key = item.original_description.casefold()
                if key in seen:
                    continue
                if not _has_candidate(item, entry.quotation, threshold):
                    seen.add(key)
                    missing.append(item.original_description)

    return result


# ---------------------------------------------------------
# Public API
# ---------------------------------------------------------

def recommend(
    entries: Iterable[QuotationEntry],
    tie_margin_percent: Decimal | int | float | str = (
        DEFAULT_TIE_MARGIN_PERCENT
    ),
    scope_threshold: float = DEFAULT_SCOPE_THRESHOLD,
) -> Recommendation:
    """
    Rank quotations by stated grand total and explain the outcome.

    Quotations that fail eligibility checks are listed but not ranked.
    Totals within tie_margin_percent of the lowest are reported as
    effectively tied rather than as a win.
    """
    entries = list(entries)

    if not entries:
        raise ValueError("At least one quotation is required.")

    for entry in entries:
        if not isinstance(entry.quotation, Quotation):
            raise TypeError("Each entry must hold a Quotation.")

    if len({entry.label for entry in entries}) != len(entries):
        raise ValueError("Quotation labels must be unique.")

    margin = Decimal(str(tie_margin_percent))
    if margin < 0:
        raise ValueError("tie_margin_percent cannot be negative.")

    scope = _scope_differences(entries, scope_threshold)

    assessments: list[Assessment] = []
    for entry in entries:
        quotation = entry.quotation
        currency = (quotation.vendor.currency or "").strip().upper() or None
        reasons = _ineligible_reasons(quotation)
        extra, missing = scope[entry.label]

        assessment = Assessment(
            label=entry.label,
            grand_total=quotation.totals.grand_total,
            currency=currency,
            eligible=not reasons,
            ineligible_reasons=reasons,
            extra_items=list(extra),
            missing_items=list(missing),
        )

        issue_count = len(quotation.validation.issues)
        if quotation.validation.status == "needs_review":
            detail = (
                f" ({_plural(issue_count, 'issue')})" if issue_count else ""
            )
            assessment.cautions.append(f"Validation needs review{detail}.")

        if currency is None:
            assessment.cautions.append("Currency is not stated.")

        if extra:
            assessment.cautions.append(
                "Includes items the other quotations do not: "
                + ", ".join(extra) + "."
            )

        if missing:
            assessment.cautions.append(
                "Lacks items the other quotations have: "
                + ", ".join(missing) + "."
            )

        assessments.append(assessment)

    notes = [
        "Ranking uses each quotation's stated grand total only.",
        "Scope differences are approximate until item matches are "
        "confirmed, so check them before relying on a ranking.",
    ]

    # Mixed currencies cannot be ranked without conversion.
    currencies = {a.currency for a in assessments if a.currency}
    if len(currencies) > 1:
        for assessment in assessments:
            assessment.eligible = False
            assessment.ineligible_reasons.append(
                "Quotations use different currencies "
                f"({', '.join(sorted(currencies))})."
            )

        return Recommendation(
            status=STATUS_NONE,
            headline=(
                "No recommendation: the quotations use different "
                f"currencies ({', '.join(sorted(currencies))}). "
                "Convert them before comparing."
            ),
            tie_margin_percent=margin,
            assessments=assessments,
            notes=notes,
        )

    eligible = sorted(
        (a for a in assessments if a.eligible),
        key=lambda a: a.grand_total,  # type: ignore[arg-type, return-value]
    )
    ineligible = [a for a in assessments if not a.eligible]

    if not eligible:
        return Recommendation(
            status=STATUS_NONE,
            headline=(
                "No recommendation: no quotation passed the eligibility "
                "checks. See the reasons for each."
            ),
            tie_margin_percent=margin,
            assessments=ineligible,
            notes=notes,
        )

    lowest = eligible[0].grand_total
    assert lowest is not None
    band_limit = lowest * (Decimal("1") + margin / Decimal("100"))

    for assessment in eligible:
        total = assessment.grand_total
        assert total is not None
        assessment.rank = 1 + sum(
            1 for other in eligible
            if other.grand_total is not None and other.grand_total < total
        )
        assessment.difference_from_lowest = _money(total - lowest)
        assessment.percent_above_lowest = _percent(
            (total - lowest) / lowest * Decimal("100")
        )
        assessment.within_tie_band = total <= band_limit

    best = eligible[0]
    contenders = [a for a in eligible if a.within_tie_band]

    if len(eligible) == 1:
        status = STATUS_SINGLE_ELIGIBLE
        headline = (
            f"Only {best.label} could be ranked, so there is nothing to "
            "compare it with. Check why the others were excluded."
        )
    elif len(contenders) > 1:
        status = STATUS_EFFECTIVELY_TIED
        headline = (
            f"{len(contenders)} quotations are within {margin}% of the "
            "lowest total: "
            + ", ".join(a.label for a in contenders)
            + f". {best.label} is lowest at "
            f"{_amount(lowest, best.currency)}, but the difference is too "
            "small to decide on. Compare scope and terms."
        )
    else:
        second = eligible[1]
        status = STATUS_CLEAR_WINNER
        headline = (
            f"{best.label} has the lowest total at "
            f"{_amount(lowest, best.currency)}. The next option, "
            f"{second.label}, is "
            f"{_amount(second.difference_from_lowest or Decimal('0'), best.currency)}"
            f" ({second.percent_above_lowest}%) higher."
        )

    return Recommendation(
        status=status,
        headline=headline,
        tie_margin_percent=margin,
        assessments=eligible + ineligible,
        notes=notes,
    )