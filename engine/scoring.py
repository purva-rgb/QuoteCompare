"""
Combine price with manually entered service-quality ratings.

Quotation PDFs do not contain service-quality information, and fetching
online reviews would break the app's local-only promise and is unreliable,
so ratings are entered by the user. This module only does the arithmetic:
it never invents a rating, and a supplier without ratings is left out of
the weighted score instead of being given an average.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Mapping

CRITERIA = (
    "Reliability and track record",
    "Communication and responsiveness",
    "Quality of service or work",
    "References and reviews",
)

MIN_RATING = 1
MAX_RATING = 5


@dataclass(frozen=True)
class ScoreInput:
    label: str
    grand_total: Decimal | None
    eligible: bool
    ratings: Mapping[str, Any]


@dataclass
class ScoredSupplier:
    label: str
    grand_total: Decimal | None
    quality_average: float | None = None
    price_score: float | None = None
    quality_score: float | None = None
    combined_score: float | None = None
    rank: int | None = None
    excluded_reason: str | None = None


def validate_rating(value: Any) -> int | None:
    """
    Return a whole-number rating from 1 to 5, or None if nothing was
    entered (None or NaN). Raises ValueError for anything else.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError("Ratings must be whole numbers from 1 to 5.")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Ratings must be whole numbers from {MIN_RATING} to "
            f"{MAX_RATING}, got {value!r}."
        ) from exc

    if number != number:  # NaN, which spreadsheets use for an empty cell
        return None

    if number != int(number) or not MIN_RATING <= number <= MAX_RATING:
        raise ValueError(
            f"Ratings must be whole numbers from {MIN_RATING} to "
            f"{MAX_RATING}, got {value!r}."
        )

    return int(number)


def average_rating(ratings: Mapping[str, Any]) -> float | None:
    """Mean of the criteria that were rated; None if none were."""
    values = [
        rating
        for rating in (validate_rating(v) for v in ratings.values())
        if rating is not None
    ]

    if not values:
        return None

    return sum(values) / len(values)


def score_suppliers(
    inputs: Iterable[ScoreInput],
    price_weight_percent: float | int | Decimal,
) -> list[ScoredSupplier]:
    """
    Score each eligible, rated supplier from 0 to 100.

    Price score: the lowest eligible total scores 100 and others score
    lowest / total * 100. Quality score: the average rating mapped from
    1-5 onto 0-100. The combined score weights the two. Suppliers that are
    not eligible or have no ratings are listed after the ranked ones.
    """
    inputs = list(inputs)
    weight = float(price_weight_percent)

    if not 0 <= weight <= 100:
        raise ValueError("price_weight_percent must be between 0 and 100.")

    price_weight = weight / 100
    quality_weight = 1 - price_weight

    totals = [
        item.grand_total
        for item in inputs
        if item.eligible and item.grand_total is not None
        and item.grand_total > 0
    ]
    lowest = min(totals) if totals else None

    results: list[ScoredSupplier] = []

    for item in inputs:
        supplier = ScoredSupplier(
            label=item.label,
            grand_total=item.grand_total,
            quality_average=average_rating(item.ratings),
        )

        if (
            not item.eligible
            or lowest is None
            or item.grand_total is None
            or item.grand_total <= 0
        ):
            supplier.excluded_reason = "Not eligible for price ranking."
        elif supplier.quality_average is None:
            supplier.excluded_reason = "No quality ratings entered."
        else:
            supplier.price_score = float(
                lowest / item.grand_total * Decimal("100")
            )
            supplier.quality_score = (
                (supplier.quality_average - MIN_RATING)
                / (MAX_RATING - MIN_RATING) * 100
            )
            supplier.combined_score = (
                price_weight * supplier.price_score
                + quality_weight * supplier.quality_score
            )

        results.append(supplier)

    scored = sorted(
        (r for r in results if r.combined_score is not None),
        key=lambda r: -(r.combined_score or 0.0),
    )

    for supplier in scored:
        current = round(supplier.combined_score or 0.0, 4)
        supplier.rank = 1 + sum(
            1 for other in scored
            if round(other.combined_score or 0.0, 4) > current
        )

    excluded = [r for r in results if r.combined_score is None]
    return scored + excluded


def winner_ranges(
    inputs: Iterable[ScoreInput],
    step: int = 5,
) -> list[tuple[int, int, tuple[str, ...]]]:
    """
    Which supplier leads as the weight on quality goes from 0% to 100%.

    Returns (from_percent, to_percent, winners) segments, checked every
    `step` percent. A single segment means the leader never changes.
    """
    if step <= 0 or 100 % step:
        raise ValueError("step must be a positive divisor of 100.")

    inputs = list(inputs)
    ranges: list[tuple[int, int, tuple[str, ...]]] = []

    for quality in range(0, 101, step):
        ranked = score_suppliers(inputs, 100 - quality)
        winners = tuple(sorted(r.label for r in ranked if r.rank == 1))

        if not winners:
            return []

        if ranges and ranges[-1][2] == winners:
            ranges[-1] = (ranges[-1][0], quality, winners)
        else:
            ranges.append((quality, quality, winners))

    return ranges


# ---------------------------------------------------------
# Public-rating score used by the simplified comparison screen
# ---------------------------------------------------------

DEFAULT_PRICE_WEIGHT_PERCENT = 70
BEST_PICK_UNAVAILABLE = "Best Pick unavailable"

_CRITICAL_MARKERS = (
    "tax",
    "grand total",
    "subtotal",
    "quotation total",
    "currenc",
    "not comparable",
    "invalid",
)


@dataclass(frozen=True)
class ComparisonInput:
    """One supplier row before scores are calculated."""

    label: str
    supplier_key: str
    grand_total: Decimal | None
    currency: str | None
    eligible: bool
    public_rating: float | None
    review_count: int | None
    warnings: tuple[str, ...] = ()
    critical: bool = False


@dataclass
class ScoredComparison:
    label: str
    supplier_key: str
    grand_total: Decimal | None
    currency: str | None
    public_rating: float | None
    review_count: int | None
    price_score: float | None
    quality_score: float | None
    combined_score: float | None
    rank: int | None
    status: str
    warnings: tuple[str, ...]
    eligible: bool
    critical: bool


@dataclass(frozen=True)
class BestPick:
    pending: bool
    headline: str
    explanation: str
    supplier_key: str | None = None
    label: str | None = None
    combined_score: float | None = None
    grand_total: Decimal | None = None
    currency: str | None = None
    public_rating: float | None = None
    review_count: int | None = None
    warnings: tuple[str, ...] = ()


@dataclass
class Ranking:
    rows: list[ScoredComparison]
    price_only: bool
    price_only_reason: str
    best: BestPick
    price_weight_percent: float


def is_advisory_extraction_warning(issue: str) -> bool:
    """Unreadable glyphs are a review note, not a failed financial check.

    The validator keeps the original message. A currency that is missing,
    unconfirmed, or in conflict is not this kind of note.
    """
    text = issue.casefold()
    return (
        "extraction uncertainty" in text
        and "unreadable character" in text
    )


def is_critical_issue(issue: str) -> bool:
    """Price, tax, currency, or comparability problems block a Best Pick."""
    if is_advisory_extraction_warning(issue):
        return False
    text = issue.casefold()
    return any(marker in text for marker in _CRITICAL_MARKERS)


def quality_score_from_public_rating(public_rating: float | None) -> float | None:
    """Map a public rating of 1–5 onto 0–100. Missing stays missing."""
    if public_rating is None or isinstance(public_rating, bool):
        return None
    try:
        number = float(public_rating)
    except (TypeError, ValueError):
        return None
    if number != number or not 1 <= number <= 5:
        return None
    return number / 5 * 100


def price_score_from_totals(lowest: Decimal, total: Decimal) -> float:
    """Lowest eligible total scores 100. Others score lowest / total × 100."""
    if total <= 0:
        raise ValueError("Supplier price must be greater than zero.")
    return float(lowest / total * Decimal("100"))


def _price_weight(price_weight_percent: float | int | Decimal) -> float:
    weight = float(price_weight_percent)
    if not 0 <= weight <= 100:
        raise ValueError("Price weight must be between 0 and 100.")
    return weight


def rank_comparisons(
    inputs: Iterable[ComparisonInput],
    price_weight_percent: float | int | Decimal = DEFAULT_PRICE_WEIGHT_PERCENT,
) -> Ranking:
    """
    Rank suppliers for the comparison table.

    A combined score exists only when both the price score and the
    public-rating quality score exist. A missing score is not replaced
    with zero and the other score is not given its weight. Review count
    is stored for display and is not part of either formula.
    """
    weight = _price_weight(price_weight_percent)
    quality_weight = 100 - weight
    items = list(inputs)

    currencies = {
        (item.currency or "").strip().upper()
        for item in items
        if (item.currency or "").strip()
    }
    mixed = len(currencies) > 1

    prepared: list[dict[str, Any]] = []
    for item in items:
        warnings = list(item.warnings)
        eligible = bool(item.eligible)
        critical = bool(item.critical)
        currency = (item.currency or "").strip().upper() or None
        if mixed:
            eligible = False
            critical = True
            note = (
                "Quotations use different currencies "
                f"({', '.join(sorted(currencies))}) and cannot be compared."
            )
            if note not in warnings:
                warnings.append(note)

        total = item.grand_total
        can_price = (
            eligible
            and total is not None
            and total > 0
        )
        prepared.append({
            "item": item,
            "warnings": tuple(warnings),
            "eligible": eligible,
            "critical": critical,
            "currency": currency,
            "can_price": can_price,
            "total": total,
        })

    priced_totals = [
        row["total"] for row in prepared
        if row["can_price"] and row["total"] is not None
    ]
    lowest = min(priced_totals) if priced_totals else None

    scored: list[ScoredComparison] = []
    for row in prepared:
        item = row["item"]
        quality = quality_score_from_public_rating(item.public_rating)
        price = None
        if row["can_price"] and lowest is not None and row["total"] is not None:
            price = price_score_from_totals(lowest, row["total"])
        combined = None
        if price is not None and quality is not None:
            combined = (price * weight / 100) + (quality * quality_weight / 100)

        if not row["eligible"]:
            status = row["warnings"][0] if row["warnings"] else (
                "Quotation total is not comparable."
            )
        elif row["critical"]:
            status = row["warnings"][0] if row["warnings"] else (
                "Needs review before a Best Pick."
            )
        elif quality is None:
            status = "Combined score unavailable — public rating not provided."
        else:
            status = "Ranked"

        scored.append(ScoredComparison(
            label=item.label,
            supplier_key=item.supplier_key,
            grand_total=item.grand_total,
            currency=row["currency"],
            public_rating=item.public_rating,
            review_count=item.review_count,
            price_score=price,
            quality_score=quality,
            combined_score=combined,
            rank=None,
            status=status,
            warnings=row["warnings"],
            eligible=row["eligible"],
            critical=row["critical"],
        ))

    price_only = not any(row.combined_score is not None for row in scored)
    if price_only:
        for row in scored:
            if row.price_score is not None:
                row.status = "Price-only rank"

    combined_rows = [row for row in scored if row.combined_score is not None]
    for row in combined_rows:
        current = round(row.combined_score or 0.0, 4)
        row.rank = 1 + sum(
            1 for other in combined_rows
            if round(other.combined_score or 0.0, 4) > current
        )

    priced_rows = [row for row in scored if row.price_score is not None]
    price_ranks: dict[str, int] = {}
    for row in priced_rows:
        total = row.grand_total
        price_ranks[row.supplier_key] = 1 + sum(
            1 for other in priced_rows
            if other.grand_total is not None and total is not None
            and other.grand_total < total
        )
    if price_only:
        for row in scored:
            row.rank = price_ranks.get(row.supplier_key)

    def sort_key(row: ScoredComparison) -> tuple:
        if row.rank is None:
            return (1, 0, row.label)
        return (0, row.rank, row.label)

    scored.sort(key=sort_key)

    if not price_only:
        reason = ""
    elif priced_rows:
        reason = (
            "Combined scores are unavailable because a public rating is "
            "missing or a quotation total is not comparable. The rank "
            "column is a price-only ranking of comparable quotation "
            "totals. It is not a combined score."
        )
    else:
        reason = (
            "No comparable quotation total is available, so there is no "
            "price-only ranking."
        )

    qualified = [
        row for row in scored
        if row.combined_score is not None and row.eligible and not row.critical
    ]
    if not qualified:
        missing: list[str] = []
        if not any(row.public_rating is not None for row in scored):
            missing.append("A public rating is required for a quality score.")
        if not priced_rows:
            missing.append(
                "A comparable quotation total is required for a price score."
            )
        elif any(
            row.eligible and not row.critical and row.public_rating is None
            for row in scored
        ):
            missing.append(
                "Each candidate needs a public rating. A missing rating "
                "is not scored as zero."
            )
        notes = []
        for row in scored:
            if row.critical and row.warnings:
                notes.append(f"{row.label}: {row.warnings[0]}")
        if notes:
            missing.append("Critical review items: " + "; ".join(notes) + ".")
        best = BestPick(
            pending=True,
            headline=BEST_PICK_UNAVAILABLE,
            explanation=" ".join(missing) or (
                "A complete price score and quality score are required."
            ),
        )
    else:
        top = max(row.combined_score or 0.0 for row in qualified)
        leaders = [
            row for row in qualified
            if round(row.combined_score or 0.0, 4) == round(top, 4)
        ]
        if len(leaders) > 1:
            names = ", ".join(row.label for row in leaders)
            best = BestPick(
                pending=True,
                headline=BEST_PICK_UNAVAILABLE,
                explanation=(
                    f"{names} share the highest combined score, so no "
                    "single supplier is selected."
                ),
            )
        else:
            winner = leaders[0]
            warning_text = ""
            if winner.warnings:
                warning_text = " Review notes: " + "; ".join(winner.warnings) + "."
            blocked = [
                row.label for row in scored
                if row.critical and row.combined_score is not None
                and row.combined_score > (winner.combined_score or 0)
            ]
            if blocked:
                warning_text += (
                    " A higher combined score was not selected because of "
                    "a critical review item: " + ", ".join(blocked) + "."
                )
            best = BestPick(
                pending=False,
                headline=winner.label,
                explanation=(
                    f"{winner.label} has the highest combined score among "
                    "suppliers that can be confirmed. "
                    f"The price score is {winner.price_score:.1f} and the "
                    f"quality score is {winner.quality_score:.1f}, from a "
                    f"public rating of {winner.public_rating}. "
                    f"Price is weighted {weight:.0f}% and quality "
                    f"{quality_weight:.0f}%. The review count is shown "
                    "for context and is not part of the score."
                    + warning_text
                ),
                supplier_key=winner.supplier_key,
                label=winner.label,
                combined_score=winner.combined_score,
                grand_total=winner.grand_total,
                currency=winner.currency,
                public_rating=winner.public_rating,
                review_count=winner.review_count,
                warnings=winner.warnings,
            )

    return Ranking(
        rows=scored,
        price_only=price_only,
        price_only_reason=reason,
        best=best,
        price_weight_percent=weight,
    )