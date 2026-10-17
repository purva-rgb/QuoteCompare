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