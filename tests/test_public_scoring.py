from decimal import Decimal

import pytest

from engine.ratings import SupplierRef, supplier_key
from engine.scoring import (
    DEFAULT_PRICE_WEIGHT_PERCENT,
    ComparisonInput,
    is_critical_issue,
    price_score_from_totals,
    quality_score_from_public_rating,
    rank_comparisons,
)


def row(
    name,
    total,
    rating,
    reviews=10,
    key="key",
    eligible=True,
    critical=False,
    warnings=(),
    currency="INR",
):
    return ComparisonInput(
        label=name,
        supplier_key=key,
        grand_total=None if total is None else Decimal(str(total)),
        currency=currency,
        eligible=eligible,
        public_rating=rating,
        review_count=reviews,
        warnings=tuple(warnings),
        critical=critical,
    )


def by_key(ranking):
    return {item.supplier_key: item for item in ranking.rows}


def test_price_score_uses_the_lowest_eligible_total():
    assert price_score_from_totals(Decimal("100"), Decimal("100")) == 100
    assert price_score_from_totals(Decimal("100"), Decimal("200")) == 50


def test_public_rating_quality_score():
    assert quality_score_from_public_rating(5) == 100
    assert quality_score_from_public_rating(4.3) == pytest.approx(86)
    assert quality_score_from_public_rating(1) == 20
    assert quality_score_from_public_rating(None) is None


def test_default_weights_are_70_30():
    ranking = rank_comparisons([
        row("Low", 100, 5, key="low"),
        row("High", 200, 5, key="high"),
    ])

    assert ranking.price_weight_percent == DEFAULT_PRICE_WEIGHT_PERCENT == 70
    scored = by_key(ranking)
    # Low: price 100, quality 100 -> 100
    assert scored["low"].price_score == pytest.approx(100)
    assert scored["low"].quality_score == pytest.approx(100)
    assert scored["low"].combined_score == pytest.approx(100)
    # High: price 50, quality 100 -> 0.7*50 + 0.3*100 = 65
    assert scored["high"].price_score == pytest.approx(50)
    assert scored["high"].combined_score == pytest.approx(65)
    assert scored["low"].rank == 1
    assert ranking.best.pending is False
    assert ranking.best.supplier_key == "low"


def test_custom_weights_sum_to_100():
    ranking = rank_comparisons(
        [row("Low", 100, 5, key="low"), row("High", 200, 4, key="high")],
        price_weight_percent=40,
    )

    assert ranking.price_weight_percent == 40
    high = by_key(ranking)["high"]
    # price 50, quality 80 -> 0.4*50 + 0.6*80 = 68
    assert high.combined_score == pytest.approx(68)

    with pytest.raises(ValueError):
        rank_comparisons([row("Low", 100, 5)], price_weight_percent=101)


def test_missing_public_rating_does_not_invent_a_combined_score():
    ranking = rank_comparisons([
        row("Rated", 100, 4, key="rated"),
        row("Unrated", 80, None, reviews=None, key="unrated"),
    ])

    unrated = by_key(ranking)["unrated"]
    assert unrated.public_rating is None
    assert unrated.quality_score is None
    assert unrated.combined_score is None
    assert unrated.price_score == pytest.approx(100)
    assert "public rating not provided" in unrated.status
    assert ranking.best.supplier_key == "rated"


def test_missing_total_is_not_treated_as_zero():
    ranking = rank_comparisons([
        row("Known", 150, 5, key="known"),
        row("Missing", None, 5, key="missing", eligible=False),
    ])

    missing = by_key(ranking)["missing"]
    assert missing.grand_total is None
    assert missing.price_score is None
    assert missing.combined_score is None
    assert missing.rank is None
    assert ranking.best.supplier_key == "known"


def test_tax_uncertainty_blocks_the_best_pick():
    assert is_critical_issue("Tax total is unknown.")
    ranking = rank_comparisons([
        row(
            "Uncertain",
            100,
            5,
            key="uncertain",
            critical=True,
            warnings=("Tax total is unknown.",),
        ),
        row("Clear", 120, 4, key="clear"),
    ])

    assert by_key(ranking)["uncertain"].combined_score is not None
    assert ranking.best.pending is False
    assert ranking.best.supplier_key == "clear"
    assert "critical review item" in ranking.best.explanation


def test_only_a_critical_supplier_leaves_the_best_pick_pending():
    ranking = rank_comparisons([
        row(
            "Uncertain",
            100,
            5,
            key="uncertain",
            critical=True,
            warnings=("Tax rate is unknown.",),
        ),
    ])

    assert ranking.best.pending is True
    assert ranking.best.headline == (
        "Best Pick pending — more information required"
    )
    assert "Tax rate is unknown." in ranking.best.explanation


def test_incomparable_quotations_are_not_price_ranked():
    ranking = rank_comparisons([
        row("A", 100, 5, key="a", currency="INR"),
        row("B", 80, 5, key="b", currency="USD"),
    ])

    assert all(item.price_score is None for item in ranking.rows)
    assert all(item.combined_score is None for item in ranking.rows)
    assert ranking.best.pending is True
    assert "currenc" in ranking.best.explanation.casefold()


def test_price_only_ranking_is_labelled_when_no_combined_score_exists():
    ranking = rank_comparisons([
        row("Dear", 200, None, key="dear"),
        row("Cheap", 100, None, key="cheap"),
    ])

    assert ranking.price_only is True
    assert "price-only ranking" in ranking.price_only_reason
    assert by_key(ranking)["cheap"].rank == 1
    assert by_key(ranking)["cheap"].status == "Price-only rank"
    assert by_key(ranking)["dear"].price_score == pytest.approx(50)
    assert ranking.best.pending is True


def test_ratings_stay_with_the_supplier_key_not_the_display_name():
    first = SupplierRef(
        file_hash="file-one",
        filename="one.pdf",
        page=1,
        quotation_number="Q-1",
        supplier_name="ABC Electricals",
        occurrence=0,
    )
    second = SupplierRef(
        file_hash="file-two",
        filename="two.pdf",
        page=1,
        quotation_number="Q-2",
        supplier_name="ABC Electricals",
        occurrence=0,
    )
    assert supplier_key(first) != supplier_key(second)

    ranking = rank_comparisons([
        row("ABC Electricals", 100, 5, reviews=10, key=supplier_key(first)),
        row("ABC Electricals", 100, 2, reviews=80, key=supplier_key(second)),
    ])
    scored = by_key(ranking)

    assert scored[supplier_key(first)].quality_score == pytest.approx(100)
    assert scored[supplier_key(second)].quality_score == pytest.approx(40)
    assert scored[supplier_key(first)].review_count == 10
    assert scored[supplier_key(second)].review_count == 80
    assert ranking.best.supplier_key == supplier_key(first)
