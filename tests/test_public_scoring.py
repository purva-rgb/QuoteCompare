from decimal import Decimal

import pytest

from engine.ratings import SupplierRef, supplier_key
from engine.scoring import (
    BEST_PICK_UNAVAILABLE,
    DEFAULT_PRICE_WEIGHT_PERCENT,
    ComparisonInput,
    is_advisory_extraction_warning,
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


def test_changing_weights_changes_scores_and_can_change_the_best_pick():
    suppliers = [
        row("Low price", 100, 2, key="cheap"),
        row("High rating", 200, 5, key="rated"),
    ]

    price_heavy = rank_comparisons(suppliers, price_weight_percent=80)
    quality_heavy = rank_comparisons(suppliers, price_weight_percent=20)

    assert price_heavy.price_weight_percent == 80
    assert 80 + (100 - price_heavy.price_weight_percent) == 100
    assert 20 + (100 - quality_heavy.price_weight_percent) == 100

    # Price and quality scores do not depend on the weight.
    assert by_key(price_heavy)["cheap"].price_score == pytest.approx(100)
    assert by_key(price_heavy)["rated"].price_score == pytest.approx(50)
    assert by_key(quality_heavy)["cheap"].quality_score == pytest.approx(40)
    assert by_key(quality_heavy)["rated"].quality_score == pytest.approx(100)

    # 80/20: cheap 88, rated 60. 20/80: cheap 52, rated 90.
    assert by_key(price_heavy)["cheap"].combined_score == pytest.approx(88)
    assert by_key(price_heavy)["rated"].combined_score == pytest.approx(60)
    assert price_heavy.best.supplier_key == "cheap"
    assert price_heavy.rows[0].supplier_key == "cheap"

    assert by_key(quality_heavy)["cheap"].combined_score == pytest.approx(52)
    assert by_key(quality_heavy)["rated"].combined_score == pytest.approx(90)
    assert quality_heavy.best.supplier_key == "rated"
    assert quality_heavy.rows[0].supplier_key == "rated"
    assert "weighted 20%" in quality_heavy.best.explanation
    assert "quality 80%" in quality_heavy.best.explanation


def test_weight_change_does_not_score_missing_or_invalid_totals():
    suppliers = [
        row("Open", 100, None, reviews=None, key="open"),
        row("Bad", None, 5, key="bad", eligible=False),
        row("Good", 120, 4, key="good"),
    ]

    for weight in (0, 70, 100):
        ranking = rank_comparisons(suppliers, price_weight_percent=weight)
        assert ranking.price_weight_percent == weight
        assert ranking.price_weight_percent + (
            100 - ranking.price_weight_percent
        ) == 100
        assert by_key(ranking)["open"].quality_score is None
        assert by_key(ranking)["open"].combined_score is None
        assert by_key(ranking)["bad"].price_score is None
        assert by_key(ranking)["bad"].combined_score is None
        assert ranking.best.supplier_key == "good"


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
    assert ranking.best.headline == BEST_PICK_UNAVAILABLE
    assert ranking.best.supplier_key is None
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


GLYPH_WARNING = (
    "Extraction uncertainty: 4 unreadable character(s) were left unchanged. "
    "Verify amounts and currency against the PDF. This is not an arithmetic "
    "error."
)
UNCONFIRMED_CURRENCY = (
    "Extraction uncertainty: currency INR was not confirmed by an "
    "explicit code or unambiguous symbol in the text. Verify it manually."
)


def test_higher_combined_score_beats_the_lowest_price():
    ranking = rank_comparisons([
        row("Cheap", 100, 1, key="cheap"),
        row("Balanced", 130, 5, key="balanced"),
    ])

    cheap = by_key(ranking)["cheap"]
    balanced = by_key(ranking)["balanced"]
    assert cheap.price_score == pytest.approx(100)
    assert cheap.price_score > balanced.price_score
    assert balanced.combined_score > cheap.combined_score
    assert ranking.best.pending is False
    assert ranking.best.supplier_key == "balanced"
    assert ranking.rows[0].supplier_key == ranking.best.supplier_key
    assert ranking.best.combined_score == pytest.approx(
        balanced.combined_score
    )


def test_glyph_warning_alone_does_not_block_best_pick():
    assert is_advisory_extraction_warning(GLYPH_WARNING) is True
    assert is_critical_issue(GLYPH_WARNING) is False
    assert is_critical_issue(UNCONFIRMED_CURRENCY) is True

    ranking = rank_comparisons([
        row("Checked", 140, 5, key="checked", warnings=(GLYPH_WARNING,)),
        row("Cheap", 100, 1, key="cheap"),
    ])

    assert ranking.best.pending is False
    assert ranking.best.supplier_key == "checked"
    assert by_key(ranking)["cheap"].price_score > by_key(ranking)["checked"].price_score


def test_unresolved_currency_still_blocks_the_winner():
    ranking = rank_comparisons([
        row(
            "Unconfirmed",
            100,
            5,
            key="unconfirmed",
            critical=True,
            warnings=(UNCONFIRMED_CURRENCY,),
        ),
        row("Clear", 150, 4, key="clear"),
    ])

    assert ranking.best.supplier_key == "clear"
    assert ranking.best.supplier_key != "unconfirmed"


def test_missing_rating_cannot_be_a_best_pick():
    ranking = rank_comparisons([
        row("Open", 100, None, reviews=None, key="open"),
    ])

    assert ranking.best.pending is True
    assert ranking.best.headline == BEST_PICK_UNAVAILABLE
    assert ranking.best.supplier_key is None
    assert ranking.best.combined_score is None
    assert "public rating" in ranking.best.explanation.casefold()


def test_best_pick_matches_the_highest_eligible_table_score():
    ranking = rank_comparisons([
        row("Cheap", 100, 1, key="cheap"),
        row("Balanced", 140, 5, key="balanced"),
        row(
            "Broken",
            80,
            5,
            key="broken",
            critical=True,
            warnings=("Tax total is unknown.",),
        ),
    ])

    eligible = [
        item for item in ranking.rows
        if item.combined_score is not None and item.eligible and not item.critical
    ]
    leader = max(eligible, key=lambda item: item.combined_score or 0)
    assert leader.supplier_key == "balanced"
    assert ranking.best.pending is False
    assert ranking.best.supplier_key == leader.supplier_key
    assert ranking.best.combined_score == pytest.approx(leader.combined_score)
    assert ranking.rows[0].supplier_key == "broken"
    assert ranking.best.supplier_key != ranking.rows[0].supplier_key
