from decimal import Decimal

import pytest

from engine.scoring import (
    CRITERIA,
    ScoreInput,
    average_rating,
    score_suppliers,
    validate_rating,
    winner_ranges,
)


def ratings(value):
    return {criterion: value for criterion in CRITERIA}


@pytest.fixture
def sample_inputs():
    # Crescent is cheapest but rated lowest; BluePeak is 0.28% dearer.
    return [
        ScoreInput("BluePeak", Decimal("214760"), True, ratings(5)),
        ScoreInput("Crescent", Decimal("214170"), True, ratings(3)),
        ScoreInput("SilverFern", Decimal("228920"), True, ratings(4)),
    ]


def by_label(scored):
    return {s.label: s for s in scored}


def test_price_only_follows_price_order(sample_inputs):
    scored = score_suppliers(sample_inputs, 100)

    assert [s.label for s in scored] == ["Crescent", "BluePeak", "SilverFern"]
    assert by_label(scored)["Crescent"].price_score == pytest.approx(100)
    assert by_label(scored)["Crescent"].combined_score == pytest.approx(100)


def test_quality_only_follows_ratings(sample_inputs):
    scored = score_suppliers(sample_inputs, 0)

    assert [s.label for s in scored] == ["BluePeak", "SilverFern", "Crescent"]
    assert by_label(scored)["BluePeak"].quality_score == pytest.approx(100)
    assert by_label(scored)["Crescent"].quality_score == pytest.approx(50)


def test_blended_weights(sample_inputs):
    scored = by_label(score_suppliers(sample_inputs, 70))

    # Crescent: 0.7 * 100 + 0.3 * 50 = 85
    assert scored["Crescent"].combined_score == pytest.approx(85)
    assert scored["BluePeak"].rank == 1
    assert scored["Crescent"].rank == 3


def test_unrated_supplier_is_excluded_not_averaged(sample_inputs):
    sample_inputs[0] = ScoreInput(
        "BluePeak", Decimal("214760"), True, {c: None for c in CRITERIA}
    )

    scored = score_suppliers(sample_inputs, 70)

    assert scored[-1].label == "BluePeak"
    assert scored[-1].rank is None
    assert scored[-1].excluded_reason == "No quality ratings entered."


def test_ineligible_supplier_is_excluded(sample_inputs):
    sample_inputs[1] = ScoreInput(
        "Crescent", Decimal("214170"), False, ratings(5)
    )

    scored = score_suppliers(sample_inputs, 70)

    assert scored[-1].label == "Crescent"
    assert scored[-1].excluded_reason == "Not eligible for price ranking."


def test_price_score_uses_lowest_eligible_total_even_if_unrated(sample_inputs):
    sample_inputs[1] = ScoreInput(
        "Crescent", Decimal("214170"), True, {c: None for c in CRITERIA}
    )

    scored = by_label(score_suppliers(sample_inputs, 100))

    # BluePeak is compared with Crescent's lower total, so it cannot be 100.
    assert scored["BluePeak"].price_score < 100


def test_equal_scores_share_a_rank():
    inputs = [
        ScoreInput("A", Decimal("100"), True, ratings(4)),
        ScoreInput("B", Decimal("100"), True, ratings(4)),
    ]

    scored = score_suppliers(inputs, 50)

    assert [s.rank for s in scored] == [1, 1]


def test_partial_ratings_average_only_rated_criteria():
    partial = {CRITERIA[0]: 5, CRITERIA[1]: 3, CRITERIA[2]: None}

    assert average_rating(partial) == pytest.approx(4)
    assert average_rating({c: None for c in CRITERIA}) is None


def test_validate_rating():
    assert validate_rating(None) is None
    assert validate_rating(float("nan")) is None
    assert validate_rating(4) == 4
    assert validate_rating(4.0) == 4
    assert validate_rating("3") == 3

    for bad in (0, 6, 2.5, True, "abc"):
        with pytest.raises(ValueError):
            validate_rating(bad)


def test_price_weight_must_be_a_percentage(sample_inputs):
    with pytest.raises(ValueError):
        score_suppliers(sample_inputs, 101)

    with pytest.raises(ValueError):
        score_suppliers(sample_inputs, -1)


def test_winner_changes_as_quality_weight_rises(sample_inputs):
    ranges = winner_ranges(sample_inputs)

    assert ranges == [
        (0, 0, ("Crescent",)),
        (5, 100, ("BluePeak",)),
    ]


def test_winner_never_changes_when_one_dominates():
    inputs = [
        ScoreInput("Cheap", Decimal("100"), True, ratings(5)),
        ScoreInput("Dear", Decimal("150"), True, ratings(2)),
    ]

    assert winner_ranges(inputs) == [(0, 100, ("Cheap",))]


def test_winner_ranges_rejects_bad_step(sample_inputs):
    with pytest.raises(ValueError):
        winner_ranges(sample_inputs, step=7)


def test_no_scored_suppliers_gives_no_ranges():
    inputs = [ScoreInput("A", Decimal("100"), True, {})]

    assert winner_ranges(inputs) == []
    assert score_suppliers(inputs, 50)[0].excluded_reason