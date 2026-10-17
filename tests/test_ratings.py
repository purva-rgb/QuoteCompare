import pytest

from engine.ratings import (
    PUBLIC_PROVENANCE,
    QUALITY_PROVENANCE,
    RatingInputError,
    SupplierRef,
    supplier_key,
    validate_supplier_rating,
)
from engine.scoring import CRITERIA


def ref(**overrides):
    data = {
        "file_hash": "abc123",
        "filename": "abc.pdf",
        "page": 1,
        "quotation_number": "Q-1",
        "supplier_name": "ABC Electricals",
        "occurrence": 0,
    }
    data.update(overrides)
    return SupplierRef(**data)


def test_valid_rating_keeps_provenance_separate():
    quality = {criterion: 4 for criterion in CRITERIA}
    quality[CRITERIA[0]] = 5

    record = validate_supplier_rating(
        ref(),
        quality=quality,
        public_rating=4.6,
        review_count=28,
        listing_url="https://maps.google.com/?cid=1",
        identity_verified=True,
        date_checked="2026-10-17",
    )

    assert record.quality[CRITERIA[0]] == 5
    assert record.public_rating == 4.6
    assert record.review_count == 28
    assert record.identity_is_verified is True
    assert record.date_checked == "2026-10-17"
    assert record.quality_provenance == QUALITY_PROVENANCE
    assert record.public_provenance == PUBLIC_PROVENANCE
    assert record.quality_provenance != record.public_provenance
    assert record.to_dict()["public_rating"] == 4.6


def test_public_rating_and_review_count_can_be_saved_alone():
    record = validate_supplier_rating(
        ref(),
        public_rating=4.3,
        review_count=128,
    )

    assert record.public_rating == 4.3
    assert record.review_count == 128
    assert all(value is None for value in record.quality.values())
    assert record.listing_url is None
    assert record.identity_verified is None
    assert record.date_checked is None
    assert 0 not in record.quality.values()


def test_either_public_field_can_stay_blank():
    rating_only = validate_supplier_rating(ref(), public_rating=4.3)
    count_only = validate_supplier_rating(ref(), review_count=128)

    assert rating_only.public_rating == 4.3
    assert rating_only.review_count is None
    assert count_only.review_count == 128
    assert count_only.public_rating is None


def test_missing_values_stay_missing():
    record = validate_supplier_rating(ref(), quality={})

    assert record.public_rating is None
    assert record.review_count is None
    assert record.listing_url is None
    assert record.identity_verified is None
    assert record.date_checked is None
    assert all(record.quality[criterion] is None for criterion in CRITERIA)
    assert 0 not in record.quality.values()


def test_blank_text_is_missing_not_zero():
    record = validate_supplier_rating(
        ref(),
        quality={CRITERIA[0]: ""},
        public_rating="  ",
        review_count="",
        listing_url="  ",
        identity_verified="",
        date_checked="  ",
    )

    assert record.quality[CRITERIA[0]] is None
    assert record.public_rating is None
    assert record.review_count is None


def test_invalid_quality_ratings_are_rejected():
    for bad in (0, 6, -1, 2.5, True, "good"):
        with pytest.raises(RatingInputError):
            validate_supplier_rating(
                ref(),
                quality={CRITERIA[1]: bad},
            )


def test_invalid_public_rating_and_review_count_are_rejected():
    with pytest.raises(RatingInputError):
        validate_supplier_rating(ref(), public_rating=0)

    with pytest.raises(RatingInputError):
        validate_supplier_rating(ref(), public_rating=5.1)

    with pytest.raises(RatingInputError):
        validate_supplier_rating(ref(), review_count=-1)

    with pytest.raises(RatingInputError):
        validate_supplier_rating(ref(), review_count=True)

    with pytest.raises(RatingInputError):
        validate_supplier_rating(ref(), listing_url="javascript:alert(1)")

    with pytest.raises(RatingInputError):
        validate_supplier_rating(ref(), date_checked="15 October 2026")


def test_zero_reviews_is_a_real_count():
    record = validate_supplier_rating(ref(), review_count=0)

    assert record.review_count == 0


def test_duplicate_supplier_names_use_different_records():
    first = ref(
        file_hash="file-one",
        quotation_number="Q-104",
        supplier_name="ABC Electricals",
    )
    second = ref(
        file_hash="file-two",
        quotation_number="Q-208",
        supplier_name="ABC Electricals",
    )

    assert supplier_key(first) != supplier_key(second)

    saved = {
        supplier_key(first): validate_supplier_rating(
            first,
            quality={CRITERIA[0]: 5},
        ),
        supplier_key(second): validate_supplier_rating(
            second,
            quality={CRITERIA[0]: 2},
        ),
    }

    assert saved[supplier_key(first)].quality[CRITERIA[0]] == 5
    assert saved[supplier_key(second)].quality[CRITERIA[0]] == 2


def test_same_quotation_keeps_one_key_when_only_the_name_differs():
    original = ref(supplier_name="ABC Electricals")
    renamed = ref(supplier_name="ABC Electrical")

    assert supplier_key(original) == supplier_key(renamed)


def test_unverified_listing_is_not_treated_as_verified():
    unchecked = validate_supplier_rating(
        ref(),
        public_rating=4,
        identity_verified=None,
    )
    rejected = validate_supplier_rating(
        ref(),
        public_rating=4,
        identity_verified=False,
    )
    confirmed = validate_supplier_rating(
        ref(),
        public_rating=4,
        identity_verified="yes",
    )

    assert unchecked.identity_verified is None
    assert unchecked.identity_is_verified is False
    assert rejected.identity_verified is False
    assert rejected.identity_is_verified is False
    assert confirmed.identity_is_verified is True
    assert confirmed.public_provenance == "manual_google_maps"
