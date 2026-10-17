"""Supplier ratings entered by the user.

Quality scores are the user's own assessment. Google Maps figures are only
values the user transcribed from a listing. This module does not contact
Google, and it does not calculate a combined supplier score.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Any, Mapping

from engine.scoring import CRITERIA, validate_rating

QUALITY_PROVENANCE = "user_entered"
PUBLIC_PROVENANCE = "manual_google_maps"

_VERIFIED_TRUE = {True, "yes", "y", "true", "verified"}
_VERIFIED_FALSE = {False, "no", "n", "false", "not verified"}


class RatingInputError(ValueError):
    """One or more rating fields were present but not usable."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


@dataclass(frozen=True)
class SupplierRef:
    """The quotation record a rating belongs to.

    The supplier name is for display. The key also uses the source file,
    page, quotation number, and position in that file, so two businesses
    with the same name stay separate.
    """

    file_hash: str
    filename: str
    page: int | None
    quotation_number: str | None
    supplier_name: str | None
    occurrence: int

    def __post_init__(self) -> None:
        if not str(self.file_hash or "").strip():
            raise ValueError(
                "A supplier rating needs the source file hash."
            )
        if self.occurrence < 0:
            raise ValueError("Supplier occurrence cannot be negative.")


@dataclass(frozen=True)
class SupplierRating:
    ref: SupplierRef
    supplier_key: str
    quality: dict[str, int | None]
    public_rating: float | None
    review_count: int | None
    listing_url: str | None
    identity_verified: bool | None
    date_checked: str | None
    quality_provenance: str = QUALITY_PROVENANCE
    public_provenance: str = PUBLIC_PROVENANCE

    @property
    def identity_is_verified(self) -> bool:
        """True only when the user explicitly confirmed the listing."""
        return self.identity_verified is True

    def to_dict(self) -> dict[str, Any]:
        return {
            "supplier_key": self.supplier_key,
            "supplier_name": self.ref.supplier_name,
            "file_hash": self.ref.file_hash,
            "filename": self.ref.filename,
            "page": self.ref.page,
            "quotation_number": self.ref.quotation_number,
            "occurrence": self.ref.occurrence,
            "quality": dict(self.quality),
            "quality_provenance": self.quality_provenance,
            "public_rating": self.public_rating,
            "review_count": self.review_count,
            "listing_url": self.listing_url,
            "identity_verified": self.identity_verified,
            "date_checked": self.date_checked,
            "public_provenance": self.public_provenance,
        }


def supplier_key(ref: SupplierRef) -> str:
    """Identity of the quotation record. The display name is not included."""
    page = "" if ref.page is None else str(ref.page)
    number = " ".join(str(ref.quotation_number or "").split())
    material = "\n".join([
        str(ref.file_hash).strip(),
        page,
        number,
        str(ref.occurrence),
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _validate_public_rating(value: Any) -> float | None:
    if _blank(value):
        return None
    if isinstance(value, bool):
        raise ValueError("Public rating must be from 1 to 5, or left empty.")
    if isinstance(value, float) and value != value:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Public rating must be from 1 to 5, or left empty."
        ) from exc
    if number != number:
        return None
    if not 1 <= number <= 5:
        raise ValueError("Public rating must be from 1 to 5, or left empty.")
    return float(number)


def _validate_review_count(value: Any) -> int | None:
    if _blank(value):
        return None
    if isinstance(value, bool):
        raise ValueError("Review count must be zero or greater, or left empty.")
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, float) and value != int(value):
        raise ValueError("Review count must be a whole number.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Review count must be zero or greater, or left empty."
        ) from exc
    if number < 0:
        raise ValueError("Review count cannot be negative.")
    return number


def _validate_url(value: Any) -> str | None:
    if _blank(value):
        return None
    if not isinstance(value, str):
        raise ValueError("Listing URL must be a web address, or left empty.")
    url = value.strip()
    if not (url.startswith("https://") or url.startswith("http://")):
        raise ValueError(
            "Listing URL must start with http:// or https://, or be left empty."
        )
    return url


def _validate_verified(value: Any) -> bool | None:
    if _blank(value):
        return None
    if value in _VERIFIED_TRUE:
        return True
    if value in _VERIFIED_FALSE:
        return False
    raise ValueError(
        "Identity verified must be yes, no, or left blank."
    )


def _validate_date(value: Any) -> str | None:
    if _blank(value):
        return None
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise ValueError("Date checked must be YYYY-MM-DD, or left empty.")
    try:
        return date.fromisoformat(value.strip()).isoformat()
    except ValueError as exc:
        raise ValueError(
            "Date checked must be YYYY-MM-DD, or left empty."
        ) from exc


def _validate_quality(values: Mapping[str, Any] | None) -> dict[str, int | None]:
    incoming = dict(values or {})
    unknown = sorted(set(incoming) - set(CRITERIA))
    if unknown:
        raise RatingInputError([
            "Unknown quality field: " + ", ".join(unknown) + "."
        ])

    quality: dict[str, int | None] = {}
    errors: list[str] = []
    for criterion in CRITERIA:
        raw = incoming.get(criterion)
        if criterion not in incoming or _blank(raw):
            quality[criterion] = None
            continue
        try:
            quality[criterion] = validate_rating(raw)
        except ValueError as exc:
            errors.append(f"{criterion}: {exc}")
    if errors:
        raise RatingInputError(errors)
    return quality


def validate_supplier_rating(
    ref: SupplierRef,
    quality: Mapping[str, Any] | None = None,
    public_rating: Any = None,
    review_count: Any = None,
    listing_url: Any = None,
    identity_verified: Any = None,
    date_checked: Any = None,
) -> SupplierRating:
    """Validate one supplier's ratings. Missing fields stay missing.

    Provenance is assigned here. Callers cannot mark a public rating as
    fetched from an external service.
    """
    if not isinstance(ref, SupplierRef):
        raise TypeError("ref must be a SupplierRef.")

    errors: list[str] = []
    try:
        checked_quality = _validate_quality(quality)
    except RatingInputError as exc:
        checked_quality = {}
        errors.extend(exc.errors)

    checked_public: float | None = None
    checked_count: int | None = None
    checked_url: str | None = None
    checked_verified: bool | None = None
    checked_date: str | None = None

    for label, parser, raw in (
        ("public_rating", _validate_public_rating, public_rating),
        ("review_count", _validate_review_count, review_count),
        ("listing_url", _validate_url, listing_url),
        ("identity_verified", _validate_verified, identity_verified),
        ("date_checked", _validate_date, date_checked),
    ):
        try:
            parsed = parser(raw)
        except ValueError as exc:
            errors.append(str(exc))
            parsed = None
        if label == "public_rating":
            checked_public = parsed
        elif label == "review_count":
            checked_count = parsed
        elif label == "listing_url":
            checked_url = parsed
        elif label == "identity_verified":
            checked_verified = parsed
        else:
            checked_date = parsed

    if errors:
        raise RatingInputError(errors)

    return SupplierRating(
        ref=ref,
        supplier_key=supplier_key(ref),
        quality=checked_quality,
        public_rating=checked_public,
        review_count=checked_count,
        listing_url=checked_url,
        identity_verified=checked_verified,
        date_checked=checked_date,
        quality_provenance=QUALITY_PROVENANCE,
        public_provenance=PUBLIC_PROVENANCE,
    )
