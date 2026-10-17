"""Main processing pipeline for QuoteCompare."""

from __future__ import annotations

import logging

from pydantic import ValidationError

from engine.extractor import extract_quotation
from engine.normalizer import normalize_quotation
from engine.schemas import Quotation
from engine.validator import validate_quotation

logger = logging.getLogger(__name__)


class QuotationProcessingError(RuntimeError):
    """Raised when quotation processing cannot safely continue."""


def process_extracted_quotation(
    quotation: Quotation,
) -> Quotation:
    """
    Normalize and validate an already-extracted quotation.

    Does not call Ollama. Useful for unit tests and debugging.
    """
    if not isinstance(quotation, Quotation):
        raise TypeError(
            "quotation must be an instance of Quotation."
        )

    try:
        # Work on a copy so callers retain their original object.
        result = quotation.model_copy(deep=True)

        # Normalize known representations without inventing facts.
        result = normalize_quotation(result)

        # Check fields and arithmetic after normalization.
        validation = validate_quotation(result)
        result.validation = validation

        logger.info(
            "Quotation processed: items=%d, validation=%s",
            len(result.items),
            result.validation.status,
        )

        return result

    except (ValueError, TypeError, ValidationError):
        logger.exception(
            "Quotation processing failed during normalization "
            "or validation."
        )
        raise
    except Exception as exc:
        logger.exception("Unexpected quotation processing failure.")
        raise QuotationProcessingError(
            "Could not safely process the extracted quotation: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def process_quotation_text(pdf_text: str) -> Quotation:
    """
    Run the complete local pipeline from extracted PDF text.

    PDF reading/OCR happens before this function. Page splitting is done
    by the caller (app.py), which calls this once per quotation.
    """
    if not isinstance(pdf_text, str):
        raise TypeError("pdf_text must be a string.")

    if not pdf_text.strip():
        raise ValueError(
            "PDF extraction returned no text. "
            "Check the PDF or run OCR for scanned documents."
        )

    try:
        # Step 1: Extract facts using local Ollama.
        extracted = extract_quotation(pdf_text)

        # Step 2: Normalize and validate the extracted facts.
        return process_extracted_quotation(extracted)

    except (ValueError, TypeError, ValidationError):
        # These errors contain useful information for debugging.
        raise
    except QuotationProcessingError:
        raise
    except Exception as exc:
        # Includes the extractor's RuntimeError for Ollama problems
        # (not running, model missing, timeout). Keep its message so the
        # user sees the real cause instead of a generic failure.
        logger.exception("Quotation extraction pipeline failed.")
        raise QuotationProcessingError(
            f"Quotation extraction failed: {exc}"
        ) from exc
