
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
            "Could not safely process the extracted quotation."
        ) from exc


def process_quotation_text(pdf_text: str) -> Quotation:
    """
    Run the complete local pipeline from extracted PDF text.

    PDF reading/OCR happens before this function.
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
    except Exception as exc:
        logger.exception("Quotation extraction pipeline failed.")
        raise QuotationProcessingError(
            "Quotation extraction failed. "
            "Check the application logs for details."
        ) from exc

def process_quotation_pages(page_texts: list[str]) -> list[Quotation]:
    """Process each PDF page as an independent supplier quotation."""
    if not page_texts:
        raise QuotationProcessingError("The PDF contains no pages to process.")

    results: list[Quotation] = []

    for page_number, page_text in enumerate(page_texts, start=1):
        if not page_text or not page_text.strip():
            logger.warning("Skipping empty PDF page %s.", page_number)
            continue

        logger.info("Processing quotation page %s...", page_number)

        try:
            quotation = process_quotation_text(page_text)
            results.append(quotation)
        except Exception as exc:
            logger.exception(
                "Failed to process quotation page %s.", page_number
            )
            raise QuotationProcessingError(
                f"Could not process PDF page {page_number}: {exc}"
            ) from exc

    if not results:
        raise QuotationProcessingError(
            "No pages contained extractable quotation text."
        )

    return results
