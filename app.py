from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import logging
import time
from datetime import datetime
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import streamlit as st
from pypdf import PdfReader

from engine.extractor import (
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
    check_ollama as _check_ollama,
    extraction_fingerprint,
    set_progress_hook,
)
from engine.matcher import find_candidates
from engine.pipeline import (
    QuotationProcessingError,
    process_quotation_text,
)
from engine.ratings import (
    RatingInputError,
    SupplierRef,
    supplier_key,
    validate_supplier_rating,
)
from engine.text_quality import diagnose_page, font_currency_hints
from engine.recommender import (
    STATUS_EFFECTIVELY_TIED,
    STATUS_NONE,
    STATUS_SINGLE_ELIGIBLE,
    QuotationEntry,
    Recommendation,
    recommend,
)
from engine.schemas import Quotation
from engine.scoring import (
    CRITERIA,
    DEFAULT_PRICE_WEIGHT_PERCENT,
    ComparisonInput,
    ScoreInput,
    is_advisory_extraction_warning,
    is_critical_issue,
    rank_comparisons,
    score_suppliers,
    validate_rating,
    winner_ranges,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

APP_NAME = "QuoteCompare"
MAX_FILES = 10
MAX_FILE_SIZE_MB = 20
MAX_PAGES = 200
MAX_TEXT_CHARS = 60_000          # whole-PDF mode limit
MAX_PAGE_TEXT_CHARS = 20_000     # per-page limit
MIN_PAGE_CHARS = 30              # below this a page is treated as blank

MAX_CONSECUTIVE_FAILURES = 2     # stop a file early if the model keeps failing

# Model, URL and timeout come from engine/extractor.py so the app
# and the engine can never disagree.
MIN_RECOMMENDED_TIMEOUT = 300
MAX_CACHED_PAGES = 200           # oldest cached pages are evicted beyond this

MODE_PER_PAGE = "Each page is a separate quotation (recommended)"
MODE_WHOLE = "The whole PDF is one quotation"

RESULTS_KEY = "quotecompare_results"
SIGNATURE_KEY = "quotecompare_signature"
PAGE_CACHE_KEY = "quotecompare_page_cache"
RATINGS_KEY = "quotecompare_ratings"  # kept when cached results are cleared
SUPPLIER_RATINGS_KEY = "quotecompare_supplier_ratings"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# Small helpers
# ---------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=1)
def _background_data_uri() -> str | None:
    """Local photograph, if the project has one. Never fetched at runtime."""
    root = Path(__file__).resolve().parent / "assets"
    for name in (
        "quotecompare_background.jpg",
        "quotecompare_background.jpeg",
        "quotecompare_background.png",
        "image.png",
    ):
        path = root / name
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if not data:
            continue
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    return None


def _page_background() -> str:
    """Photograph for every screen. A gradient is used only if the file is missing."""
    image = _background_data_uri()
    if image:
        return (
            "linear-gradient(180deg, rgba(8, 14, 26, 0.55) 0%, "
            "rgba(8, 14, 26, 0.72) 100%), "
            f"url('{image}') center center / cover no-repeat fixed"
        )
    return (
        "radial-gradient(900px 420px at 12% 0%, "
        "rgba(198, 161, 91, 0.22), transparent 60%), "
        "linear-gradient(165deg, #0b1220 0%, #17263d 52%, #0d1728 100%)"
    )


def _apply_theme(hero: bool) -> None:
    """Same photograph on the opening screen and after every results rerun."""
    background = _page_background()

    hero_text = """
    .qc-brand {
        letter-spacing: 0.28em;
        color: #c6a15b;
        font-size: 0.82rem;
        font-weight: 650;
        margin-bottom: 0.35rem;
    }
    .qc-headline {
        color: #f8f5ef;
        font-size: clamp(2rem, 4vw, 3.4rem);
        line-height: 1.12;
        font-weight: 650;
        margin: 0 0 0.6rem 0;
        max-width: 14em;
    }
    .qc-lead {
        color: rgba(247, 244, 238, 0.88);
        font-size: 1.05rem;
        max-width: 38rem;
        margin-bottom: 1.25rem;
    }
    .st-key-hero_panel {
        background: rgba(12, 20, 34, 0.58);
        border: 1px solid rgba(198, 161, 91, 0.42);
        border-radius: 18px;
        padding: 1.25rem 1.35rem 0.85rem;
        box-shadow: 0 20px 48px rgba(0, 0, 0, 0.28);
        backdrop-filter: blur(16px);
    }
    .st-key-hero_panel p,
    .st-key-hero_panel label,
    .st-key-hero_panel span {
        color: #f4efe6;
    }
    """ if hero else ""

    st.markdown(
        f"""
        <style>
        .stApp,
        [data-testid="stAppViewContainer"] {{
            background: {background} !important;
        }}
        [data-testid="stSidebar"] {{
            background: rgba(9, 14, 24, 0.94) !important;
        }}
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            color: #e7edf6;
        }}
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        [data-testid="stBottom"],
        .block-container {{
            background: transparent !important;
            background-color: transparent !important;
        }}
        [data-testid="stHeader"] {{
            background: rgba(8, 14, 26, 0.28) !important;
            backdrop-filter: blur(8px);
        }}
        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] p,
        [data-testid="stMain"] label,
        [data-testid="stMain"] li,
        [data-testid="stCaptionContainer"] p {{
            color: #f4efe6;
        }}
        [data-testid="stForm"],
        [data-testid="stExpander"],
        [data-testid="stDataFrame"],
        .st-key-best_pick {{
            background: rgba(255, 252, 247, 0.94) !important;
            color: #1c2430;
            border: 1px solid rgba(198, 161, 91, 0.38);
            border-radius: 14px;
        }}
        .st-key-best_pick {{
            border-top: 4px solid #c6a15b;
            padding: 0.35rem 0.85rem 0.15rem;
        }}
        [data-testid="stForm"] p,
        [data-testid="stForm"] label,
        [data-testid="stForm"] span,
        [data-testid="stForm"] h1,
        [data-testid="stForm"] h2,
        [data-testid="stForm"] h3,
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] label,
        [data-testid="stExpander"] span,
        [data-testid="stExpander"] summary,
        .st-key-best_pick p,
        .st-key-best_pick label,
        .st-key-best_pick span,
        .st-key-best_pick li,
        .st-key-best_pick h1,
        .st-key-best_pick h2,
        .st-key-best_pick h3,
        [data-testid="stForm"] [data-testid="stCaptionContainer"] p,
        [data-testid="stExpander"] [data-testid="stCaptionContainer"] p,
        .st-key-best_pick [data-testid="stCaptionContainer"] p,
        [data-testid="stAlert"],
        [data-testid="stAlert"] p {{
            color: #1c2430;
        }}
        [data-testid="stBaseButton-primary"] span,
        [data-testid="stBaseButton-primary"] p {{
            color: #1c1508;
        }}
        [data-testid="stMain"] input,
        [data-testid="stMain"] textarea {{
            color: #1c2430;
            background-color: #ffffff;
        }}
        .block-container {{
            padding-top: {"2.2rem" if hero else "1.4rem"};
            max-width: {"920px" if hero else "1180px"};
        }}
        [data-testid="stBaseButton-primary"] {{
            background: #c6a15b;
            color: #1c1508;
            border: none;
        }}
        [data-testid="stBaseButton-primary"]:hover {{
            background: #d4b36e;
            color: #1c1508;
        }}
        {hero_text}
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_table(rows: list[dict[str, Any]]) -> None:
    """Render a table across old and new Streamlit versions."""
    try:
        st.dataframe(rows, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(rows, use_container_width=True, hide_index=True)


def model_to_dict(value: Any) -> Any:
    """Convert Pydantic models and nested values into JSON-safe data."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if isinstance(value, dict):
        return {str(k): model_to_dict(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [model_to_dict(v) for v in value]

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def validation_dict(quotation_data: dict[str, Any]) -> dict[str, Any]:
    validation = quotation_data.get("validation")
    return validation if isinstance(validation, dict) else {}


def get_issues(validation: dict[str, Any]) -> list[str]:
    """Handle the common validation issue representations."""
    issues = validation.get("issues", [])

    if issues is None:
        return []

    if isinstance(issues, str):
        return [issues]

    if isinstance(issues, list):
        out: list[str] = []
        for issue in issues:
            if isinstance(issue, dict):
                out.append(
                    str(issue.get("message") or issue.get("detail") or issue)
                )
            else:
                out.append(str(issue))
        return out

    return [str(issues)]


def safe_stem(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0]
    cleaned = "".join(c if c.isalnum() or c in "-_ " else "_" for c in stem)
    return cleaned.strip().replace(" ", "_") or "quotation"


# ---------------------------------------------------------
# Ollama health check (advisory only, never blocks processing)
# ---------------------------------------------------------

@st.cache_data(ttl=15, show_spinner=False)
def check_ollama() -> tuple[bool, str]:
    return _check_ollama()


# ---------------------------------------------------------
# PDF reading
# ---------------------------------------------------------

def read_pdf(file_bytes: bytes) -> dict[str, Any]:
    """Read a PDF and return per-page text plus diagnostics."""

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    if not file_bytes.lstrip()[:5] == b"%PDF-":
        raise ValueError("This file does not have a valid PDF header.")

    reader = PdfReader(io.BytesIO(file_bytes), strict=False)

    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise ValueError(
                "This PDF is password-protected. "
                "Please provide an unlocked copy."
            ) from exc

        if not unlocked:
            raise ValueError(
                "This PDF is password-protected. "
                "Please provide an unlocked copy."
            )

    page_count = len(reader.pages)

    if page_count == 0:
        raise ValueError("The PDF contains no pages.")

    if page_count > MAX_PAGES:
        raise ValueError(
            f"This PDF has {page_count} pages. "
            f"The current limit is {MAX_PAGES} pages."
        )

    page_texts: list[str] = []
    diagnostics: list[dict[str, Any]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").replace("\x00", "").strip()
        except Exception as exc:
            logger.warning(
                "Text extraction failed on page %s: %s",
                page_number,
                type(exc).__name__,
            )
            text = ""

        page_texts.append(text)
        diagnostic = diagnose_page(
            page_number,
            text,
            font_currency_hints(page),
        )
        diagnostic["has_text"] = len(text) >= MIN_PAGE_CHARS
        diagnostics.append(diagnostic)

    combined = "\n\n".join(
        f"--- Page {i} ---\n{t}"
        for i, t in enumerate(page_texts, start=1)
        if t
    ).strip()

    hints: list[str] = []
    for diagnostic in diagnostics:
        for hint in diagnostic.get("font_hints") or []:
            if hint not in hints:
                hints.append(hint)

    return {
        "text": combined,
        "page_texts": page_texts,
        "page_count": page_count,
        "pages_with_text": sum(d["has_text"] for d in diagnostics),
        "page_diagnostics": diagnostics,
        "document_diagnostics": diagnose_page(None, combined, hints),
    }


@st.cache_data(show_spinner=False, max_entries=64)
def read_pdf_cached(file_hash: str, _file_bytes: bytes) -> dict[str, Any]:
    """Parse each distinct PDF once instead of on every Streamlit rerun."""
    return read_pdf(_file_bytes)


# ---------------------------------------------------------
# Processing
# ---------------------------------------------------------

def run_pipeline_cached(text: str) -> tuple[dict[str, Any], bool]:
    """Run the pipeline on one text block. Returns (data, was_cached)."""
    cache: dict[str, dict[str, Any]] = st.session_state.setdefault(
        PAGE_CACHE_KEY, {}
    )
    # The fingerprint covers model, prompt, schema and options, so cached
    # results are discarded automatically when the extractor changes.
    key = sha256_bytes(
        f"{extraction_fingerprint()}\0{text}".encode("utf-8")
    )

    if key in cache:
        return cache[key], True

    data = model_to_dict(process_quotation_text(text))

    if not isinstance(data, dict):
        raise QuotationProcessingError(
            "The pipeline returned an unexpected result type."
        )

    cache[key] = data
    while len(cache) > MAX_CACHED_PAGES:
        cache.pop(next(iter(cache)))  # dicts keep insertion order

    return data, False


def process_one_file(
    filename: str,
    file_bytes: bytes,
    mode: str,
    report: Callable[[str, float], None],
) -> dict[str, Any]:
    """Extract and process one PDF. Never raises; always returns a dict."""

    started = time.monotonic()
    file_hash = sha256_bytes(file_bytes)

    base: dict[str, Any] = {
        "filename": filename,
        "file_hash": file_hash,
        "success": False,
        "quotations": [],
        "page_errors": [],
        "skipped_pages": [],
    }

    def finish(**extra: Any) -> dict[str, Any]:
        base.update(extra)
        base["elapsed_seconds"] = round(time.monotonic() - started, 1)
        return base

    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        return finish(
            error=f"File exceeds the {MAX_FILE_SIZE_MB} MB size limit."
        )

    try:
        pdf_result = read_pdf_cached(file_hash, file_bytes)
    except Exception as exc:
        logger.warning(
            "PDF reading failed for %s: %s", filename, type(exc).__name__
        )
        return finish(error=str(exc))

    base["pdf_result"] = {
        k: v for k, v in pdf_result.items() if k not in ("text", "page_texts")
    }

    if pdf_result["pages_with_text"] == 0:
        return finish(
            error=(
                "No usable text was extracted. This PDF may be image-only. "
                "OCR was not run."
            )
        )

    diagnostics_by_page = {
        d["page"]: d for d in pdf_result["page_diagnostics"]
    }
    # Original page text is sent through unchanged. Unreadable glyphs are
    # reported, not replaced with a currency symbol.
    jobs: list[tuple[int | None, str]] = []

    if mode == MODE_WHOLE:
        text = pdf_result["text"]
        if len(text) > MAX_TEXT_CHARS:
            return finish(
                error=(
                    f"Extracted text has {len(text):,} characters, above the "
                    f"{MAX_TEXT_CHARS:,}-character limit for whole-PDF mode. "
                    "Switch to per-page mode or split the PDF."
                )
            )
        jobs.append((None, text))
    else:
        for page_number, text in enumerate(pdf_result["page_texts"], start=1):
            if len(text) < MIN_PAGE_CHARS:
                base["skipped_pages"].append(
                    {"page": page_number, "reason": "Blank or no readable text"}
                )
            elif len(text) > MAX_PAGE_TEXT_CHARS:
                base["page_errors"].append({
                    "page": page_number,
                    "error": (
                        f"Page text has {len(text):,} characters, above the "
                        f"{MAX_PAGE_TEXT_CHARS:,}-character page limit."
                    ),
                })
            else:
                jobs.append((page_number, text))

    if not jobs and not base["page_errors"]:
        return finish(error="No pages contained enough text to process.")

    consecutive_failures = 0
    total = max(len(jobs), 1)

    for position, (page_number, text) in enumerate(jobs):
        label = (
            f"page {page_number}" if page_number is not None else "whole PDF"
        )

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            base["page_errors"].append({
                "page": page_number,
                "error": (
                    "Skipped because earlier pages failed repeatedly. "
                    "Check that Ollama is running and responsive, then retry."
                ),
            })
            continue

        report(f"{filename}: processing {label}", position / total)

        try:
            data, _was_cached = run_pipeline_cached(text)
            if page_number is None:
                text_diagnostics = pdf_result.get("document_diagnostics")
            else:
                text_diagnostics = diagnostics_by_page.get(page_number)
            base["quotations"].append({
                "page": page_number,
                "data": data,
                "source_text": text,
                "text_diagnostics": text_diagnostics,
            })
            consecutive_failures = 0

        except (QuotationProcessingError, ValueError, RuntimeError) as exc:
            # Expected failures (validation, Ollama problems): show the
            # real message instead of a generic "unexpected error".
            logger.warning("%s %s failed: %s", filename, label, exc)
            base["page_errors"].append(
                {"page": page_number, "error": str(exc)}
            )
            consecutive_failures += 1

        except Exception:
            logger.exception("Unexpected failure for %s %s", filename, label)
            base["page_errors"].append({
                "page": page_number,
                "error": (
                    "An unexpected error occurred. Check the local "
                    "terminal for diagnostic details."
                ),
            })
            consecutive_failures += 1

    report(f"{filename}: done", 1.0)

    if base["quotations"]:
        return finish(success=True)

    return finish(
        error="No quotations were extracted. See the page errors below."
    )


# ---------------------------------------------------------
# Export helpers
# ---------------------------------------------------------

def build_export(
    results: list[dict[str, Any]],
    supplier_ratings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": OLLAMA_MODEL,
        "files": [
            {
                "filename": r["filename"],
                "file_sha256": r["file_hash"],
                "success": r["success"],
                "error": r.get("error"),
                "page_errors": r.get("page_errors", []),
                "quotations": r.get("quotations", []),
            }
            for r in results
        ],
    }
    if supplier_ratings:
        payload["supplier_ratings"] = supplier_ratings
    return payload


def build_line_items_csv(results: list[dict[str, Any]]) -> str:
    columns = [
        "file", "page", "supplier", "quotation_number", "quotation_date",
        "currency", "currency_status", "validation_status",
        "item_id", "description", "normalized_name", "quantity", "unit",
        "unit_price", "discount", "tax_rate", "line_total",
        "match_status", "warnings",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()

    for result in results:
        for entry in result.get("quotations", []):
            data = entry["data"]
            vendor = data.get("vendor") or {}
            if not isinstance(vendor, dict):
                vendor = {}

            quality = data.get("text_quality") or {}
            if not isinstance(quality, dict):
                quality = {}
            validation_status = validation_dict(data).get("status")

            for item in data.get("items") or []:
                if not isinstance(item, dict):
                    continue
                writer.writerow({
                    "file": result["filename"],
                    "page": entry["page"] or "",
                    "supplier": vendor.get("name"),
                    "quotation_number": vendor.get("quotation_number"),
                    "quotation_date": vendor.get("quotation_date"),
                    "currency": _csv_cell(vendor.get("currency")),
                    "currency_status": quality.get("currency_status")
                    or "unknown",
                    "validation_status": validation_status or "unknown",
                    "item_id": item.get("item_id"),
                    "description": item.get("original_description"),
                    "normalized_name": item.get("normalized_name"),
                    "quantity": _csv_cell(item.get("quantity")),
                    "unit": _csv_cell(item.get("unit")),
                    "unit_price": _csv_cell(item.get("unit_price")),
                    "discount": _csv_cell(item.get("discount")),
                    "tax_rate": _csv_cell(item.get("tax_rate")),
                    "line_total": _csv_cell(item.get("line_total")),
                    "match_status": item.get("match_status"),
                    "warnings": "; ".join(
                        str(w) for w in (item.get("warnings") or [])
                    ),
                })

    return buffer.getvalue()


# ---------------------------------------------------------
# Supplier comparison (candidates only, never auto-confirmed)
# ---------------------------------------------------------

STATUS_TEXT = {
    "candidate_strong": "Strong candidate — not confirmed",
    "candidate_possible": "Possible match — manual review",
    "needs_review": "Needs review",
}


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _shown(value: Any) -> str:
    """Show a missing extracted value as unknown, never as zero."""
    if value is None:
        return "Unknown"
    if isinstance(value, str) and not value.strip():
        return "Unknown"
    return str(value)


def _csv_cell(value: Any) -> Any:
    return "unknown" if value is None else value


def _qty_unit(item: dict[str, Any] | None) -> str:
    if not item:
        return ""

    parts = (_text(item.get("quantity")), _text(item.get("unit")))
    return " ".join(part for part in parts if part)


def _as_number(value: Any) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _same_basis(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> str:
    """Say whether two items are priced on the same quantity and unit."""
    if not left or not right:
        return ""

    qty_left, qty_right = left.get("quantity"), right.get("quantity")
    unit_left = _text(left.get("unit")).strip().casefold()
    unit_right = _text(right.get("unit")).strip().casefold()

    if qty_left is None or qty_right is None or not unit_left or not unit_right:
        return "Check quantity and unit (missing)"

    number_left, number_right = _as_number(qty_left), _as_number(qty_right)
    if number_left is not None and number_right is not None:
        same_qty = abs(number_left - number_right) < 1e-9
    else:
        same_qty = _text(qty_left) == _text(qty_right)

    if same_qty and unit_left == unit_right:
        return "Same quantity and unit"

    return "Different quantity or unit"


def _items_by_id(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("item_id")): item
        for item in (data.get("items") or [])
        if isinstance(item, dict)
    }


def build_comparison_pool(
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect every extracted quotation as a comparable entry."""
    pool: list[dict[str, Any]] = []
    skipped: list[str] = []
    used_labels: set[str] = set()

    for result in results:
        for entry in result.get("quotations", []):
            data = entry["data"]
            vendor = data.get("vendor")
            name = vendor.get("name") if isinstance(vendor, dict) else None
            where = result["filename"] + (
                f", page {entry['page']}" if entry["page"] else ""
            )
            label = f"{name or 'Unnamed supplier'} ({where})"

            if label in used_labels:
                label = f"{label} #{len(pool) + 1}"
            used_labels.add(label)

            try:
                quotation = Quotation.model_validate(data)
            except Exception:
                logger.warning("Could not rebuild %s for comparison.", label)
                skipped.append(label)
                continue

            pool.append({"label": label, "data": data, "quotation": quotation})

    return pool, skipped


def build_comparison_rows(
    pool: list[dict[str, Any]],
    base_index: int,
    threshold: float,
) -> list[dict[str, Any]]:
    """One row per (base item, other supplier) with the closest candidate."""
    base = pool[base_index]
    base_items = _items_by_id(base["data"])
    rows: list[dict[str, Any]] = []

    for item in base["quotation"].items:
        base_dict = base_items.get(str(item.item_id))

        for other_index, other in enumerate(pool):
            if other_index == base_index:
                continue

            row: dict[str, Any] = {
                "Base supplier": base["label"],
                "Base item": item.original_description,
                "Compared supplier": other["label"],
            }

            try:
                candidates = find_candidates(
                    item, other["quotation"], threshold
                )
            except Exception:
                logger.exception(
                    "Matching failed for %s against %s",
                    item.item_id,
                    other["label"],
                )
                candidates = None

            if candidates is None:
                row["Closest item"] = "Matching failed (see terminal)"
                rows.append(row)
                continue

            if not candidates:
                row["Closest item"] = "No close match"
                rows.append(row)
                continue

            best = candidates[0]
            other_dict = _items_by_id(other["data"]).get(
                str(best["candidate_item_id"])
            )

            comparable = bool(best.get("prices_comparable"))
            reasons = "; ".join(best.get("review_reasons") or [])
            row.update({
                "Closest item": _text(
                    (other_dict or {}).get("original_description")
                ),
                "Name similarity": best["similarity"],
                "Status": STATUS_TEXT.get(best["status"], best["status"]),
                "Why it needs review": reasons,
                "Specification conflicts": ", ".join(
                    best["conflicting_specifications"]
                ),
                "Basis": _same_basis(base_dict, other_dict),
                "Base qty / unit": _qty_unit(base_dict) or "Unknown",
                "Their qty / unit": _qty_unit(other_dict) or "Unknown",
                "Base unit price": (
                    _shown((base_dict or {}).get("unit_price"))
                    if comparable else "Not comparable"
                ),
                "Their unit price": (
                    _shown((other_dict or {}).get("unit_price"))
                    if comparable else "Not comparable"
                ),
                "Base line total": (
                    _shown((base_dict or {}).get("line_total"))
                    if comparable else "Not comparable"
                ),
                "Their line total": (
                    _shown((other_dict or {}).get("line_total"))
                    if comparable else "Not comparable"
                ),
            })
            rows.append(row)

    return rows


def recommendation_rows(rec: Recommendation) -> list[dict[str, Any]]:
    rows = []

    for a in rec.assessments:
        if a.eligible:
            status = "In the tie band" if a.within_tie_band else "Ranked"
            notes = list(a.cautions)
        else:
            status = "Not ranked"
            notes = list(a.ineligible_reasons)

        rows.append({
            "Rank": a.rank if a.rank is not None else "-",
            "Quotation": a.label,
            "Grand total": (
                f"{a.currency + ' ' if a.currency else ''}"
                f"{a.grand_total:,.2f}"
                if a.grand_total is not None else "Not extracted"
            ),
            "Above lowest": (
                f"{a.difference_from_lowest:,.2f} "
                f"({a.percent_above_lowest}%)"
                if a.difference_from_lowest is not None else ""
            ),
            "Status": status,
            "Review notes": " ".join(notes),
        })

    return rows


def render_recommendation(
    pool: list[dict[str, Any]],
) -> Recommendation | None:
    st.markdown("**Recommendation**")

    tie_margin = st.slider(
        "Treat totals within this percentage of the lowest as a tie",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        key="recommend_tie_margin",
        help=(
            "If more than one quotation is inside this band, no single "
            "winner is named and scope and terms should decide."
        ),
    )

    try:
        rec = recommend(
            [QuotationEntry(p["label"], p["quotation"]) for p in pool],
            tie_margin_percent=str(tie_margin),
        )
    except Exception:
        logger.exception("Recommendation failed.")
        st.error("The recommendation could not be calculated. See the terminal.")
        return None

    uncertain = any(
        a.cautions or a.extra_items or a.missing_items
        for a in rec.assessments
    )
    if rec.status == STATUS_NONE:
        st.warning(rec.headline)
    elif rec.status in (STATUS_SINGLE_ELIGIBLE, STATUS_EFFECTIVELY_TIED):
        st.info(rec.headline)
    elif uncertain:
        st.warning(rec.headline)
    else:
        st.success(rec.headline)

    show_table(recommendation_rows(rec))

    scope_rows = [
        {
            "Quotation": a.label,
            "Items only in this quotation": ", ".join(a.extra_items),
            "Items other quotations have that this lacks": ", ".join(
                a.missing_items
            ),
        }
        for a in rec.assessments
        if a.extra_items or a.missing_items
    ]

    if scope_rows:
        with st.expander("Scope differences (approximate)"):
            show_table(scope_rows)

    for note in rec.notes:
        st.caption(note)

    return rec


def _clean_records(edited: Any) -> list[dict[str, Any]]:
    """Editor results arrive as a DataFrame or a list of dicts."""
    if hasattr(edited, "to_dict"):
        return list(edited.to_dict("records"))

    return [dict(row) for row in edited]


def render_scorecard(rec: Recommendation) -> None:
    st.markdown("**Service quality and weighted score**")
    st.caption(
        "Ratings are your own judgement: quotation PDFs do not contain "
        "service quality, and the app does not look suppliers up online so "
        "that everything stays local. Rate from 1 (poor) to 5 (excellent), "
        "leave a cell empty if you have no basis for a rating, and note "
        "your evidence so the score can be defended."
    )

    labels = [a.label for a in rec.assessments]
    store: dict[str, dict[str, Any]] = st.session_state.setdefault(
        RATINGS_KEY, {}
    )

    rows = []
    for label in labels:
        saved = store.get(label, {})
        row: dict[str, Any] = {"Quotation": label}
        for criterion in CRITERIA:
            row[criterion] = saved.get(criterion)
        row["Evidence / notes"] = saved.get("notes", "")
        rows.append(row)

    editor_key = "ratings_editor_" + sha256_bytes(
        "|".join(labels).encode("utf-8")
    )[:12]

    column_config = {
        criterion: st.column_config.NumberColumn(
            criterion, min_value=1, max_value=5, step=1, format="%d"
        )
        for criterion in CRITERIA
    }
    column_config["Evidence / notes"] = st.column_config.TextColumn(
        "Evidence / notes"
    )

    try:
        edited = st.data_editor(
            rows,
            key=editor_key,
            hide_index=True,
            disabled=["Quotation"],
            column_config=column_config,
            width="stretch",
        )
    except TypeError:
        edited = st.data_editor(
            rows,
            key=editor_key,
            hide_index=True,
            disabled=["Quotation"],
            column_config=column_config,
            use_container_width=True,
        )

    inputs: list[ScoreInput] = []
    totals_by_label = {a.label: a for a in rec.assessments}

    for record in _clean_records(edited):
        label = str(record.get("Quotation"))
        assessment = totals_by_label.get(label)
        if assessment is None:
            continue

        cleaned: dict[str, int | None] = {}
        for criterion in CRITERIA:
            try:
                cleaned[criterion] = validate_rating(record.get(criterion))
            except ValueError as exc:
                st.warning(f"{label}, {criterion}: {exc} It was ignored.")
                cleaned[criterion] = None

        notes = record.get("Evidence / notes")
        store[label] = {
            **cleaned,
            "notes": "" if notes is None or notes != notes else str(notes),
        }

        inputs.append(ScoreInput(
            label=label,
            grand_total=assessment.grand_total,
            eligible=assessment.eligible,
            ratings=cleaned,
        ))

    price_weight = st.slider(
        "How much should price count? The rest goes to service quality.",
        min_value=0,
        max_value=100,
        value=70,
        step=5,
        format="%d%%",
        key="score_price_weight",
    )

    scored = score_suppliers(inputs, price_weight)
    ranked = [r for r in scored if r.rank is not None]

    if not ranked:
        st.info(
            "Enter ratings for at least one eligible quotation to see a "
            "weighted score."
        )
        return

    leaders = [r.label for r in ranked if r.rank == 1]
    price_leaders = [a.label for a in rec.assessments if a.rank == 1]

    if len(leaders) > 1:
        st.info(
            f"With price at {price_weight}% and quality at "
            f"{100 - price_weight}%, {', '.join(leaders)} are tied."
        )
    else:
        message = (
            f"With price at {price_weight}% and quality at "
            f"{100 - price_weight}%, {leaders[0]} scores highest."
        )
        if leaders[0] not in price_leaders:
            message += (
                " Quality ratings change the result: on price alone the "
                f"lowest total is {', '.join(price_leaders)}."
            )
        st.success(message)

    def one_decimal(value: float | None) -> str:
        return "" if value is None else f"{value:.1f}"

    show_table([
        {
            "Rank": r.rank if r.rank is not None else "-",
            "Quotation": r.label,
            "Grand total": (
                f"{r.grand_total:,.2f}" if r.grand_total is not None else ""
            ),
            "Price score": one_decimal(r.price_score),
            "Average rating (of 5)": one_decimal(r.quality_average),
            "Quality score": one_decimal(r.quality_score),
            "Combined score": one_decimal(r.combined_score),
            "Note": r.excluded_reason or "",
        }
        for r in scored
    ])

    ranges = winner_ranges(inputs)
    if len(ranges) > 1:
        parts = [
            f"{start}% to {end}%: {', '.join(winners)}"
            for start, end, winners in ranges
        ]
        st.caption(
            "Who leads as the weight on quality rises (checked in 5% steps): "
            + "; ".join(parts) + "."
        )
    elif ranges:
        st.caption(
            f"{', '.join(ranges[0][2])} leads at every weighting, so the "
            "result does not depend on how you balance price and quality."
        )

    st.caption(
        "Price score: the lowest eligible total scores 100 and others "
        "score lowest divided by their total. Quality score: the average "
        "rating mapped from 1-5 onto 0-100. Unrated or ineligible "
        "quotations are not scored."
    )


def render_comparison(results: list[dict[str, Any]]) -> None:
    st.subheader("Compare suppliers")

    pool, skipped = build_comparison_pool(results)

    for label in skipped:
        st.warning(f"{label} could not be rebuilt for comparison.")

    if len(pool) < 2:
        st.info("At least two valid quotations are needed to compare.")
        return

    recommendation = render_recommendation(pool)

    if recommendation is not None:
        # Optional and collapsed: it needs ratings the app cannot get from
        # the PDFs, so it should not take up space until the user wants it.
        with st.expander("Optional: add your own service quality ratings"):
            render_scorecard(recommendation)

    st.markdown("**Totals stated by each supplier**")
    totals_rows = []
    for entry in pool:
        totals = entry["data"].get("totals")
        totals = totals if isinstance(totals, dict) else {}
        quality = entry["data"].get("text_quality") or {}
        if not isinstance(quality, dict):
            quality = {}
        vendor = entry["data"].get("vendor") or {}
        if not isinstance(vendor, dict):
            vendor = {}
        totals_rows.append({
            "Quotation": entry["label"],
            "Subtotal": _shown(totals.get("subtotal")),
            "Tax rate (%)": _shown(totals.get("tax_rate")),
            "Tax total": _shown(totals.get("tax_total")),
            "Grand total": _shown(totals.get("grand_total")),
            "Currency": _shown(vendor.get("currency")),
            "Currency evidence": quality.get("currency_status") or "unknown",
            "Validation": validation_dict(entry["data"]).get(
                "status", "unknown"
            ),
        })
    show_table(totals_rows)
    st.caption(
        "These are the totals printed on each quotation. Unknown means the "
        "value was not extracted. What each total includes can differ, so "
        "compare scope before comparing totals."
    )

    st.markdown("**Closest matching items**")

    if st.session_state.get("compare_base", 0) >= len(pool):
        st.session_state["compare_base"] = 0

    col_base, col_threshold = st.columns([2, 1])
    base_index = col_base.selectbox(
        "Base supplier",
        list(range(len(pool))),
        format_func=lambda i: pool[i]["label"],
        key="compare_base",
    )
    threshold = col_threshold.slider(
        "Minimum name similarity",
        min_value=0.50,
        max_value=0.95,
        value=0.65,
        step=0.05,
        key="compare_threshold",
    )

    rows = build_comparison_rows(pool, base_index, threshold)

    if rows:
        show_table(rows)
    else:
        st.info("The base supplier has no line items to compare.")

    st.caption(
        "These are candidates only; no match is confirmed automatically. "
        "Unit prices and line totals are shown only when the quantity and "
        "unit match. Not comparable means the price gap should not be used."
    )


# ---------------------------------------------------------
# Rendering
# ---------------------------------------------------------

def render_quotation(data: dict[str, Any], key_prefix: str, filename: str,
                     page: int | None) -> None:
    vendor = data.get("vendor") or {}
    items = data.get("items") or []
    totals = data.get("totals") or {}
    validation = validation_dict(data)
    issues = get_issues(validation)

    if not isinstance(vendor, dict):
        vendor = {}

    st.markdown("**Supplier details**")
    cols = st.columns(3)
    cols[0].write(f"**Name:** {vendor.get('name') or 'Not extracted'}")
    cols[1].write(
        f"**Quotation no.:** {vendor.get('quotation_number') or 'Not available'}"
    )
    cols[2].write(
        f"**Date:** {vendor.get('quotation_date') or 'Not available'}"
    )

    quality = data.get("text_quality") or {}
    if not isinstance(quality, dict):
        quality = {}
    currency_status = quality.get("currency_status") or "unknown"
    currency_code = quality.get("currency_code")
    if currency_status == "identified" and currency_code:
        source = quality.get("currency_source") or "the document"
        currency_line = (
            f"Identified as {currency_code} from {str(source).replace('_', ' ')}."
        )
    else:
        currency_line = (
            "Unknown. No explicit currency code or unambiguous symbol "
            "was found."
        )
    stated = vendor.get("currency")
    st.write(
        f"**Currency field:** {_shown(stated)}. **Evidence:** {currency_line}"
    )

    st.markdown("**Line items**")
    if items:
        show_table([
            {
                "Item ID": item.get("item_id"),
                "Description": item.get("original_description"),
                "Normalized name": item.get("normalized_name"),
                "Quantity": _shown(item.get("quantity")),
                "Unit": _shown(item.get("unit")),
                "Unit price": _shown(item.get("unit_price")),
                "Discount": _shown(item.get("discount")),
                "Tax rate": _shown(item.get("tax_rate")),
                "Line total": _shown(item.get("line_total")),
                "Match status": item.get("match_status"),
                "Warnings": "; ".join(
                    str(w) for w in (item.get("warnings") or [])
                ),
            }
            for item in items
            if isinstance(item, dict)
        ])
    else:
        st.warning("No line items were extracted.")

    st.markdown("**Supplier-stated totals**")
    st.caption(
        "These are the amounts printed on the quotation. Unknown is not zero."
    )
    if isinstance(totals, dict) and totals:
        show_table([{
            "Subtotal": _shown(totals.get("subtotal")),
            "Tax rate (%)": _shown(totals.get("tax_rate")),
            "Tax total": _shown(totals.get("tax_total")),
            "Grand total": _shown(totals.get("grand_total")),
        }])
    else:
        st.write("No totals were extracted.")

    extraction_issues = [
        issue for issue in issues if issue.startswith("Extraction uncertainty")
    ]
    diagnostic_issues = [
        issue for issue in issues if issue.startswith("Diagnostic only")
    ]
    review_issues = [
        issue for issue in issues
        if issue not in extraction_issues and issue not in diagnostic_issues
    ]

    st.markdown("**Validation**")
    status = validation.get("status", "unknown")
    st.write(f"Status: `{status}`")

    if review_issues:
        st.warning("\n\n".join(f"- {issue}" for issue in review_issues))
    elif status == "valid" and not extraction_issues:
        st.success("No validation issues were reported.")
    elif not review_issues and not extraction_issues:
        st.info(
            "No issue list was returned. Review the quotation against "
            "the original PDF."
        )

    if extraction_issues:
        st.markdown("**Extraction uncertainty**")
        st.info("\n\n".join(f"- {issue}" for issue in extraction_issues))

    if diagnostic_issues:
        st.markdown("**Diagnostic calculations**")
        st.caption("These figures were not printed on the quotation.")
        st.info("\n\n".join(f"- {issue}" for issue in diagnostic_issues))

    notes = data.get("extraction_notes")
    if notes:
        st.markdown("**Extraction notes**")
        st.write(notes)

    st.caption("Complete structured data")
    st.json(data, expanded=False)

    suffix = f"_page{page}" if page else ""
    st.download_button(
        "Download this quotation (JSON)",
        data=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{safe_stem(filename)}{suffix}_quotation.json",
        mime="application/json",
        key=f"download_{key_prefix}",
    )


def _supplier_refs(results: list[dict[str, Any]]) -> list[SupplierRef]:
    """One record per extracted quotation, not per supplier name."""
    refs: list[SupplierRef] = []
    for result in results:
        file_hash = str(result.get("file_hash") or "").strip()
        if not file_hash:
            continue
        for occurrence, entry in enumerate(result.get("quotations") or []):
            data = entry.get("data") if isinstance(entry, dict) else None
            vendor = data.get("vendor") if isinstance(data, dict) else None
            if not isinstance(vendor, dict):
                vendor = {}
            page = entry.get("page") if isinstance(entry, dict) else None
            refs.append(SupplierRef(
                file_hash=file_hash,
                filename=str(result.get("filename") or ""),
                page=page if isinstance(page, int) else None,
                quotation_number=vendor.get("quotation_number"),
                supplier_name=vendor.get("name"),
                occurrence=occurrence,
            ))
    return refs


def _saved_rating(store: dict[str, Any], key: str) -> dict[str, Any]:
    saved = store.get(key)
    return saved if isinstance(saved, dict) else {}


def _optional_number(value: Any) -> int | float | None:
    if value is None or value == "":
        return None
    return value


def render_supplier_ratings(results: list[dict[str, Any]]) -> None:
    """Enter public supplier ratings with safe blank-field handling."""
    refs = _supplier_refs(results)
    if not refs:
        return

    st.subheader("Supplier ratings")
    st.caption(
        "Enter a public rating from 1 to 5 and the number of reviews. "
        "Leave either field blank if the information is unavailable. "
        "Review count is displayed but does not affect the score."
    )
    store: dict[str, Any] = st.session_state.setdefault(SUPPLIER_RATINGS_KEY, {})

    for ref in refs:
        key = supplier_key(ref)
        saved = _saved_rating(store, key)
        where = f"Page {ref.page}" if ref.page is not None else "Whole PDF"

        # Text inputs support blank values consistently across Streamlit versions.
        with st.form(f"supplier_rating_{key}"):
            st.markdown(f"**{ref.supplier_name or 'Unnamed supplier'}**")
            st.caption(
                f"{ref.filename} · {where} · "
                f"Quotation no.: {ref.quotation_number or 'Unknown'}"
            )
            public_col, count_col = st.columns(2)
            saved_public = saved.get("public_rating")
            saved_count = saved.get("review_count")
            public_text = public_col.text_input(
                "Public rating (1–5)",
                value="" if saved_public is None else str(saved_public),
                key=f"public_{key}",
                placeholder="e.g. 4.5",
            )
            count_text = count_col.text_input(
                "Review count",
                value="" if saved_count is None else str(saved_count),
                key=f"reviews_{key}",
                placeholder="e.g. 120",
            )
            submitted = st.form_submit_button("Save Rating", type="primary")

        if saved:
            public = saved.get("public_rating")
            reviews = saved.get("review_count")
            st.caption(
                "Previously saved — Public rating: "
                f"{'Not entered' if public is None else public}; review count: "
                f"{'Not entered' if reviews is None else reviews}."
            )

        if not submitted:
            continue

        raw_public: Any = public_text.strip() or None
        raw_count: Any = count_text.strip() or None
        try:
            record = validate_supplier_rating(
                ref, public_rating=raw_public, review_count=raw_count
            )
        except RatingInputError as exc:
            for message in exc.errors:
                st.error(message)
            continue
        except Exception:
            logger.exception("Unexpected error while saving supplier rating")
            st.error("The rating could not be saved. Check the values and try again.")
            continue

        store[key] = record.to_dict()
        st.session_state[SUPPLIER_RATINGS_KEY] = store
        st.success(f"Rating saved for {ref.supplier_name or 'this quotation' }.")

def _supplier_label(
    name: str,
    number: Any,
    page: Any,
    filename: Any,
) -> str:
    if number:
        return f"{name} ({number})"
    if isinstance(page, int):
        return f"{name} (page {page})"
    if filename:
        return f"{name} ({filename})"
    return name


def _money_cell(total: Decimal | None, currency: str | None) -> str:
    if total is None:
        return "Not provided"
    prefix = f"{currency} " if currency else ""
    return f"{prefix}{total:,.2f}"


def _score_cell(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    return f"{value:.1f}"


def _rating_cell(value: float | None) -> str:
    if value is None:
        return "Not provided"
    return f"{value:.1f}"


def _count_cell(value: int | None) -> str:
    if value is None:
        return "Not provided"
    return str(value)


def _comparison_inputs(results: list[dict[str, Any]]) -> list[ComparisonInput]:
    """Pair each extracted quotation with the rating saved for its key."""
    store = st.session_state.get(SUPPLIER_RATINGS_KEY, {})
    built: list[ComparisonInput] = []

    for result in results:
        file_hash = str(result.get("file_hash") or "").strip()
        if not file_hash:
            continue
        for occurrence, entry in enumerate(result.get("quotations") or []):
            if not isinstance(entry, dict):
                continue
            data = entry.get("data")
            vendor = data.get("vendor") if isinstance(data, dict) else None
            if not isinstance(vendor, dict):
                vendor = {}
            page = entry.get("page")
            ref = SupplierRef(
                file_hash=file_hash,
                filename=str(result.get("filename") or ""),
                page=page if isinstance(page, int) else None,
                quotation_number=vendor.get("quotation_number"),
                supplier_name=vendor.get("name"),
                occurrence=occurrence,
            )
            key = supplier_key(ref)
            saved = store.get(key) if isinstance(store.get(key), dict) else {}
            name = vendor.get("name") or "Unnamed supplier"
            number = vendor.get("quotation_number")
            label = _supplier_label(name, number, page, result.get("filename"))

            try:
                quotation = Quotation.model_validate(data)
            except Exception:
                built.append(ComparisonInput(
                    label=label,
                    supplier_key=key,
                    grand_total=None,
                    currency=None,
                    eligible=False,
                    public_rating=saved.get("public_rating"),
                    review_count=saved.get("review_count"),
                    warnings=("This quotation could not be compared.",),
                    critical=True,
                ))
                continue

            issues = list(quotation.validation.issues)
            warnings = [issue for issue in issues if is_critical_issue(issue)]
            grand = quotation.totals.grand_total
            subtotal = quotation.totals.subtotal
            tax_total = quotation.totals.tax_total
            reasons: list[str] = []
            if quotation.validation.status == "invalid":
                reasons.append("Validation status is invalid.")
            if grand is None:
                reasons.append("No quotation total was extracted.")
            elif grand <= 0:
                reasons.append("The quotation total is zero.")
            elif (
                subtotal is not None
                and tax_total is not None
                and abs((subtotal + tax_total) - grand) > Decimal("0.02")
            ):
                reasons.append(
                    "The stated subtotal plus tax does not equal the "
                    "quotation total."
                )
            for reason in reasons:
                if reason not in warnings:
                    warnings.append(reason)
            if quotation.vendor.name:
                name = quotation.vendor.name
                number = quotation.vendor.quotation_number
                label = _supplier_label(
                    name, number, page, result.get("filename")
                )

            built.append(ComparisonInput(
                label=label,
                supplier_key=key,
                grand_total=grand,
                currency=quotation.vendor.currency,
                eligible=not reasons,
                public_rating=saved.get("public_rating"),
                review_count=saved.get("review_count"),
                warnings=tuple(warnings),
                critical=bool(warnings) or quotation.validation.status == "invalid",
            ))
    return built


def _needs_pdf_check(results: list[dict[str, Any]]) -> bool:
    for result in results:
        for entry in result.get("quotations") or []:
            data = entry.get("data") if isinstance(entry, dict) else None
            validation = data.get("validation") if isinstance(data, dict) else None
            issues = validation.get("issues") if isinstance(validation, dict) else None
            if any(is_advisory_extraction_warning(issue) for issue in issues or []):
                return True
    return False


def _render_best_pick(ranking) -> None:
    best = ranking.best
    with st.container(border=True, key="best_pick"):
        st.subheader("Best Pick")
        if best.pending:
            st.info(f"{best.headline}. {best.explanation}")
            return

        st.success(best.headline)
        total = _money_cell(best.grand_total, best.currency)
        rating = _rating_cell(best.public_rating)
        reviews = _count_cell(best.review_count)
        score = _score_cell(best.combined_score)
        st.write(
            f"**Combined score:** {score} · **Quotation total:** {total} · "
            f"**Public rating:** {rating} · **Review count:** {reviews}"
        )
        st.write(best.explanation)
        if best.warnings:
            st.warning(" ".join(best.warnings))
        others = [
            f"{row.label}: {row.warnings[0]}"
            for row in ranking.rows
            if row.critical and row.warnings
            and row.supplier_key != best.supplier_key
        ]
        if others:
            st.warning("Critical review items: " + "; ".join(others))


def _render_comparison_table(ranking) -> None:
    st.subheader("Supplier comparison")
    if ranking.price_only_reason:
        st.caption(ranking.price_only_reason)

    show_table([
        {
            "Rank": (
                str(item.rank) if item.rank is not None else "Unavailable"
            ),
            "Supplier": item.label,
            "Quotation total": _money_cell(item.grand_total, item.currency),
            "Public rating": _rating_cell(item.public_rating),
            "Review count": _count_cell(item.review_count),
            "Price score": _score_cell(item.price_score),
            "Quality score": _score_cell(item.quality_score),
            "Combined score": _score_cell(item.combined_score),
            "Status": item.status,
        }
        for item in ranking.rows
    ])


@st.fragment
def render_weighted_comparison(results: list[dict[str, Any]]) -> None:
    """Rescore the stored quotations. Does not read PDFs or call Ollama."""
    price_weight = st.slider(
        "Price weight",
        min_value=0,
        max_value=100,
        value=DEFAULT_PRICE_WEIGHT_PERCENT,
        step=5,
        format="%d%%",
        key="comparison_price_weight",
        help=(
            "Quality receives the remaining share, so the two weights "
            "sum to 100%. Scores update from the saved quotations."
        ),
    )
    price_weight = int(price_weight)
    quality_weight = 100 - price_weight
    st.caption(f"Price {price_weight}% · Quality {quality_weight}%")

    inputs = _comparison_inputs(results)
    if not inputs:
        st.info("No suppliers were extracted to compare.")
        return

    ranking = rank_comparisons(inputs, price_weight)
    _render_best_pick(ranking)
    _render_comparison_table(ranking)


def render_results(results: list[dict[str, Any]]) -> None:
    for result in results:
        if result.get("success"):
            continue
        st.warning(
            f"{result.get('filename', 'A file')}: "
            f"{result.get('error', 'This file could not be processed.')}"
        )

    all_quotations = [q for r in results for q in r.get("quotations", [])]
    if not all_quotations:
        st.info("No quotations were extracted.")
        return

    if _needs_pdf_check(results):
        st.warning(
            "Extracted values should be checked against the original PDF. "
            "Unreadable characters were left unchanged."
        )

    render_weighted_comparison(results)
    render_supplier_ratings(results)

    saved_ratings = [
        value
        for value in st.session_state.get(SUPPLIER_RATINGS_KEY, {}).values()
        if isinstance(value, dict)
        and value.get("supplier_key") in {
            supplier_key(ref) for ref in _supplier_refs(results)
        }
    ]
    with st.expander("Download results"):
        e1, e2 = st.columns(2)
        e1.download_button(
            "Download all results (JSON)",
            data=json.dumps(
                build_export(results, saved_ratings),
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
            file_name="quotecompare_results.json",
            mime="application/json",
            key="download_all_json",
        )
        e2.download_button(
            "Download all line items (CSV)",
            data=build_line_items_csv(results).encode("utf-8-sig"),
            file_name="quotecompare_line_items.csv",
            mime="text/csv",
            key="download_all_csv",
        )



# ---------------------------------------------------------
# Page layout
# ---------------------------------------------------------

hero = not bool(st.session_state.get(RESULTS_KEY))
_apply_theme(hero=hero)

with st.sidebar:
    st.header("Settings")

    mode = st.radio(
        "How should each PDF be read?",
        [MODE_PER_PAGE, MODE_WHOLE],
        help=(
            "Per-page mode sends one page at a time to the model. It is "
            "faster and gives one record per supplier when a PDF bundles "
            "several quotations. Use whole-PDF mode only when one "
            "quotation spans several pages."
        ),
    )

    st.caption(
        "Per-page mode assumes one quotation per page. If a quotation "
        "spans several pages, or a page holds two quotations, use "
        "whole-PDF mode instead."
    )

    st.write(f"**Model:** {OLLAMA_MODEL}")
    if OLLAMA_TIMEOUT < MIN_RECOMMENDED_TIMEOUT:
        st.warning(
            f"The local model timeout is {OLLAMA_TIMEOUT} seconds, which "
            "is short for this model."
        )

    ollama_ok, ollama_message = check_ollama()
    if ollama_ok:
        st.success(ollama_message)
    else:
        st.warning(ollama_message)

    if st.button("Re-check Ollama"):
        check_ollama.clear()
        st.rerun()

    st.caption(
        "Scanned PDFs need OCR support, which is not included in this "
        "version."
    )

    if st.button("Clear cached results"):
        for key in (RESULTS_KEY, SIGNATURE_KEY, PAGE_CACHE_KEY):
            st.session_state.pop(key, None)
        st.rerun()


# ---------------------------------------------------------
# Upload PDFs
# ---------------------------------------------------------

if hero:
    st.markdown(
        """
        <div class="qc-brand">QUOTECOMPARE</div>
        <h1 class="qc-headline">Choose a supplier with a clear score.</h1>
        <p class="qc-lead">
        Compare quotation totals and public ratings on this computer.
        The Best Pick is the highest complete score for the weights you set.
        </p>
        """,
        unsafe_allow_html=True,
    )
    upload_panel = st.container(key="hero_panel")
else:
    upload_panel = st.sidebar.expander("Upload quotations", expanded=False)

with upload_panel:
    st.markdown(
        "Privacy: this app processes PDFs locally and calls your "
        "configured local Ollama pipeline. Do not configure a "
        "cloud-based model endpoint if you require fully local processing."
    )
    uploaded_files = st.file_uploader(
        "Upload supplier quotation PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more quotations. Bundled PDFs are supported.",
        key="quotation_pdfs",
    )
    if not ollama_ok:
        st.caption(
            "Ollama did not respond to the health check. You can still "
            "try processing, but it will probably fail."
        )
    process_clicked = st.button(
        "Process Quotations",
        type="primary",
        key="process_quotations",
    )
    if process_clicked and not uploaded_files:
        st.warning("Add at least one PDF before processing.")

if uploaded_files and len(uploaded_files) > MAX_FILES:
    st.error(
        f"You selected {len(uploaded_files)} files. "
        f"Please select at most {MAX_FILES} files per batch."
    )
    st.stop()

uploaded_records: list[dict[str, Any]] = []
for uploaded_file in uploaded_files or []:
    raw = uploaded_file.getvalue()
    uploaded_records.append({
        "filename": uploaded_file.name,
        "bytes": raw,
        "hash": sha256_bytes(raw),
    })

hashes = [r["hash"] for r in uploaded_records]
if len(set(hashes)) != len(hashes):
    st.warning(
        "Some uploaded files are identical. They will be processed "
        "separately; remove duplicates to save time."
    )

if hero:
    for record in uploaded_records:
        size_mb = len(record["bytes"]) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            st.error(
                f"{record['filename']} exceeds the "
                f"{MAX_FILE_SIZE_MB} MB limit."
            )
            continue
        try:
            preview = read_pdf_cached(record["hash"], record["bytes"])
        except Exception as exc:
            st.error(f"{record['filename']} could not be read: {exc}")
            continue
        affected = [
            d["page"] for d in preview["page_diagnostics"]
            if d.get("unreadable_chars")
        ]
        if affected:
            st.warning(
                f"{record['filename']}: unreadable characters on page(s) "
                f"{', '.join(map(str, affected))}. Extracted values should "
                "be checked against the original PDF."
            )
        if preview["pages_with_text"] == 0:
            st.warning(
                f"{record['filename']}: no usable text was found. This PDF "
                "may be image-only. OCR was not run."
            )

signature = {
    "hashes": hashes,
    "mode": mode,
    "extractor": extraction_fingerprint(),
}

if process_clicked and uploaded_records:
    results: list[dict[str, Any]] = []
    progress = st.progress(0.0)
    status = st.empty()
    live = st.empty()
    total_files = len(uploaded_records)

    def live_hook(info: dict[str, Any]) -> None:
        elapsed = int(info["elapsed"])

        live.info(f"Processing quotations… {elapsed}s")

    set_progress_hook(live_hook)

    try:
        for index, record in enumerate(uploaded_records):

            def report(
                message: str, fraction: float, _i: int = index
            ) -> None:
                status.write(f"File {_i + 1} of {total_files}")
                progress.progress(min((_i + fraction) / total_files, 1.0))

            results.append(
                process_one_file(
                    record["filename"], record["bytes"], mode, report
                )
            )
    finally:
        set_progress_hook(None)
        live.empty()

    st.session_state[RESULTS_KEY] = results
    st.session_state[SIGNATURE_KEY] = signature
    progress.progress(1.0)
    status.success("Processing batch finished.")


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

saved_results = st.session_state.get(RESULTS_KEY)
saved_signature = st.session_state.get(SIGNATURE_KEY)

files_changed = bool(hashes) and saved_signature != signature
if saved_results and not files_changed:
    render_results(saved_results)
elif saved_results:
    st.info(
        "The uploaded files or settings changed since the last run. "
        "Process them again to refresh the results."
    )