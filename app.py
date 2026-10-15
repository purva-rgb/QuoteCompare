from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import time
from datetime import datetime
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
from engine.pipeline import (
    QuotationProcessingError,
    process_quotation_text,
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

# Characters PDFs draw when a font lacks a glyph (for example a rupee sign).
# Amounts and currency near them cannot be trusted without checking the PDF.
PLACEHOLDER_CHARS = "\u25a0\u25a1\u25af\ufffd"
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Small helpers
# ---------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def count_placeholders(text: str) -> int:
    """Count characters that stand in for a glyph the PDF font lacked."""
    return sum(text.count(ch) for ch in PLACEHOLDER_CHARS)


def unreadable_chars_for(result: dict[str, Any], page: int | None) -> int:
    """Placeholder-character count for one page, or the whole PDF."""
    info = result.get("pdf_result") or {}
    diagnostics = info.get("page_diagnostics") or []

    if page is None:
        return sum(d.get("unreadable_chars", 0) for d in diagnostics)

    if 1 <= page <= len(diagnostics):
        return int(diagnostics[page - 1].get("unreadable_chars", 0))

    return 0


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
        diagnostics.append({
            "page": page_number,
            "characters": len(text),
            "has_text": len(text) >= MIN_PAGE_CHARS,
            "unreadable_chars": count_placeholders(text),
        })

    combined = "\n\n".join(
        f"--- Page {i} ---\n{t}"
        for i, t in enumerate(page_texts, start=1)
        if t
    ).strip()

    return {
        "text": combined,
        "page_texts": page_texts,
        "page_count": page_count,
        "pages_with_text": sum(d["has_text"] for d in diagnostics),
        "page_diagnostics": diagnostics,
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
                "No readable text was found. "
                "This may be a scanned PDF and require local OCR."
            )
        )

    # Build the list of (page_number_or_None, text) jobs.
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
            base["quotations"].append({"page": page_number, "data": data})
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

def build_export(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
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


def build_line_items_csv(results: list[dict[str, Any]]) -> str:
    columns = [
        "file", "page", "supplier", "quotation_number", "quotation_date",
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

            for item in data.get("items") or []:
                if not isinstance(item, dict):
                    continue
                writer.writerow({
                    "file": result["filename"],
                    "page": entry["page"] or "",
                    "supplier": vendor.get("name"),
                    "quotation_number": vendor.get("quotation_number"),
                    "quotation_date": vendor.get("quotation_date"),
                    "item_id": item.get("item_id"),
                    "description": item.get("original_description"),
                    "normalized_name": item.get("normalized_name"),
                    "quantity": item.get("quantity"),
                    "unit": item.get("unit"),
                    "unit_price": item.get("unit_price"),
                    "discount": item.get("discount"),
                    "tax_rate": item.get("tax_rate"),
                    "line_total": item.get("line_total"),
                    "match_status": item.get("match_status"),
                    "warnings": "; ".join(
                        str(w) for w in (item.get("warnings") or [])
                    ),
                })

    return buffer.getvalue()


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

    st.markdown("**Line items**")
    if items:
        show_table([
            {
                "Item ID": item.get("item_id"),
                "Description": item.get("original_description"),
                "Normalized name": item.get("normalized_name"),
                "Quantity": item.get("quantity"),
                "Unit": item.get("unit"),
                "Unit price": item.get("unit_price"),
                "Discount": item.get("discount"),
                "Tax rate": item.get("tax_rate"),
                "Line total": item.get("line_total"),
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

    st.markdown("**Totals**")
    if isinstance(totals, dict) and totals:
        st.json(totals)
    else:
        st.write("No totals were extracted.")

    st.markdown("**Validation**")
    status = validation.get("status", "unknown")
    st.write(f"Status: `{status}`")

    if issues:
        for issue in issues:
            st.warning(issue)
    elif status == "valid":
        st.success("No validation issues were reported.")
    else:
        st.info(
            "No issue list was returned. Review the quotation against "
            "the original PDF."
        )

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


def render_results(results: list[dict[str, Any]]) -> None:
    st.divider()
    st.header("Results")

    all_quotations = [q for r in results for q in r.get("quotations", [])]
    page_error_count = sum(len(r.get("page_errors", [])) for r in results)
    with_issues = sum(
        1 for q in all_quotations
        if get_issues(validation_dict(q["data"]))
        or validation_dict(q["data"]).get("status") not in (None, "valid")
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Files", len(results))
    c2.metric("Quotations extracted", len(all_quotations))
    c3.metric("Pages that failed", page_error_count)
    c4.metric("With validation issues", with_issues)

    for file_index, result in enumerate(results):
        has_problem = (
            not result["success"] or bool(result.get("page_errors"))
        )
        icon = "⚠️ " if has_problem else "✅ "
        elapsed = result.get("elapsed_seconds")
        title = result["filename"] + (
            f" ({elapsed}s)" if elapsed is not None else ""
        )

        with st.expander(icon + title, expanded=has_problem):
            if not result["success"]:
                st.error(result.get("error", "Processing failed."))

            pdf_info = result.get("pdf_result")
            if pdf_info:
                st.caption(
                    f"Pages: {pdf_info['page_count']} · "
                    f"Pages with text: {pdf_info['pages_with_text']}"
                )

            for err in result.get("page_errors", []):
                where = (
                    f"Page {err['page']}" if err["page"] is not None
                    else "Whole PDF"
                )
                st.error(f"{where}: {err['error']}")

            for skipped in result.get("skipped_pages", []):
                st.info(f"Page {skipped['page']} skipped: {skipped['reason']}.")

            for q_index, entry in enumerate(result.get("quotations", [])):
                label = (
                    f"Page {entry['page']}" if entry["page"] is not None
                    else "Whole PDF"
                )
                vendor = entry["data"].get("vendor")
                name = vendor.get("name") if isinstance(vendor, dict) else None

                with st.container(border=True):
                    st.subheader(f"{label}" + (f" · {name}" if name else ""))

                    unreadable = unreadable_chars_for(result, entry["page"])
                    if unreadable:
                        st.warning(
                            f"This source page contains {unreadable} "
                            "unreadable placeholder character(s), usually a "
                            "currency symbol the PDF font could not draw. "
                            "Check amounts and currency against the PDF."
                        )

                    render_quotation(
                        entry["data"],
                        key_prefix=f"{file_index}_{q_index}",
                        filename=result["filename"],
                        page=entry["page"],
                    )

    if all_quotations:
        st.subheader("Export")
        e1, e2 = st.columns(2)
        e1.download_button(
            "Download all results (JSON)",
            data=json.dumps(
                build_export(results), ensure_ascii=False, indent=2
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

    if len(all_quotations) >= 2:
        st.info(
            "Multiple quotations were processed. The next step is to match "
            "equivalent items across suppliers. Do not compare line totals "
            "until units, quantities, discounts and tax treatment are "
            "checked."
        )


# ---------------------------------------------------------
# Page layout
# ---------------------------------------------------------

st.title("📄 QuoteCompare")
st.caption("Compare supplier quotations privately on your computer.")

st.info(
    "Privacy: this app processes PDFs locally and calls your configured "
    "local Ollama pipeline. Do not configure a cloud-based model endpoint "
    "if you require fully local processing."
)

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

    st.write("**Processing:** Local Python + Ollama")
    st.write(f"**Model:** {OLLAMA_MODEL}")
    st.write(f"**Ollama URL:** {OLLAMA_URL}")

    st.write(f"**Ollama timeout:** {OLLAMA_TIMEOUT} s")
    if OLLAMA_TIMEOUT < MIN_RECOMMENDED_TIMEOUT:
        st.warning(
            f"OLLAMA_TIMEOUT is {OLLAMA_TIMEOUT} s, which is short for a "
            "7B model. Set the OLLAMA_TIMEOUT environment variable to 600 "
            "or higher."
        )
    st.write(f"**Maximum file size:** {MAX_FILE_SIZE_MB} MB")
    st.write(f"**Maximum pages per PDF:** {MAX_PAGES}")
    st.write(f"**Maximum files per batch:** {MAX_FILES}")

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

uploaded_files = st.file_uploader(
    "Upload supplier quotation PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select one or more quotations. Bundled PDFs are supported.",
)

if not uploaded_files:
    st.markdown(
        """
        ### Get started

        1. Upload your supplier quotation PDFs.
        2. Review the extracted text.
        3. Run local AI processing.
        4. Check validation warnings and download the results.
        """
    )
    st.stop()

if len(uploaded_files) > MAX_FILES:
    st.error(
        f"You selected {len(uploaded_files)} files. "
        f"Please select at most {MAX_FILES} files per batch."
    )
    st.stop()

uploaded_records: list[dict[str, Any]] = []
for uploaded_file in uploaded_files:
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

st.subheader("Uploaded quotations")

for index, record in enumerate(uploaded_records):
    size_mb = len(record["bytes"]) / (1024 * 1024)

    with st.expander(f"📄 {record['filename']} — {size_mb:.2f} MB"):
        st.write(f"SHA-256: `{record['hash']}`")

        if size_mb > MAX_FILE_SIZE_MB:
            st.error(f"This file exceeds the {MAX_FILE_SIZE_MB} MB limit.")
            continue

        try:
            preview = read_pdf_cached(record["hash"], record["bytes"])
        except Exception as exc:
            st.error(f"Cannot preview this PDF: {exc}")
            continue

        st.write(
            f"Pages: {preview['page_count']} · "
            f"Pages with text: {preview['pages_with_text']}"
        )

        affected = [
            d["page"] for d in preview["page_diagnostics"]
            if d.get("unreadable_chars")
        ]
        if affected:
            st.warning(
                "Unreadable placeholder characters (such as a box) were "
                f"found on page(s) {', '.join(map(str, affected))}. The PDF's "
                "font is missing a symbol, often a currency sign. Check "
                "amounts and currency on those pages against the PDF."
            )

        if preview["pages_with_text"] == 0:
            st.warning("No selectable text found. Local OCR is needed.")
            continue

        if mode == MODE_PER_PAGE:
            st.caption(
                f"{preview['pages_with_text']} page(s) will be sent to the "
                "model as separate quotations."
            )

        st.text_area(
            "Extracted PDF text",
            preview["text"],
            height=220,
            key=f"preview_{index}_{record['hash']}",
        )
        st.caption("Page extraction diagnostics")
        show_table(preview["page_diagnostics"])


# ---------------------------------------------------------
# Process using the local Ollama pipeline
# ---------------------------------------------------------

st.divider()
st.subheader("Process quotations")

st.warning(
    "AI-extracted values must be checked against the original PDFs. "
    "A successful run does not guarantee that every value is correct."
)

if not ollama_ok:
    st.caption(
        "Ollama did not respond to the health check in the sidebar. "
        "You can still try processing, but it will probably fail."
    )

signature = {
    "hashes": hashes,
    "mode": mode,
    "extractor": extraction_fingerprint(),
}

if st.button("🚀 Process quotations locally", type="primary"):
    results: list[dict[str, Any]] = []
    progress = st.progress(0.0)
    status = st.empty()
    live = st.empty()
    total_files = len(uploaded_records)

    def live_hook(info: dict[str, Any]) -> None:
        elapsed = int(info["elapsed"])

        if info["phase"] == "waiting":
            live.info(
                "Sent to the model. Waiting for the first output. Loading "
                "the model and reading the page can take a minute or two, "
                "especially on a CPU."
            )
            return

        rate = info["tokens"] / info["elapsed"] if info["elapsed"] > 0 else 0
        live.info(
            f"Model is generating: about {info['tokens']} tokens · "
            f"{elapsed}s elapsed · {rate:.1f} tokens/s"
        )

    set_progress_hook(live_hook)

    try:
        for index, record in enumerate(uploaded_records):

            def report(
                message: str, fraction: float, _i: int = index
            ) -> None:
                status.write(f"File {_i + 1} of {total_files} — {message}")
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

if saved_results and saved_signature == signature:
    render_results(saved_results)
elif saved_results:
    st.info(
        "The uploaded files or settings changed since the last run. "
        "Process them again to refresh the results."
    )