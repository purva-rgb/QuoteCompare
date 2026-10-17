"""Detect unreadable PDF glyphs and explicit currency evidence.

The original text is never rewritten. A box, replacement character, or
private-use glyph is not turned into a currency symbol. Currency is
recorded only when the text itself states one code or one unambiguous
symbol.
"""

from __future__ import annotations

import re
from typing import Any

# Glyphs PDFs emit when a font has no usable mapping. Ordinary letters,
# digits, punctuation, and readable currency symbols are not in this set.
SUSPICIOUS_CHARS = frozenset(
    "\ufffd\ufffc\ufffe"
    "\u25a0\u25a1\u25a2\u25a3\u25ab\u25aa\u25af"
    "\u25fb\u25fc\u25fd\u25fe"
    "\u2610\u2611\u2b1b\u2b1c"
)

# Codes the document may state in words or labels. This is not a default.
KNOWN_CURRENCIES = frozenset({
    "INR", "USD", "EUR", "GBP", "AED", "SGD", "AUD", "CAD",
    "JPY", "CNY", "CHF", "NZD", "ZAR",
})

# Symbols that identify one currency. "$" is intentionally absent.
UNAMBIGUOUS_SYMBOLS = {
    "₹": "INR",
    "€": "EUR",
    "£": "GBP",
}

_CODE_IN_PARENS = re.compile(r"\(([A-Z]{3})\)")
_CODE_WORD = re.compile(
    r"\b(INR|USD|EUR|GBP|AED|SGD|AUD|CAD|JPY|CNY|CHF|NZD|ZAR)\b"
)
_WORD_CURRENCIES = (
    (re.compile(r"\brupees?\b", re.IGNORECASE), "INR"),
    (re.compile(r"\bus\s+dollars?\b", re.IGNORECASE), "USD"),
    (re.compile(r"\beuros?\b", re.IGNORECASE), "EUR"),
    (re.compile(r"\bpounds?\s+sterling\b", re.IGNORECASE), "GBP"),
)

# Font glyph names that map to one currency. Dollar and yen names are
# omitted because they do not identify a single currency code.
_GLYPH_TO_CODE = {
    "rupee": "INR",
    "indianrupee": "INR",
    "rupeeindian": "INR",
    "euro": "EUR",
    "sterling": "GBP",
    "poundsterling": "GBP",
}


def is_suspicious_character(character: str) -> bool:
    """True for missing-glyph stand-ins, not for ordinary text."""
    if character in SUSPICIOUS_CHARS:
        return True

    code = ord(character)
    if 0xE000 <= code <= 0xF8FF:
        return True
    if 0xF0000 <= code <= 0xFFFFD or 0x100000 <= code <= 0x10FFFD:
        return True
    return False


def count_suspicious(text: str) -> int:
    return sum(1 for character in text if is_suspicious_character(character))


def currency_evidence(
    text: str,
) -> tuple[str, str | None, str, str]:
    """Return status, code, source, and an optional note.

    Status is "identified" or "unknown". Source is "stated_code",
    "stated_symbol", or "none". Conflicting evidence stays unknown.
    """
    stated: set[str] = set()

    for code in _CODE_IN_PARENS.findall(text):
        if code in KNOWN_CURRENCIES:
            stated.add(code)

    for code in _CODE_WORD.findall(text):
        stated.add(code)

    for pattern, code in _WORD_CURRENCIES:
        if pattern.search(text):
            stated.add(code)

    symbols = {
        code for symbol, code in UNAMBIGUOUS_SYMBOLS.items() if symbol in text
    }
    found = stated | symbols
    dollar = "$" in text or "＄" in text

    if len(found) > 1:
        listed = ", ".join(sorted(found))
        return (
            "unknown",
            None,
            "none",
            f"The text names more than one currency ({listed}), "
            "so none was selected.",
        )

    if len(found) == 1:
        code = next(iter(found))
        source = "stated_code" if code in stated else "stated_symbol"
        label = (
            "a stated currency code"
            if source == "stated_code"
            else "an unambiguous currency symbol"
        )
        return (
            "identified",
            code,
            source,
            f"Currency {code} is identified from {label}.",
        )

    if dollar:
        return (
            "unknown",
            None,
            "none",
            "A dollar sign is present. It does not identify a currency code.",
        )

    return "unknown", None, "none", ""


def assess_text(text: str) -> dict[str, Any]:
    """Describe text quality without changing the text."""
    count = count_suspicious(text)
    status, code, source, currency_note = currency_evidence(text)
    parts: list[str] = []

    if count:
        parts.append(
            f"{count} unreadable character(s) were found. "
            "The original text was kept. Verify the affected amounts "
            "manually."
        )
        if status != "identified":
            parts.append(
                "No currency was substituted for the unreadable characters."
            )

    if currency_note:
        parts.append(currency_note)

    return {
        "assessed": True,
        "suspicious_characters": count,
        "currency_status": status,
        "currency_code": code,
        "currency_source": source,
        "message": " ".join(parts),
    }


def diagnose_page(
    page: int | None,
    text: str,
    font_hints: list[str] | None = None,
) -> dict[str, Any]:
    """Page-level diagnostic for the UI. The text argument is unchanged."""
    assessment = assess_text(text)
    hints = list(dict.fromkeys(font_hints or []))
    message = assessment["message"]

    if assessment["suspicious_characters"] and hints:
        if len(hints) == 1:
            message = (
                f"{message} A font encoding names {hints[0]}, but that "
                "mapping could not be applied to a specific character, "
                "so the extracted text was not changed."
            ).strip()
        else:
            message = (
                f"{message} Font encodings name more than one currency "
                f"({', '.join(hints)}). None was substituted."
            ).strip()

    where = f"Page {page}" if page is not None else "This text"
    if assessment["suspicious_characters"]:
        message = f"{where}: {message}"

    return {
        "page": page,
        "characters": len(text),
        "suspicious_characters": assessment["suspicious_characters"],
        "unreadable_chars": assessment["suspicious_characters"],
        "currency_status": assessment["currency_status"],
        "currency_code": assessment["currency_code"],
        "currency_source": assessment["currency_source"],
        "font_hints": hints,
        "message": message,
    }


def _glyph_names(encoding: Any) -> list[str]:
    if encoding is None:
        return []

    if hasattr(encoding, "get_object"):
        try:
            encoding = encoding.get_object()
        except Exception:
            return []

    if not hasattr(encoding, "get"):
        return []

    names: list[str] = []
    for item in encoding.get("/Differences") or []:
        text = str(item).lstrip("/")
        if text and not text.isdigit():
            names.append(text)
    return names


def font_currency_hints(page: Any) -> list[str]:
    """Currency codes named by a font encoding, if any.

    This does not rewrite extracted text. A glyph name is only a hint
    that a font once contained that character. Dollar and yen names are
    ignored because they are ambiguous.
    """
    found: set[str] = set()

    try:
        resources = page.get("/Resources") if hasattr(page, "get") else None
        if resources is None:
            return []
        if hasattr(resources, "get_object"):
            resources = resources.get_object()

        fonts = resources.get("/Font") if hasattr(resources, "get") else None
        if fonts is None:
            return []
        if hasattr(fonts, "get_object"):
            fonts = fonts.get_object()

        values = fonts.values() if hasattr(fonts, "values") else []
        for font in values:
            if hasattr(font, "get_object"):
                font = font.get_object()
            encoding = font.get("/Encoding") if hasattr(font, "get") else None
            for name in _glyph_names(encoding):
                code = _GLYPH_TO_CODE.get(re.sub(r"[^a-z]", "", name.casefold()))
                if code:
                    found.add(code)
    except Exception:
        return []

    return sorted(found)
