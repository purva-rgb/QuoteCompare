from engine.text_quality import (
    assess_text,
    count_suspicious,
    diagnose_page,
    font_currency_hints,
)


def test_ordinary_text_has_no_suspicious_characters():
    text = "Copper cable 2.5 sq mm, 10 meters at 500."

    assert count_suspicious(text) == 0
    quality = assess_text(text)

    assert quality["suspicious_characters"] == 0
    assert quality["currency_status"] == "unknown"
    assert quality["message"] == ""


def test_replacement_and_box_characters_are_counted_and_kept():
    original = "Subtotal \u25a0 182,000.00\nTotal \ufffd 214,760.00"

    quality = assess_text(original)

    assert quality["suspicious_characters"] == 2
    assert "\u25a0" in original
    assert "\ufffd" in original
    assert "INR" not in quality["message"] or "substituted" in quality["message"]
    assert "original text was kept" in quality["message"].lower()


def test_private_use_glyph_is_suspicious():
    assert count_suspicious("Amount \ue000 10") == 1


def test_readable_rupee_symbol_identifies_inr_without_rewriting():
    original = "Total \u20b9 500.00"
    quality = assess_text(original)

    assert quality["currency_status"] == "identified"
    assert quality["currency_code"] == "INR"
    assert quality["currency_source"] == "stated_symbol"
    assert original == "Total \u20b9 500.00"
    assert quality["suspicious_characters"] == 0


def test_dollar_sign_does_not_identify_a_currency():
    quality = assess_text("Total $500.00")

    assert quality["currency_status"] == "unknown"
    assert quality["currency_code"] is None
    assert "does not identify" in quality["message"]


def test_stated_code_is_identified_and_boxes_are_not_replaced():
    original = "Amount (INR)\nCatering \u25a1 85,000.00"
    quality = assess_text(original)

    assert original == "Amount (INR)\nCatering \u25a1 85,000.00"
    assert quality["currency_status"] == "identified"
    assert quality["currency_code"] == "INR"
    assert quality["currency_source"] == "stated_code"
    assert quality["suspicious_characters"] == 1


def test_conflicting_currency_evidence_stays_unknown():
    quality = assess_text("Amount (INR) and also USD 10")

    assert quality["currency_status"] == "unknown"
    assert quality["currency_code"] is None
    assert "more than one currency" in quality["message"]


def test_page_diagnostic_records_the_page_and_font_hint_without_editing():
    original = "Total \u25a0 10"
    diagnostic = diagnose_page(2, original, font_hints=["INR"])

    assert original == "Total \u25a0 10"
    assert diagnostic["page"] == 2
    assert diagnostic["unreadable_chars"] == 1
    assert diagnostic["currency_status"] == "unknown"
    assert "Page 2" in diagnostic["message"]
    assert "not changed" in diagnostic["message"]
    assert "INR" in diagnostic["font_hints"]


def test_font_hint_reads_an_unambiguous_glyph_name():
    class Encoding(dict):
        def get_object(self):
            return self

    class Font(dict):
        def get_object(self):
            return self

    class Fonts(dict):
        def get_object(self):
            return self

    class Resources(dict):
        def get_object(self):
            return self

    class Page(dict):
        pass

    page = Page({
        "/Resources": Resources({
            "/Font": Fonts({
                "/F1": Font({
                    "/Encoding": Encoding({
                        "/Differences": [1, "/rupee", "/dollar"],
                    }),
                }),
            }),
        }),
    })

    assert font_currency_hints(page) == ["INR"]


def test_font_hint_failure_returns_nothing():
    assert font_currency_hints(object()) == []
