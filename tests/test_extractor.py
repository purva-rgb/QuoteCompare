from decimal import Decimal

from engine.extractor import _apply_stated_facts
from engine.schemas import Quotation, QuotationItem, QuotationTotals, Vendor

PAGE = """Supplier quotation Page 1
QUOTATION
BluePeak Events & Catering
Prepared for: ABC Business Solutions Event date: 20 November 2026
Quotation No. BP-2026-114 Date 15 October 2026
Validity 15 days Payment terms See terms below
Description Quantity / basis Amount (INR)
Venue and hall setup 1 package INR 45,000.00
Catering 100 guests x INR 850 INR 85,000.00
Subtotal INR 182,000.00
GST (18%) INR 32,760.00
Total payable INR 214,760.00
"""


def make(**totals):
    return Quotation(
        vendor=Vendor(name="BluePeak"),
        items=[QuotationItem(item_id="a", original_description="Catering")],
        totals=QuotationTotals(**totals),
    )


def test_dropped_totals_are_filled_from_labelled_lines():
    quotation = make(subtotal=Decimal("182000"))

    _apply_stated_facts(quotation, PAGE)

    assert quotation.totals.tax_total == Decimal("32760.00")
    assert quotation.totals.grand_total == Decimal("214760.00")
    assert quotation.totals.tax_rate == Decimal("18")
    assert "tax total" in quotation.extraction_notes[-1]
    assert "grand total" in quotation.extraction_notes[-1]


def test_all_header_facts_are_filled():
    quotation = make()

    _apply_stated_facts(quotation, PAGE)

    assert quotation.vendor.quotation_number == "BP-2026-114"
    assert quotation.vendor.quotation_date == "15 October 2026"
    assert quotation.vendor.currency == "INR"
    assert quotation.totals.subtotal == Decimal("182000.00")


def test_existing_model_values_are_kept():
    quotation = make(
        subtotal=Decimal("1"),
        tax_total=Decimal("2"),
        grand_total=Decimal("3"),
    )

    _apply_stated_facts(quotation, PAGE)

    assert quotation.totals.subtotal == Decimal("1")
    assert quotation.totals.tax_total == Decimal("2")
    assert quotation.totals.grand_total == Decimal("3")


def test_wrong_scale_tax_rate_is_corrected_from_the_page():
    quotation = make(tax_rate=Decimal("0.18"))

    _apply_stated_facts(quotation, PAGE)

    assert quotation.totals.tax_rate == Decimal("18")


def test_subtotal_line_is_never_read_as_the_grand_total():
    text = "Subtotal INR 182,000.00\nGST (18%) INR 32,760.00"
    quotation = make()

    _apply_stated_facts(quotation, text)

    assert quotation.totals.subtotal == Decimal("182000.00")
    assert quotation.totals.grand_total is None


def test_event_date_is_not_taken_as_the_quotation_date():
    text = "Event date: 20 November 2026\nOther text"
    quotation = make()

    _apply_stated_facts(quotation, text)

    assert quotation.vendor.quotation_date is None


def test_label_without_a_number_does_not_capture_a_word():
    text = "Quotation No.\nPrepared for: ABC Business Solutions"
    quotation = make()

    _apply_stated_facts(quotation, text)

    assert quotation.vendor.quotation_number is None


def test_label_and_value_on_separate_lines():
    text = "Quotation No.\nBP-2026-114\nDate\n15 October 2026"
    quotation = make()

    _apply_stated_facts(quotation, text)

    assert quotation.vendor.quotation_number == "BP-2026-114"
    assert quotation.vendor.quotation_date == "15 October 2026"


def test_ambiguous_text_with_two_quotations_stays_empty():
    text = PAGE + "\nQuotation No. CCE-26-208 Date 16 October 2026\n"
    text += "Total payable INR 214,170.00\n"
    quotation = make()

    _apply_stated_facts(quotation, text)

    assert quotation.vendor.quotation_number is None
    assert quotation.vendor.quotation_date is None
    assert quotation.totals.grand_total is None


def test_split_taxes_are_not_given_a_single_rate():
    text = "CGST (9%) INR 16,380.00\nSGST (9%) INR 16,380.00"
    quotation = make()

    _apply_stated_facts(quotation, text)

    assert quotation.totals.tax_rate is None


# The text pypdf produces for the sample PDFs: a table, so every label and
# its value are on separate lines (with boxes already replaced by INR).
TABLE_LAYOUT_PAGE = (
    "Supplier quotation\nPage 1\n QUOTATION\n BluePeak Events & Catering\n"
    "Prepared for: ABC Business Solutions    Event date: 20 November 2026\n"
    "Event: Annual Corporate Networking Evening    Location: Mumbai    "
    "Guests: 100\nQuotation No.\nBP-2026-114\nDate\n15 October 2026\n"
    "Validity\n15 days\nPayment terms\nSee terms below\nPrice breakdown\n"
    "Description\nQuantity / basis\nAmount (INR)\nVenue and hall setup\n"
    "1 package\nINR 45,000.00\nCatering\n100 guests \u00d7 INR 850\n"
    "INR 85,000.00\nSubtotal\nINR 182,000.00\nGST (18%)\nINR 32,760.00\n"
    "Total payable\nINR 214,760.00\nTerms and conditions\n"
)


def test_table_layout_with_labels_and_values_on_separate_lines():
    quotation = make()

    _apply_stated_facts(quotation, TABLE_LAYOUT_PAGE)

    assert quotation.vendor.quotation_number == "BP-2026-114"
    assert quotation.vendor.quotation_date == "15 October 2026"
    assert quotation.vendor.currency == "INR"
    assert quotation.totals.subtotal == Decimal("182000.00")
    assert quotation.totals.tax_total == Decimal("32760.00")
    assert quotation.totals.grand_total == Decimal("214760.00")
    assert quotation.totals.tax_rate == Decimal("18")