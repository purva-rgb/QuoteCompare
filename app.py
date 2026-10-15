import streamlit as st
from pypdf import PdfReader
from io import BytesIO

st.set_page_config(
    page_title="QuoteCompare",
    page_icon="📄",
    layout="wide"
)

st.title("QuoteCompare")
st.write("Compare supplier quotations privately on your computer.")

uploaded_files = st.file_uploader(
    "Upload supplier quotation PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"Uploaded {len(uploaded_files)} quotation(s).")

    for uploaded_file in uploaded_files:
        with st.expander(uploaded_file.name):
            try:
                pdf = PdfReader(
                    BytesIO(uploaded_file.getvalue())
                )

                extracted_text = "\n".join(
                    page.extract_text() or ""
                    for page in pdf.pages
                )

                if extracted_text.strip():
                    st.text_area(
                        "Extracted quotation text",
                        extracted_text,
                        height=250,
                        key=uploaded_file.name
                    )
                else:
                    st.warning(
                        "No readable text found. "
                        "This may be a scanned PDF; "
                        "local OCR support will be added later."
                    )

            except Exception as error:
                st.error(f"Could not read this PDF: {error}")
