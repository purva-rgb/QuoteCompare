# QuoteCompare

**QuoteCompare** is a local-first quotation analysis and supplier
comparison application built with Python, Streamlit, and Ollama. It
helps users extract information from supplier quotation PDFs, review
pricing details, compare offers, and evaluate suppliers using a weighted
price-and-quality score.

## Features

-   **PDF quotation processing:** Upload supplier quotation PDFs and
    extract structured information.
-   **Local AI extraction:** Use Ollama to process quotation content on
    your own machine.
-   **Data normalization:** Convert extracted values into a consistent
    format for comparison.
-   **Quotation validation:** Review totals, taxes, discounts, and other
    key quotation details.
-   **Line-item matching:** Compare related items across supplier
    quotations.
-   **Supplier comparison:** View supplier prices and quotation details
    in one place.
-   **Quality scorecards:** Record supplier ratings for reliability,
    communication, service quality, and references.
-   **Weighted ranking:** Combine price and quality scores using a
    configurable weighting, with a default of 70% price and 30% quality.
-   **Recommendation explanations:** Understand how the scoring criteria
    contribute to supplier rankings.
-   **Streamlit interface:** Work through the quotation comparison
    process in an interactive web interface.
-   **Automated tests:** Validate core engine, pipeline, scoring, and
    recommendation behavior with pytest.

## How It Works

1.  **Upload quotations** --- Select supplier quotation PDFs in the
    Streamlit interface.
2.  **Extract information** --- Ollama helps identify supplier details,
    line items, quantities, prices, taxes, and totals.
3.  **Normalize the data** --- Extracted information is organized into a
    consistent structure.
4.  **Validate quotations** --- Review calculated totals and validation
    notes.
5.  **Match line items** --- Identify corresponding items across
    supplier offers.
6.  **Evaluate suppliers** --- Enter quality ratings based on available
    evidence.
7.  **Calculate scores** --- Combine price and quality scores using the
    selected weights.
8.  **Compare and rank** --- Review supplier results and identify offers
    that best fit your evaluation criteria.

## Scoring Method

QuoteCompare uses a weighted scoring approach to support transparent
supplier comparison.

### Price score

The lowest eligible quotation total receives the highest price score.

``` text
Price Score = (Lowest Eligible Price / Supplier Price) × 100
```

### Quality score

Supplier quality can be evaluated using ratings from 1 to 5 for criteria
such as:

-   Reliability and track record
-   Communication and responsiveness
-   Quality of service or work
-   References and reviews

The average rating is converted to a 0--100 scale:

``` text
Quality Score = (Average Rating / 5) × 100
```

### Combined score

The default weighting gives price a 70% contribution and quality a 30%
contribution.

``` text
Combined Score = (Price Score × 0.70) + (Quality Score × 0.30)
```

Users can adjust the weighting to reflect their procurement priorities.
Scores are calculated from the available valid inputs, and the scorecard
provides a clear basis for comparing suppliers.

## Google Maps and Supplier Ratings

Supplier quality ratings can be informed by information reviewed on
Google Maps. Users can manually verify the correct supplier listing and
enter relevant ratings into the QuoteCompare scorecard. This workflow
does not require a Google Places API key or Google Cloud billing setup.

Manual ratings are recorded as user-provided assessments. Confirm the
supplier identity before entering a rating.

## Project Structure

``` text
QuoteCompare/
├── app.py
├── engine/
│   ├── extractor.py
│   ├── normalizer.py
│   ├── matcher.py
│   ├── validator.py
│   ├── pipeline.py
│   ├── schemas.py
│   ├── scoring.py
│   └── recommender.py
├── tests/
│   ├── test_engine.py
│   ├── test_extractor.py
│   ├── test_pipeline.py
│   ├── test_recommender.py
│   ├── test_scoring.py
│   └── test_validator.py
├── requirements.txt
├── .gitignore
└── .streamlit/
    └── config.toml
```

### Main modules

  -----------------------------------------------------------------------
  Module                              Purpose
  ----------------------------------- -----------------------------------
  `app.py`                            Streamlit user interface

  `engine/extractor.py`               Extract quotation content using the
                                      configured local model

  `engine/normalizer.py`              Normalize extracted quotation data

  `engine/matcher.py`                 Match related line items between
                                      quotations

  `engine/validator.py`               Validate quotation values and
                                      totals

  `engine/pipeline.py`                Coordinate the quotation processing
                                      workflow

  `engine/schemas.py`                 Define structured data models

  `engine/scoring.py`                 Calculate price and quality scores

  `engine/recommender.py`             Rank suppliers and explain
                                      recommendations
  -----------------------------------------------------------------------

## Technology Stack

-   **Python** --- Application logic and processing
-   **Streamlit** --- Interactive user interface
-   **Ollama** --- Local large language model runtime
-   **Pandas** --- Tabular data handling
-   **Pydantic** --- Structured data validation, where used by the
    project
-   **Pytest** --- Automated testing
-   **pypdf** --- PDF text extraction

## Requirements

-   Python 3.10 or newer
-   Ollama installed on the computer
-   A supported Ollama model downloaded locally
-   Windows, macOS, or Linux
-   The packages listed in `requirements.txt`

The default model configured for this guide is `qwen2.5:3b`. If your
project uses another model, use the model name configured in your
application.

## Installation

### 1. Open the project

Open the `QuoteCompare` folder in Visual Studio Code and launch the
integrated terminal.

### 2. Create a virtual environment

**Windows PowerShell**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS or Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

``` bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Set up Ollama

Install Ollama from [ollama.com](https://ollama.com/), then download the
configured model:

``` bash
ollama pull qwen2.5:3b
```

Verify that the model runs:

``` bash
ollama run qwen2.5:3b
```

Ollama's default local service address is:

``` text
http://127.0.0.1:11434
```

### 5. Start QuoteCompare

From the project root, run:

``` bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, typically:

``` text
http://localhost:8501
```

## Using QuoteCompare

1.  Start Ollama and confirm that the configured model is available.
2.  Launch the Streamlit application.
3.  Upload the quotation PDFs you want to compare.
4.  Select the appropriate PDF processing mode offered by the
    application.
5.  Run extraction and review supplier details, line items, totals, and
    validation notes.
6.  Compare quotation prices and matched items.
7.  Enter supplier quality ratings supported by your research or
    experience.
8.  Adjust the price-versus-quality weighting if needed.
9.  Review the ranked comparison and recommendation explanations.

## Running Tests

Run the full test suite from the project root:

``` bash
python -m pytest
```

Run individual test modules:

``` bash
python -m pytest tests/test_scoring.py
python -m pytest tests/test_pipeline.py
python -m pytest tests/test_recommender.py
```

## Privacy-Focused Workflow

QuoteCompare is designed around local quotation processing:

-   Ollama performs model inference through the local service.
-   Quotation PDFs can be processed on the user's own computer.
-   Supplier quality ratings can be entered manually.
-   The manual rating workflow avoids the need for a Google Places API
    key.
-   Users can review extracted data and validation notes before making
    procurement decisions.

Always check important extracted figures against the original quotation,
particularly taxes, discounts, currency symbols, and grand totals.

## Development Practices

-   Keep extraction, normalization, matching, validation, scoring, and
    recommendations in separate modules.
-   Add tests when introducing new behavior.
-   Keep local secrets and private quotation files out of version
    control.
-   Run the test suite before committing changes.

## License

Add a `LICENSE` file to specify the terms under which QuoteCompare may
be used and distributed.
