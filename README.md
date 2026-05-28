# Universal AI Financial Scraper 🕵️‍♂️📈

An autonomous, AI-powered web scraper designed to extract highly specific financial data points from complex search queries. It utilizes Playwright (with Stealth Mode) to bypass bot protections, BM25 text filtering to compress massive Annual Reports into relevant chunks, and a custom LLM Router to extract structured JSON data.

## Features
- **Stealth Scraping:** Uses Playwright Stealth to scrape heavily protected financial sites without getting blocked.
- **AI-Powered Queries:** Understands 1000-word complex queries and distills them into targeted search parameters.
- **BM25 Relevance Filtering:** Can download a 300-page PDF, rank the paragraphs based on your query, and only pass the top `X` chunks to the LLM to drastically reduce token costs and improve speed.
- **Dynamic Domain Filtering:** Search the entire internet, or restrict searches exclusively to a custom list of trusted financial domains (e.g. `finance.yahoo.com`).

## Installation

1. Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
playwright install chromium
```

## Usage

### 1. Single Query Extraction (`scraper.py`)
Run a standard query. The scraper will find the best sources, download them concurrently, and extract the answer.
```bash
python scraper.py "What were Apple's Q3 2023 earnings, and how did their services division perform?"
```

**CLI Arguments:**
- `--sources`: Number of websites to scrape concurrently (default: 2)
- `--filter`: Enable domain filtering (searches only `ALLOWED_DOMAINS` by default)
- `--domains`: Comma-separated list of custom domains (e.g., `--domains "finance.yahoo.com,bloomberg.com"`)

Example:
```bash
python scraper.py "Apple Q3 2023 earnings" --filter --sources 5
```

## Architecture

- `scraper.py`: Core logic (Search -> Scrape -> HTML-to-Markdown -> BM25 Filter -> LLM Extraction).
- `llm_router.py`: Handles model execution and prompt routing.
- `gemini_manager.py`: API key rotation and management to handle heavy workloads.
