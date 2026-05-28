# Universal AI Scraper 🕵️‍♂️🌍

An autonomous, AI-powered universal web scraper designed to extract highly specific data points from complex search queries across any domain. It utilizes Playwright (with Stealth Mode) to bypass bot protections, BM25 text filtering to compress massive documents into relevant chunks, and a custom LLM Router to extract structured JSON data.

## Features
- **Stealth Scraping:** Uses Playwright Stealth to scrape heavily protected sites without getting blocked.
- **AI-Powered Queries:** Understands 1000-word complex queries and distills them into targeted search parameters.
- **BM25 Relevance Filtering:** Can download a 300-page PDF, rank the paragraphs based on your query, and only pass the top `X` chunks to the LLM to drastically reduce token costs and improve speed.
- **Dynamic Domain Filtering:** Search the entire internet, or restrict searches exclusively to a custom list of trusted domains.

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

## Configuration (API Keys)

Before running the scraper, you must set up your API keys. The project uses Groq for fast routing and Gemini for heavy text processing.

1. **Groq API Key**: 
   - Rename `config.example.json` to `config.json`.
   - Replace the placeholder with your actual Groq API key.

2. **Gemini API Keys**: 
   - Rename `gemini_config_10keys.example.json` to `gemini_config_10keys.json`.
   - Replace the placeholders with your Gemini API keys (the system supports multiple keys for auto-rotation to bypass rate limits).

## Usage

### Configuring Allowed Domains
If you want to restrict the AI to only search specific, trusted websites, you can enable the `--filter` flag. By default, this will force the scraper to only pull data from the websites listed in the `allowed_domains.txt` file.

To add your own trusted domains:
1. Open the `allowed_domains.txt` file.
2. Add one domain per line (e.g., `reddit.com` or `wsj.com`).
3. Save the file.

When you run `scraper.py` with `--filter`, it will automatically read from this list!

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
