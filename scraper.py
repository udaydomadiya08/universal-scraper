import os
import sys
import argparse
import requests
from typing import List, Optional
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth
from bs4 import BeautifulSoup
# pyrefly: ignore [missing-import]
import markdownify

# Load allowed domains from text file
ALLOWED_DOMAINS = []
try:
    with open('allowed_domains.txt', 'r') as f:
        ALLOWED_DOMAINS = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    pass

def search_for_urls(query: str, num_results: int = 3, filter_domains: bool = False, allowed_domains: List[str] = None) -> List[str]:
    """Search Yahoo Search to find urls, optionally filtering by allowed domains."""
    print("[*] Finding relevant sources...")
    urls = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    import urllib.parse
    
    if filter_domains and allowed_domains:
        # Check ALL allowed domains using Yahoo Search
        for domain in allowed_domains:
            search_query = f"{query} site:{domain}"
            try:
                url = f"https://search.yahoo.com/search?p={urllib.parse.quote(search_query)}"
                response = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")
                
                for a in soup.select('div.compTitle a'):
                    href = a.get('href')
                    if not href: continue
                    
                    real_url = href
                    if "RU=" in href:
                        try:
                            real_url = urllib.parse.unquote(href.split('RU=')[1].split('/RK=')[0])
                        except:
                            pass
                    
                    if domain in real_url and real_url not in urls:
                        urls.append(real_url)
                        if len(urls) >= num_results:
                            return urls
            except Exception:
                continue
    else:
        # No filtering, just search the entire web
        try:
            url = f"https://search.yahoo.com/search?p={urllib.parse.quote(query)}"
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            
            for a in soup.select('div.compTitle a'):
                href = a.get('href')
                if not href: continue
                
                real_url = href
                if "RU=" in href:
                    try:
                        real_url = urllib.parse.unquote(href.split('RU=')[1].split('/RK=')[0])
                    except:
                        pass
                
                if real_url not in urls:
                    urls.append(real_url)
                    if len(urls) >= num_results:
                        return urls
        except Exception as e:
            print(f"[!] Global search error: {e}")

    return urls

def scrape_url(url: str) -> str:
    """Scrape a URL using Playwright with stealth mode."""
    print(f"[*] Scraping {url}...")
    html_content = ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            # Apply stealth to avoid basic bot detection
            Stealth().apply_stealth_sync(page)
            
            # Navigate to the page, wait for it to load
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            html_content = page.content()
            browser.close()
    except Exception as e:
        print(f"[!] Error scraping {url}: {e}")
        
    return html_content

def html_to_markdown(html: str) -> str:
    """Convert HTML to clean Markdown for the LLM context."""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script, style, and navigation elements that add noise
    for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        element.extract()
        
    clean_html = str(soup)
    md_text = markdownify.markdownify(clean_html, heading_style="ATX").strip()
    return md_text

def extract_search_keywords(complex_query: str) -> str:
    """Use LLM to extract a short, concise search engine query from a complex user prompt."""
    if len(complex_query.split()) < 8:
        return complex_query # Already short enough
        
    print("[*] Simplifying complex query for search engines...")
    try:
        from llm_router import LLMRouter
        router = LLMRouter()
        prompt = f"Extract a very short (2-6 words) web search query to find the financial data requested here:\n\n{complex_query}\n\nJust output the search terms, nothing else. Do not use quotes."
        result = router.get_response(prompt, category="text")
        if result.get("status") == "success":
            return result["content"].strip().replace('"', '')
    except Exception as e:
        print(f"[!] Keyword extraction failed: {e}")
        
    return " ".join(complex_query.split()[:8]) # Fallback to first 8 words

def extract_answer(query: str, search_query: str, contexts: List[str]) -> str:
    """Use LLMRouter to extract the final answer from the collected contexts using BM25 filtering."""
    print("[*] Analyzing data with AI...")
    
    from rank_bm25 import BM25Okapi
    import re
    
    # Combine all contexts
    full_text = "\n\n".join(contexts)
    
    # Split into manageable chunks (e.g. by paragraphs)
    raw_chunks = [c.strip() for c in re.split(r'\n\s*\n', full_text) if len(c.strip()) > 30]
    
    # If the text is small enough, just use it all
    if len(raw_chunks) <= 500:
        filtered_chunks = raw_chunks
    else:
        print(f"[*] BM25: Filtering {len(raw_chunks)} paragraphs down to the most relevant 500...")
        # Tokenize chunks for BM25
        tokenized_chunks = [chunk.lower().split() for chunk in raw_chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        
        # Tokenize the full complex query for better keyword matching in BM25
        # We want BM25 to find paragraphs containing 'revenue', 'EPS', etc. from the user's prompt
        tokenized_query = query.lower().split()
        
        # Get top 500 chunks
        filtered_chunks = bm25.get_top_n(tokenized_query, raw_chunks, n=500)
        
    combined_context = "\n\n...[SNIP]...\n\n".join(filtered_chunks)
    
    prompt = f"""
You are an expert financial data extractor and analyzer. 
I have scraped the following highly relevant snippets from financial websites.
Use ONLY the information provided in the context below to answer my query in detail. 
If the information is not present in the context, explicitly state that you couldn't find it.

Query: {query}

Context Data:
{combined_context}
    """
    
    try:
        from llm_router import LLMRouter
        router = LLMRouter()
        result = router.get_response(prompt, category="extraction")
        
        if result.get("status") == "success":
            return result["content"]
        else:
            return f"Error during AI extraction: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        return f"Error integrating LLM Router: {e}"

def main():
    parser = argparse.ArgumentParser(description="Universal AI Financial Scraper (CLI)")
    parser.add_argument("query", type=str, help="Natural language query to search for")
    parser.add_argument("--sources", type=int, default=2, help="Number of sources to scrape")
    parser.add_argument("--filter", action="store_true", help="Enable domain filtering")
    parser.add_argument("--domains", type=str, default="", help="Comma separated list of allowed domains if filter is True")
    args = parser.parse_args()
    
    query = args.query
    print(f"\n[+] Complex Query Received (length: {len(query.split())} words)")
    
    search_query = extract_search_keywords(query)
    print(f"[*] Generated Search Keywords: '{search_query}'")
    
    # Process domains
    allowed_domains = [d.strip() for d in args.domains.split(",")] if args.domains else []
    if args.filter and not allowed_domains:
        # Fallback to the default list if filter is true but no domains provided
        allowed_domains = ALLOWED_DOMAINS
        print("[*] Filter flag is true but no domains provided. Using default financial domains.")
    
    # 1. Find URLs
    urls = search_for_urls(search_query, num_results=args.sources, filter_domains=args.filter, allowed_domains=allowed_domains)
    if not urls:
        print("[-] No relevant sources found on allowed domains.")
        sys.exit(0)
        
    print(f"[+] Found {len(urls)} sources.")
    
    # 2. Scrape and convert URLs concurrently
    contexts = []
    
    def process_url(url):
        html = scrape_url(url)
        if html:
            md = html_to_markdown(html)
            return f"Source URL: {url}\n\nContent:\n{md}"
        return None
        
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    print(f"[*] Starting concurrent download of {len(urls)} sources...")
    with ThreadPoolExecutor(max_workers=args.sources) as executor:
        future_to_url = {executor.submit(process_url, url): url for url in urls}
        for future in as_completed(future_to_url):
            result = future.result()
            if result:
                contexts.append(result)
            
    if not contexts:
        print("[-] Failed to scrape any content.")
        sys.exit(1)
        
    # 3. Extract final answer
    print("\n[+] Context gathered. Generating response...\n")
    answer = extract_answer(query, search_query, contexts)
    
    print("-" * 50)
    print(answer)
    print("-" * 50)

if __name__ == "__main__":
    main()
