import re
import httpx
from bs4 import BeautifulSoup
from typing import Dict, Any
from app.config import settings
from app.utils.logger import logger


class URLReaderTool:
    """Extracts readable content from URLs using Firecrawl or fallback HTML parsing."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.firecrawl_api_key = settings.FIRECRAWL_API_KEY

    def read_url(self, url: str) -> Dict[str, Any]:
        """Fetches and extracts clean main text content from a given URL."""
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"https://{url}"

        logger.info(f"Extracting web page content from: {url}")

        # Strategy 1: Use Firecrawl if configured
        if self.firecrawl_api_key:
            try:
                from firecrawl import FirecrawlApp
                app = FirecrawlApp(api_key=self.firecrawl_api_key)
                scrape_result = app.scrape_url(url, params={'formats': ['markdown']})
                markdown_content = scrape_result.get('markdown', '')
                
                if markdown_content:
                    logger.info(f"Successfully scraped {url} via Firecrawl.")
                    return {
                        "url": url,
                        "title": scrape_result.get('metadata', {}).get('title', url),
                        "content": markdown_content[:8000],  # Token limit guard
                        "method": "firecrawl"
                    }
            except Exception as e:
                logger.warning(f"Firecrawl scrape failed for {url}: {str(e)}. Falling back to HTTP client.")

        # Strategy 2: Reliable HTTP + BeautifulSoup Fallback
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Strip non-content script/style elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()

            title = soup.title.string.strip() if soup.title and soup.title.string else url
            
            # Extract main text content
            text = soup.get_text(separator="\n")
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = "\n".join(chunk for chunk in chunks if chunk)

            logger.info(f"Successfully scraped {url} via HTTP/BeautifulSoup.")
            return {
                "url": url,
                "title": title,
                "content": clean_text[:8000],
                "method": "bs4_fallback"
            }

        except Exception as e:
            logger.error(f"Failed to fetch content from URL '{url}': {str(e)}")
            return {
                "url": url,
                "title": "Extraction Error",
                "content": f"Unable to fetch web page content: {str(e)}",
                "method": "error"
            }


def extract_url_from_text(text: str) -> str:
    """Helper to parse raw HTTP/HTTPS URL strings out of user prompts."""
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    match = re.search(url_pattern, text)
    if match:
        return match.group(0)
    return ""