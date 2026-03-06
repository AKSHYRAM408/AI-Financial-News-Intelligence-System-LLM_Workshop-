"""
news_fetcher.py — NEWSDATA.io API client for financial news.

Replaces the Selenium-based scraper with a lightweight API approach.
Fetches latest market news using keyword search via the NEWSDATA.io API.
"""

import logging
import os
from typing import List, Dict, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NEWSDATA_BASE_URL = "https://newsdata.io/api/1/latest"

# Default keywords for broad market coverage
DEFAULT_KEYWORDS = [
    "stock market", "NIFTY", "SENSEX", "NASDAQ", "S&P 500",
    "BSE", "NSE", "share price", "market rally", "market crash",
    "IPO", "mutual fund", "RBI", "Federal Reserve", "inflation",
    "earnings", "quarterly results",
]

# Preferred financial news domains
PREFERRED_DOMAINS = [
    "moneycontrol.com", "economictimes.com", "livemint.com",
    "thehindu.com", "ndtv.com", "reuters.com", "bloomberg.com",
    "cnbc.com", "marketwatch.com",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_query(user_query: str) -> str:
    """
    Build an effective search query string.

    If the user provides a specific query, use it directly.
    Otherwise, combine a few default keywords for broad coverage.
    """
    if user_query and user_query.strip():
        return user_query.strip()
    # Fallback to broad market query
    return "stock market OR NIFTY OR NASDAQ OR BSE OR SENSEX"


def _parse_articles(raw_results: List[Dict]) -> List[Dict[str, str]]:
    """
    Transform raw NEWSDATA.io response articles into a clean list of dicts.

    Each dict has: headline, content, published_time, source, link
    """
    articles: List[Dict[str, str]] = []

    for item in raw_results:
        headline = (item.get("title") or "").strip()
        # Free plan: 'content' returns "ONLY AVAILABLE IN PAID PLANS"
        # so we use 'description' as primary content source
        raw_content = (item.get("content") or "").strip()
        description = (item.get("description") or "").strip()
        # Use content only if it's real (not the paid-plan placeholder)
        if raw_content and "ONLY AVAILABLE" not in raw_content:
            content = raw_content
        else:
            content = description
        published = (item.get("pubDate") or "").strip()
        source = (item.get("source_id") or item.get("source_name") or "").strip()
        link = (item.get("link") or "").strip()

        if not headline and not content:
            continue

        articles.append({
            "headline": headline,
            "content": content,
            "published_time": published,
            "source": source,
            "link": link,
        })

    return articles


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_market_news(
    user_query: str = "",
    language: str = "en",
    max_results: int = 10,
    category: str = "business",
) -> List[Dict[str, str]]:
    """
    Fetch latest financial news from NEWSDATA.io API.

    Parameters
    ----------
    user_query : str
        The investor's query or topic of interest.
    language : str
        Language code for news results (default: English).
    max_results : int
        Maximum number of articles to fetch (API may return fewer).
    category : str
        News category filter (default: business).

    Returns
    -------
    list[dict]
        Each dict has keys: headline, content, published_time, source, link.
        Returns an empty list on failure.
    """
    api_key = os.getenv("NEWSDATA_API_KEY", "")
    if not api_key or api_key == "your-newsdata-api-key-here":
        logger.error("NEWSDATA_API_KEY is not set in .env")
        return []

    query = _build_query(user_query)
    logger.info("Fetching news from NEWSDATA.io — query: '%s'", query)

    print("\n" + "=" * 80)
    print("📰 NEWS FETCHER PIPELINE — START")
    print("=" * 80)
    print(f"\n🔎 Query: \"{query}\"")
    print(f"   Language: {language}")
    print(f"   Max results: {max_results}")
    print(f"   API: NEWSDATA.io")

    # NOTE: NEWSDATA.io free plan does NOT allow combining 'q' with 'category'.
    # Using 'q' alone for keyword search.
    params = {
        "apikey": api_key,
        "q": query,
        "language": language,
        "size": min(max_results, 10),  # Free plan max is 10 per request
    }

    try:
        print(f"\n📡 Sending request to NEWSDATA.io API...")
        response = requests.get(NEWSDATA_BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        status = data.get("status")
        if status != "success":
            error_msg = data.get("results", {}).get("message", "Unknown error")
            logger.error("NEWSDATA.io returned status='%s': %s", status, error_msg)
            print(f"❌ API Error: {error_msg}")
            return []

        raw_results = data.get("results", [])
        if not raw_results:
            logger.warning("NEWSDATA.io returned 0 results for query: '%s'", query)
            print(f"⚠️ No results found for query: '{query}'")
            return []

        print(f"\n✅ API returned {len(raw_results)} raw results")

        articles = _parse_articles(raw_results)

        print(f"\n📰 PARSED ARTICLES ({len(articles)} valid):")
        print("-" * 60)
        for i, art in enumerate(articles, 1):
            print(f"   {i}. [{art.get('source', 'Unknown')}] {art.get('headline', 'N/A')[:70]}")
            print(f"      Content: {len(art.get('content', ''))} chars | Published: {art.get('published_time', 'N/A')}")

        print("\n" + "=" * 80)
        print("📰 NEWS FETCHER PIPELINE — COMPLETE")
        print("=" * 80 + "\n")

        logger.info("Fetched %d article(s) from NEWSDATA.io", len(articles))
        return articles

    except requests.exceptions.Timeout:
        logger.error("NEWSDATA.io request timed out.")
        return []
    except requests.exceptions.HTTPError as exc:
        logger.error("NEWSDATA.io HTTP error: %s", exc)
        return []
    except requests.exceptions.RequestException as exc:
        logger.error("NEWSDATA.io request failed: %s", exc)
        return []
    except Exception as exc:
        logger.error("Unexpected error fetching news: %s", exc, exc_info=True)
        return []
