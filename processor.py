"""
processor.py — Text cleaning and merging for news articles.

Takes article dicts from the NEWSDATA.io fetcher and produces a single
clean string suitable for LLM consumption.
"""

import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace into single spaces and strip."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _remove_boilerplate(text: str) -> str:
    """Remove common boilerplate phrases found on news sites."""
    patterns = [
        r"(?i)subscribe\s+to\s+our\s+newsletter.*",
        r"(?i)sign\s+up\s+for\s+.*newsletter.*",
        r"(?i)click\s+here\s+to\s+read\s+more.*",
        r"(?i)advertisement.*",
        r"(?i)follow\s+us\s+on\s+(twitter|facebook|instagram|linkedin).*",
        r"(?i)share\s+this\s+(article|story).*",
        r"(?i)read\s+more:.*",
        r"(?i)also\s+read:.*",
        r"(?i)recommended\s+for\s+you.*",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    return text.strip()


def _deduplicate(articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove duplicate articles based on headline similarity."""
    seen: set = set()
    unique: List[Dict[str, str]] = []
    for article in articles:
        key = re.sub(r"\s+", " ", article.get("headline", "").lower().strip())
        if key and key not in seen:
            seen.add(key)
            unique.append(article)
    return unique


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_and_merge_articles(article_list: List[Dict[str, str]]) -> str:
    """
    Clean, deduplicate, and merge articles into a single LLM-ready string.

    Parameters
    ----------
    article_list : list[dict]
        Each dict should have keys: headline, content, published_time,
        and optionally source, link.

    Returns
    -------
    str
        A single merged string where each article is separated by a
        clear delimiter, ready for LLM consumption.
    """
    if not article_list:
        logger.warning("No articles to process.")
        return ""

    print("\n" + "=" * 80)
    print("🧹 TEXT PROCESSOR PIPELINE — START")
    print("=" * 80)
    print(f"\n📥 Input: {len(article_list)} articles")

    logger.info("Processing %d article(s).", len(article_list))

    # Deduplicate
    articles = _deduplicate(article_list)
    removed = len(article_list) - len(articles)
    print(f"🔄 Deduplication: {len(articles)} unique articles (removed {removed} duplicates)")
    logger.info("%d unique article(s) after deduplication.", len(articles))

    merged_parts: list[str] = []
    print(f"\n🧹 CLEANING ARTICLES:")
    print("-" * 60)

    for idx, article in enumerate(articles, start=1):
        headline = _normalize_whitespace(article.get("headline", "Untitled"))
        raw_content = article.get("content", "")
        content = _normalize_whitespace(raw_content)
        content = _remove_boilerplate(content)
        published = article.get("published_time", "").strip()
        source = article.get("source", "").strip()

        chars_removed = len(raw_content) - len(content)
        print(f"   Article {idx}: \"{headline[:60]}...\"")
        print(f"              Raw: {len(raw_content)} chars → Cleaned: {len(content)} chars (removed {chars_removed} chars)")

        if not content:
            print(f"              ⚠️ SKIPPED (empty after cleaning)")
            continue

        section = f"--- Article {idx} ---\n"
        section += f"Headline: {headline}\n"
        if source:
            section += f"Source: {source}\n"
        if published:
            section += f"Published: {published}\n"
        section += f"Content: {content}\n"
        merged_parts.append(section)

    merged = "\n".join(merged_parts)

    print(f"\n📋 MERGED OUTPUT:")
    print(f"   Total sections: {len(merged_parts)}")
    print(f"   Total characters: {len(merged):,}")
    print(f"   Preview (first 200 chars):")
    print(f"   {merged[:200]}...")
    print("\n" + "=" * 80)
    print("🧹 TEXT PROCESSOR PIPELINE — COMPLETE")
    print("=" * 80 + "\n")

    logger.info("Merged text length: %d characters.", len(merged))
    return merged
