"""
embedder.py — BERT-based article relevance ranker.

Uses HuggingFace's sentence-similarity pipeline to score how relevant
each news article is to the user's query. Only the most relevant
articles are sent to the LLM for summarization.

Model: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional embeddings)
"""

import logging
import os
from typing import List, Dict, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

# Track last error for UI display
last_error: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rank_articles_by_relevance(
    query: str,
    articles: List[Dict[str, str]],
    top_k: int = 5,
    min_score: float = 0.1,
) -> List[Tuple[Dict[str, str], float]]:
    """
    Rank articles by semantic relevance to the user's query using BERT.

    Uses HuggingFace sentence-similarity pipeline:
      - source_sentence = user query
      - sentences = article headlines + content

    Parameters
    ----------
    query : str
        The user's search query (e.g. "NIFTY today").
    articles : list[dict]
        Articles with headline, content, etc.
    top_k : int
        Number of top relevant articles to return.
    min_score : float
        Minimum similarity score (0-1) to include.

    Returns
    -------
    list[tuple[dict, float]]
        List of (article_dict, similarity_score) sorted by relevance (highest first).
    """
    global last_error
    last_error = ""

    if not articles:
        return []

    if not query or not query.strip():
        # No query to compare — return all articles with score 1.0
        return [(a, 1.0) for a in articles]

    api_key = os.getenv("HF_API_KEY", "")
    if not api_key:
        last_error = "HF_API_KEY is not set in .env"
        logger.error(last_error)
        return []

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # ── Step 1: Build sentence list from articles ──
    print("\n" + "=" * 80)
    print("🧠 BERT EMBEDDING PIPELINE — START")
    print("=" * 80)
    print(f"\n📝 SOURCE SENTENCE (User Query):")
    print(f"   \"{query.strip()}\"")
    print(f"\n📰 BUILDING SENTENCE LIST FROM {len(articles)} ARTICLES:")
    print("-" * 60)

    sentences = []
    for i, a in enumerate(articles, 1):
        headline = a.get("headline", "")
        content = a.get("content", "")
        # Combine headline + first ~300 chars of content for scoring
        text = f"{headline}. {content[:300]}"
        sentences.append(text)
        print(f"   Article {i}: \"{headline[:70]}{'...' if len(headline) > 70 else ''}\"")
        print(f"              Text length: {len(text)} chars")

    print(f"\n✅ Built {len(sentences)} sentences for BERT encoding")

    # ── Step 2: Send to HuggingFace API ──
    payload = {
        "inputs": {
            "source_sentence": query.strip(),
            "sentences": sentences,
        }
    }

    print(f"\n📡 SENDING TO HUGGINGFACE API:")
    print(f"   Model: sentence-transformers/all-MiniLM-L6-v2")
    print(f"   URL: {API_URL}")
    print(f"   Payload: source_sentence + {len(sentences)} sentences")

    try:
        logger.info(
            "BERT ranking %d articles against query: '%s'",
            len(sentences), query[:80],
        )

        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        # ── Step 3: Parse similarity scores ──
        scores = response.json()  # List of floats, e.g. [0.85, 0.42, 0.78]

        print(f"\n✅ BERT API RESPONSE RECEIVED")
        print(f"\n📊 COSINE SIMILARITY SCORES:")
        print("-" * 60)
        for i, (art, score) in enumerate(zip(articles, scores), 1):
            bar_len = int(score * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            headline = art.get("headline", "N/A")[:50]
            print(f"   Article {i}: [{bar}] {score:.4f}  \"{headline}...\"")

        logger.info("BERT similarity scores: %s", [f"{s:.3f}" for s in scores])

        # ── Step 4: Pair, sort, and filter ──
        paired = list(zip(articles, scores))
        paired.sort(key=lambda x: x[1], reverse=True)

        print(f"\n🔍 SIMILARITY FILTERING (threshold ≥ {min_score}):")
        print("-" * 60)

        results = []
        for rank, (art, score) in enumerate(paired, 1):
            headline = art.get("headline", "N/A")[:60]
            if score >= min_score and len(results) < top_k:
                results.append((art, score))
                print(f"   #{rank} ✅ KEPT   — Score: {score:.4f} — \"{headline}\"")
            elif score < min_score:
                print(f"   #{rank} ❌ REMOVED — Score: {score:.4f} — \"{headline}\" (below threshold)")
            else:
                print(f"   #{rank} ⏭️ SKIPPED — Score: {score:.4f} — \"{headline}\" (top_k={top_k} reached)")

        print(f"\n📋 FINAL RESULT:")
        print(f"   Total articles:    {len(articles)}")
        print(f"   Articles passed:   {len(results)}")
        print(f"   Articles removed:  {len(articles) - len(results)}")
        if results:
            print(f"   Best score:        {results[0][1]:.4f}")
            print(f"   Worst kept score:  {results[-1][1]:.4f}")
        print(f"   Threshold:         {min_score}")
        print(f"   Top-K limit:       {top_k}")

        print("\n" + "=" * 80)
        print("🧠 BERT EMBEDDING PIPELINE — COMPLETE")
        print("=" * 80 + "\n")

        logger.info(
            "BERT ranker: %d/%d articles passed (top score=%.3f, cutoff=%.2f)",
            len(results), len(articles),
            results[0][1] if results else 0, min_score,
        )
        return results

    except requests.exceptions.HTTPError as exc:
        last_error = f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
        logger.error("HuggingFace API HTTP error: %s — %s", exc, exc.response.text[:500])
        print(f"\n❌ BERT API ERROR: {last_error}")
        return []
    except requests.exceptions.RequestException as exc:
        last_error = f"Request failed: {exc}"
        logger.error("HuggingFace API request failed: %s", exc)
        print(f"\n❌ BERT REQUEST FAILED: {last_error}")
        return []
    except Exception as exc:
        last_error = f"Error: {exc}"
        logger.error("BERT ranking failed: %s", exc, exc_info=True)
        print(f"\n❌ BERT RANKING FAILED: {last_error}")
        return []
