"""
vector_store.py — In-memory vector store for semantic article search.

Stores article embeddings and supports cosine-similarity search
to find the most relevant articles for a given query.
"""

import logging
from typing import List, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Simple in-memory vector store using numpy.

    Stores articles alongside their BERT embeddings and
    supports cosine similarity search.
    """

    def __init__(self):
        self.articles: List[Dict[str, str]] = []
        self.embeddings: Optional[np.ndarray] = None  # shape: (n_articles, dim)
        self._is_indexed = False

    def clear(self):
        """Clear all stored articles and embeddings."""
        self.articles = []
        self.embeddings = None
        self._is_indexed = False

    def add_articles(
        self,
        articles: List[Dict[str, str]],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add articles with their precomputed embeddings.

        Parameters
        ----------
        articles : list[dict]
            Article dicts with headline, content, etc.
        embeddings : np.ndarray
            Shape (len(articles), dim).
        """
        if len(articles) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(articles)} articles but {len(embeddings)} embeddings"
            )

        self.articles = articles
        self.embeddings = embeddings.astype(np.float32)

        # Normalize embeddings for fast cosine similarity (dot product)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # avoid division by zero
        self.embeddings = self.embeddings / norms

        self._is_indexed = True
        logger.info(
            "Vector store indexed %d articles (embedding dim: %d)",
            len(self.articles), self.embeddings.shape[1],
        )

    @property
    def is_indexed(self) -> bool:
        return self._is_indexed

    @property
    def count(self) -> int:
        return len(self.articles)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[Dict[str, str], float]]:
        """
        Find the most similar articles to the query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray
            1D embedding vector for the query.
        top_k : int
            Number of top results to return.
        threshold : float
            Minimum similarity score (0-1) to include.

        Returns
        -------
        list[tuple[dict, float]]
            List of (article_dict, similarity_score) sorted by relevance.
        """
        if not self._is_indexed or self.embeddings is None:
            logger.warning("Vector store is empty — no search possible.")
            return []

        # Normalize query
        q = query_embedding.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Cosine similarity = dot product (both normalized)
        similarities = self.embeddings @ q

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results: List[Tuple[Dict[str, str], float]] = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                results.append((self.articles[idx], score))

        logger.info(
            "Vector search: top-%d results, best score=%.4f, worst score=%.4f",
            len(results),
            results[0][1] if results else 0,
            results[-1][1] if results else 0,
        )
        return results

    def get_all_articles(self) -> List[Dict[str, str]]:
        """Return all stored articles."""
        return self.articles
