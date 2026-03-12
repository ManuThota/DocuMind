"""
Reranker Service
BM25 hybrid reranking to improve retrieval relevance
Combines dense vector scores with sparse BM25 lexical scores
"""

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class BM25Reranker:
    """
    BM25-based reranker that combines vector similarity scores with
    lexical BM25 scores for improved retrieval accuracy.

    BM25 (Best Match 25) is a probabilistic ranking function that considers:
    - Term frequency in document (with saturation)
    - Inverse document frequency (rarity of term)
    - Document length normalization
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, alpha: float = 0.5):
        """
        Args:
            k1: Term frequency saturation (typically 1.2-2.0)
            b: Document length normalization (0 = no normalization, 1 = full)
            alpha: Weight for vector score vs BM25 score (0.5 = equal weight)
        """
        self.k1 = k1
        self.b = b
        self.alpha = alpha  # hybrid score = alpha * vector_score + (1-alpha) * bm25_score

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using hybrid BM25 + vector similarity scoring.

        Args:
            query: User's question
            candidates: List of retrieved chunks with text and score
            top_k: Number of top results to return

        Returns:
            Reranked list with updated scores, limited to top_k
        """
        if not candidates:
            return []

        if len(candidates) == 1:
            return candidates[:top_k]

        query_tokens = self._tokenize(query)

        if not query_tokens:
            logger.debug("No query tokens after processing, using original scores")
            return candidates[:top_k]

        # Compute BM25 scores
        corpus = [c["text"] for c in candidates]
        bm25_scores = self._compute_bm25_scores(query_tokens, corpus)

        # Normalize BM25 scores to [0, 1]
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        normalized_bm25 = [s / max_bm25 for s in bm25_scores]

        # Normalize vector scores to [0, 1] (they should already be, but ensure it)
        vector_scores = [c.get("score", 0.0) for c in candidates]
        max_vec = max(vector_scores) if max(vector_scores) > 0 else 1.0
        normalized_vec = [s / max_vec for s in vector_scores]

        # Combine scores
        results = []
        for i, candidate in enumerate(candidates):
            hybrid_score = (
                self.alpha * normalized_vec[i] +
                (1 - self.alpha) * normalized_bm25[i]
            )
            result = candidate.copy()
            result["vector_score"] = candidate.get("score", 0.0)
            result["bm25_score"] = bm25_scores[i]
            result["score"] = hybrid_score  # Replace with hybrid score
            results.append(result)

        # Sort by hybrid score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        logger.debug(
            "Reranked %d candidates -> top %d selected",
            len(candidates), min(top_k, len(results))
        )

        # Log top result details for debugging
        if results:
            top = results[0]
            logger.debug(
                "Top result: vector=%.3f, bm25=%.3f, hybrid=%.3f | %s",
                top.get("vector_score", 0),
                top.get("bm25_score", 0),
                top["score"],
                top["text"][:80]
            )

        return results[:top_k]

    def _compute_bm25_scores(
        self,
        query_tokens: List[str],
        corpus: List[str]
    ) -> List[float]:
        """
        Compute BM25 scores for query against each document in corpus.
        """
        # Tokenize corpus
        tokenized_corpus = [self._tokenize(doc) for doc in corpus]

        # Compute document lengths
        doc_lengths = [len(doc) for doc in tokenized_corpus]
        avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)

        # Build document frequency index
        df = Counter()
        for doc_tokens in tokenized_corpus:
            # Count unique terms per document
            for term in set(doc_tokens):
                df[term] += 1

        N = len(corpus)
        scores = []

        for doc_idx, doc_tokens in enumerate(tokenized_corpus):
            score = 0.0
            tf_counter = Counter(doc_tokens)
            doc_length = doc_lengths[doc_idx]

            for term in query_tokens:
                if term not in tf_counter:
                    continue

                tf = tf_counter[term]
                df_term = df.get(term, 0)

                if df_term == 0:
                    continue

                # IDF with smoothing
                idf = math.log((N - df_term + 0.5) / (df_term + 0.5) + 1)

                # TF normalization with BM25 formula
                tf_normalized = (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_length / max(avg_doc_length, 1))
                )

                score += idf * tf_normalized

            scores.append(score)

        return scores

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25: lowercase, split, remove stopwords.
        """
        # Lowercase and extract words
        text = text.lower()
        tokens = re.findall(r"\b[a-z][a-z0-9]*\b", text)

        # Remove common stopwords
        stopwords = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "shall",
            "that", "this", "these", "those", "it", "its", "as", "not",
            "what", "which", "who", "how", "when", "where", "why",
        }

        filtered = [t for t in tokens if t not in stopwords and len(t) > 1]
        return filtered


class CrossEncoderReranker:
    """
    Optional cross-encoder reranker using a local sentence-transformers model.
    Falls back to BM25 if model is not available.
    """

    def __init__(self):
        self._model = None
        self._available = False
        self._try_load()

    def _try_load(self):
        """Try to load cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._available = True
            logger.info("Cross-encoder reranker loaded successfully")
        except Exception as e:
            logger.info(
                "Cross-encoder not available (%s), will use BM25 reranking", str(e)
            )

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Rerank using cross-encoder if available, else BM25."""
        if not self._available or not self._model:
            bm25 = BM25Reranker()
            return bm25.rerank(query, candidates, top_k)

        try:
            pairs = [(query, c["text"]) for c in candidates]
            ce_scores = self._model.predict(pairs)

            results = []
            for i, candidate in enumerate(candidates):
                result = candidate.copy()
                result["cross_encoder_score"] = float(ce_scores[i])
                result["score"] = float(ce_scores[i])
                results.append(result)

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.warning("Cross-encoder reranking failed: %s, falling back to BM25", str(e))
            bm25 = BM25Reranker()
            return bm25.rerank(query, candidates, top_k)
