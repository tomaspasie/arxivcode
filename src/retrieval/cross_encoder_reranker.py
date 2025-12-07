"""
Cross-Encoder Re-ranking Pipeline for ArXivCode
Re-ranks retrieval results using cross-encoder models for improved relevance scoring.
"""

import logging
from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder based re-ranking for improved relevance scoring."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: Pre-trained cross-encoder model name
            max_length: Maximum sequence length for model input
            device: Device to run model on ('cpu', 'cuda', 'mps', or None for auto-detect)
        """
        self.model_name = model_name
        self.max_length = max_length

        # Auto-detect best device
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device

        logger.info(f"Loading cross-encoder model: {model_name}")
        try:
            self.model = CrossEncoder(
                model_name,
                max_length=max_length,
                device=device
            )
            logger.info("✅ Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Re-rank candidate results using cross-encoder.

        Args:
            query: The search query
            candidates: List of candidate results from initial retrieval
            top_k: Number of top results to return (None = all)

        Returns:
            Re-ranked results with updated scores
        """
        if not candidates:
            return []

        # Prepare query-document pairs for cross-encoder
        query_doc_pairs = []
        for candidate in candidates:
            # Create document text from metadata
            doc_text = self._create_document_text(candidate['metadata'])
            query_doc_pairs.append([query, doc_text])

        # Score all pairs with cross-encoder
        logger.info(f"Scoring {len(query_doc_pairs)} query-document pairs")
        scores = self.model.predict(query_doc_pairs)

        # Update candidate scores and sort
        reranked_candidates = []
        for candidate, ce_score in zip(candidates, scores):
            updated_candidate = candidate.copy()
            updated_candidate['cross_encoder_score'] = float(ce_score)
            # Keep original score for reference
            updated_candidate['original_score'] = candidate['score']
            # Use cross-encoder score as primary score
            updated_candidate['score'] = float(ce_score)
            reranked_candidates.append(updated_candidate)

        # Sort by cross-encoder score (descending)
        reranked_candidates.sort(key=lambda x: x['score'], reverse=True)

        # Limit to top_k if specified
        if top_k is not None:
            reranked_candidates = reranked_candidates[:top_k]

        # Update ranks
        for i, candidate in enumerate(reranked_candidates, 1):
            candidate['rank'] = i

        logger.info(f"Re-ranked {len(candidates)} candidates, returning top {len(reranked_candidates)}")
        return reranked_candidates

    def _create_document_text(self, metadata: Dict) -> str:
        """
        Create document text representation from metadata for cross-encoder input.

        Args:
            metadata: Result metadata dictionary

        Returns:
            Formatted document text
        """
        # Extract key information for cross-encoder
        paper_title = metadata.get('paper_title', '')
        repo_name = metadata.get('repo_name', '')
        description = metadata.get('description', '')
        language = metadata.get('language', '')
        topics = metadata.get('topics', [])
        code_snippet = metadata.get('code_snippet', '')
        file_path = metadata.get('file_path', '')

        # Build comprehensive document text
        doc_parts = []

        # Paper information
        if paper_title:
            doc_parts.append(f"Paper: {paper_title}")

        # Repository information
        if repo_name:
            doc_parts.append(f"Repository: {repo_name}")

        # Description
        if description:
            doc_parts.append(f"Description: {description}")

        # Technical details
        tech_info = []
        if language:
            tech_info.append(f"Language: {language}")
        if topics:
            tech_info.append(f"Topics: {', '.join(topics)}")
        if file_path and file_path != 'Repository-level (files pending Day 6)':
            tech_info.append(f"File: {file_path}")

        if tech_info:
            doc_parts.append("Technical: " + " | ".join(tech_info))

        # Code snippet (if available)
        if code_snippet and code_snippet != 'Function extraction pending (Day 6)':
            # Truncate very long code snippets
            if len(code_snippet) > 500:
                code_snippet = code_snippet[:500] + "..."
            doc_parts.append(f"Code: {code_snippet}")

        return " | ".join(doc_parts)

    def batch_rerank(
        self,
        queries: List[str],
        candidate_lists: List[List[Dict]],
        top_k: Optional[int] = None
    ) -> List[List[Dict]]:
        """
        Re-rank multiple query result sets.

        Args:
            queries: List of queries
            candidate_lists: List of candidate result lists
            top_k: Number of top results per query

        Returns:
            List of re-ranked result lists
        """
        reranked_results = []

        for query, candidates in zip(queries, candidate_lists):
            reranked = self.rerank(query, candidates, top_k=top_k)
            reranked_results.append(reranked)

        return reranked_results


class RerankingPipeline:
    """Complete re-ranking pipeline combining initial retrieval and cross-encoder re-ranking."""

    def __init__(
        self,
        dense_retriever,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        initial_top_k: int = 50,
        final_top_k: int = 20
    ):
        """
        Initialize re-ranking pipeline.

        Args:
            dense_retriever: Initialized DenseRetrieval instance
            cross_encoder_model: Cross-encoder model name
            initial_top_k: Number of candidates from initial retrieval
            final_top_k: Number of final results after re-ranking
        """
        self.dense_retriever = dense_retriever
        self.reranker = CrossEncoderReranker(model_name=cross_encoder_model)
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k

        logger.info("Initialized re-ranking pipeline")
        logger.info(f"Initial retrieval top-k: {initial_top_k}")
        logger.info(f"Final re-ranked top-k: {final_top_k}")

    def retrieve_and_rerank(
        self,
        query: str,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Perform retrieval and re-ranking in one pipeline.

        Args:
            query: Search query
            filters: Optional filters for initial retrieval

        Returns:
            Dictionary with initial and re-ranked results
        """
        # Initial retrieval (get more candidates than final top-k)
        initial_results = self.dense_retriever.retrieve(
            query,
            top_k=self.initial_top_k,
            filters=filters
        )

        # Re-rank using cross-encoder
        reranked_results = self.reranker.rerank(
            query,
            initial_results,
            top_k=self.final_top_k
        )

        return {
            'query': query,
            'initial_results': initial_results[:self.final_top_k],  # For comparison
            'reranked_results': reranked_results,
            'pipeline_stats': {
                'initial_candidates': len(initial_results),
                'final_results': len(reranked_results),
                'cross_encoder_model': self.reranker.model_name
            }
        }