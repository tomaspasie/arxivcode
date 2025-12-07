#!/usr/bin/env python3
"""
Test script for Cross-Encoder Re-ranking Pipeline
Tests the re-ranking functionality with the existing FAISS index.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval import DenseRetrieval, CrossEncoderReranker, RerankingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_cross_encoder_reranking():
    """Test the cross-encoder re-ranking pipeline."""
    print("\n" + "="*70)
    print("TESTING CROSS-ENCODER RE-RANKING PIPELINE")
    print("="*70)

    # 1. Load existing retrieval system
    print("\n1. Loading existing retrieval system...")
    retriever = DenseRetrieval(embedding_model_name="tfidf")
    retriever.load_index(
        "data/processed/FAISS/faiss_index.index",
        "data/processed/FAISS/faiss_metadata.pkl"
    )
    print("   ✅ Retrieval system loaded")

    # 2. Initialize cross-encoder reranker
    print("\n2. Initializing cross-encoder reranker...")
    try:
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu"
        )
        print("   ✅ Cross-encoder reranker initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize reranker: {e}")
        return

    # 3. Initialize complete pipeline
    print("\n3. Setting up re-ranking pipeline...")
    pipeline = RerankingPipeline(
        dense_retriever=retriever,
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        initial_top_k=50,  # Get more candidates for re-ranking
        final_top_k=20
    )
    print("   ✅ Re-ranking pipeline ready")

    # 4. Test queries
    test_queries = [
        "implement transformer attention mechanism",
        "fine-tune large language model",
        "parameter efficient training methods",
        "graph neural network",
        "self-attention layer code"
    ]

    print("\n4. Testing re-ranking with sample queries...")
    print("-" * 70)

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: '{query}'")
        print("-" * 70)

        # Get pipeline results
        results = pipeline.retrieve_and_rerank(query)

        print(f"Initial candidates: {results['pipeline_stats']['initial_candidates']}")
        print(f"Final results: {results['pipeline_stats']['final_results']}")

        # Show top 3 re-ranked results
        print("\nTop 3 Re-ranked Results:")
        for result in results['reranked_results'][:3]:
            meta = result['metadata']
            print(f"\nRank {result['rank']} (CE Score: {result['score']:.4f})")
            print(f"  📄 Paper: {meta['paper_title'][:60]}...")
            print(f"  🔗 Repo: {meta['repo_name']}")
            print(f"  ⭐ Stars: {meta['star_count']}")
            print(f"  📁 File: {meta['file_path']}")

            # Show score improvement if available
            if 'original_score' in result:
                improvement = result['score'] - result['original_score']
                print(".4f")

    # 5. Compare initial vs re-ranked for one query
    print("\n5. Detailed comparison for first query...")
    print("-" * 70)

    query = test_queries[0]
    results = pipeline.retrieve_and_rerank(query)

    print(f"Query: '{query}'")
    print("\nInitial Retrieval (Top 5):")
    for i, result in enumerate(results['initial_results'][:5], 1):
        meta = result['metadata']
        print(f"{i}. {meta['paper_title'][:50]}... (Score: {result['score']:.4f})")

    print("\nAfter Cross-Encoder Re-ranking (Top 5):")
    for i, result in enumerate(results['reranked_results'][:5], 1):
        meta = result['metadata']
        print(f"{i}. {meta['paper_title'][:50]}... (Score: {result['score']:.4f})")

    # 6. Performance statistics
    print("\n6. Pipeline Performance Statistics")
    print("-" * 70)

    total_initial = 0
    total_reranked = 0

    for query in test_queries:
        results = pipeline.retrieve_and_rerank(query)
        total_initial += results['pipeline_stats']['initial_candidates']
        total_reranked += results['pipeline_stats']['final_results']

    avg_initial = total_initial / len(test_queries)
    avg_reranked = total_reranked / len(test_queries)

    print(f"Average initial candidates per query: {avg_initial:.1f}")
    print(f"Average final results per query: {avg_reranked:.1f}")
    print(f"Re-ranking compression ratio: {avg_initial/avg_reranked:.1f}x")

    print("\n" + "="*70)
    print("✅ CROSS-ENCODER RE-RANKING TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_cross_encoder_reranking()