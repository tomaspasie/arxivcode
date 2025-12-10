"""
Example: Integrating Retrieval System with LLM Explanations

This example shows how to combine the FAISS retrieval system
with the LLM explanation service for end-to-end code search
and explanation.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval import DenseRetrieval
from src.models.explanation_llm import ExplanationLLM


def main():
    # Load environment variables
    load_dotenv()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set in .env file")
        print("Please add your OpenAI API key to .env")
        return

    print("="*70)
    print("Retrieval + Explanation Pipeline Example")
    print("="*70)

    # Step 1: Initialize retrieval system
    print("\n1. Loading retrieval system...")
    retriever = DenseRetrieval(embedding_model_name="tfidf")

    # Check if index exists
    index_path = "data/processed/FAISS/faiss_index.index"
    metadata_path = "data/processed/FAISS/faiss_metadata.pkl"

    if not os.path.exists(index_path):
        print(f"❌ Index not found at {index_path}")
        print("Please build the index first:")
        print("  python -m src.retrieval.build_index --input data/raw/papers/paper_code_pairs.json")
        return

    retriever.load_index(index_path, metadata_path)
    print("✅ Retrieval system loaded")

    # Step 2: Initialize LLM
    print("\n2. Initializing LLM...")
    llm = ExplanationLLM(model="gpt-4o", temperature=0.3)
    print("✅ LLM initialized")

    # Step 3: Search for code
    print("\n" + "="*70)
    print("Step 3: Searching for code...")
    print("="*70)

    query = "contrastive learning"
    print(f"\nQuery: '{query}'")
    print("Retrieving top 5 results...")

    results = retriever.search(query, top_k=5)

    print(f"\n✅ Found {len(results)} results")

    # Step 4: Explain top results
    print("\n" + "="*70)
    print("Step 4: Generating explanations for top results")
    print("="*70)

    for i, result in enumerate(results[:3], 1):  # Explain top 3
        print(f"\n{'='*70}")
        print(f"Result {i}/{len(results[:3])}")
        print('='*70)

        metadata = result.get('metadata', {})

        # Display result info
        print(f"\n📄 Paper: {metadata.get('paper_title', 'Unknown')}")
        print(f"🔗 Repository: {metadata.get('repo_full_name', 'Unknown')}")
        print(f"⭐ Stars: {metadata.get('stars', 'N/A')}")
        print(f"📊 Similarity: {result.get('similarity', 0):.4f}")

        # Get code snippet (truncate if too long)
        code_snippet = metadata.get('code', '')
        if len(code_snippet) > 500:
            code_snippet = code_snippet[:500] + "\n... (truncated)"

        print(f"\n💻 Code Preview:")
        print("-" * 70)
        print(code_snippet[:200] + "..." if len(code_snippet) > 200 else code_snippet)
        print("-" * 70)

        # Generate explanation
        print("\n🤖 Generating explanation...")

        try:
            explanation = llm.generate_explanation(
                query=query,
                code_snippet=code_snippet,
                paper_title=metadata.get('paper_title', 'Unknown Paper'),
                paper_context=metadata.get('abstract', '')
            )

            print(f"\n📝 Explanation:\n{explanation}")

        except Exception as e:
            print(f"❌ Error generating explanation: {e}")

    # Summary
    print("\n" + "="*70)
    print("Pipeline Summary")
    print("="*70)
    print(f"✅ Query: '{query}'")
    print(f"✅ Retrieved: {len(results)} code snippets")
    print(f"✅ Explained: 3 top results")
    print("\nComplete pipeline flow:")
    print("  1. User query → FAISS retrieval")
    print("  2. Top results → LLM explanation")
    print("  3. Contextual explanations → User")
    print("\nThis demonstrates the full ArxivCode capability:")
    print("  Search → Retrieve → Explain")


def demonstrate_with_multiple_queries():
    """Demonstrate with multiple different queries"""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return

    print("\n" + "="*70)
    print("Multi-Query Demonstration")
    print("="*70)

    # Initialize systems
    retriever = DenseRetrieval(embedding_model_name="tfidf")
    index_path = "data/processed/FAISS/faiss_index.index"
    metadata_path = "data/processed/FAISS/faiss_metadata.pkl"

    if not os.path.exists(index_path):
        print("❌ Index not found")
        return

    retriever.load_index(index_path, metadata_path)
    llm = ExplanationLLM()

    # Test queries
    test_queries = [
        "attention mechanism",
        "batch normalization",
        "residual connections",
        "dropout regularization"
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: '{query}'")
        print('='*70)

        # Retrieve
        results = retriever.search(query, top_k=1)

        if results:
            result = results[0]
            metadata = result.get('metadata', {})

            print(f"📄 {metadata.get('paper_title', 'Unknown')[:60]}...")

            # Explain
            code = metadata.get('code', '')[:500]
            try:
                explanation = llm.generate_explanation(
                    query=query,
                    code_snippet=code,
                    paper_title=metadata.get('paper_title', ''),
                    paper_context=metadata.get('abstract', '')
                )
                print(f"📝 {explanation}")
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieval + Explanation Demo")
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Run multiple query demonstration"
    )

    args = parser.parse_args()

    if args.multi:
        demonstrate_with_multiple_queries()
    else:
        main()
