#!/usr/bin/env python3
"""
Quick test to verify the results generation script can load data.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.retrieval.dense_retrieval import DenseRetrieval
    
    print("Testing data loading...")
    retriever = DenseRetrieval(embedding_model_name="microsoft/codebert-base", use_gpu=False)
    stats = retriever.get_statistics()
    
    print(f"✓ Successfully loaded {stats['total_vectors']} code snippets")
    print(f"✓ Embedding dimension: {stats['embedding_dim']}")
    print(f"✓ Model: {stats['embedding_model']}")
    
    # Test a quick retrieval
    print("\nTesting retrieval...")
    results = retriever.retrieve("transformer attention", top_k=1)
    if results:
        print(f"✓ Retrieval works! Top result score: {results[0]['score']:.3f}")
    else:
        print("⚠ No results returned")
    
    print("\n✓ All tests passed! The results script should work.")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

