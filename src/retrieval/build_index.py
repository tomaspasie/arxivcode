"""
Build FAISS index from collected paper-code pairs.
Run this after data collection (Day 3) is complete.
"""

import json
import argparse
from pathlib import Path
import logging

from .dense_retrieval import DenseRetrieval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_index_from_collection(
    input_path: str,
    output_dir: str,
    embedding_model: str = "tfidf"  # SWAP: Change to "microsoft/codebert-base" when switching to CodeBERT
):
    """
    Build FAISS index from collected paper-code pairs.
    
    Args:
        input_path: Path to paper-code pairs JSON
        output_dir: Directory to save index and metadata
        embedding_model: Model to use for embeddings
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    index_path = output_path / "faiss_index.index"
    metadata_path = output_path / "faiss_metadata.pkl"
    
    # Initialize retrieval system
    logger.info(f"Initializing retrieval with model: {embedding_model}")
    retriever = DenseRetrieval(embedding_model_name=embedding_model)
    
    # Build index
    logger.info(f"Building index from {input_path}")
    retriever.build_index_from_papers(
        input_path,
        save_index_path=str(index_path),
        save_metadata_path=str(metadata_path)
    )
    
    # Print statistics
    stats = retriever.get_statistics()
    logger.info("\n" + "="*60)
    logger.info("INDEX BUILD COMPLETE")
    logger.info("="*60)
    logger.info(f"Total vectors indexed: {stats['total_vectors']}")
    logger.info(f"Embedding dimension: {stats['embedding_dim']}")
    logger.info(f"Index saved to: {index_path}")
    logger.info(f"Metadata saved to: {metadata_path}")
    logger.info("="*60)
    
    return str(index_path), str(metadata_path)


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index from collected paper-code pairs"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/papers/paper_code_pairs.json",
        help="Path to input paper-code pairs JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/FAISS",
        help="Output directory for index files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tfidf",  # Changed default to TF-IDF for macOS stability
        help="Embedding model name ('tfidf' or sentence-transformers model)"
    )
    
    args = parser.parse_args()
    
    # Check if input exists
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        logger.info("Please run data collection first:")
        logger.info("  python -m src.data_collection.collect_papers")
        return
    
    # Build index
    build_index_from_collection(
        input_path=args.input,
        output_dir=args.output,
        embedding_model=args.model
    )


if __name__ == "__main__":
    main()