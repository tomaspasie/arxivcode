"""
Generate Code Embeddings from Pretrained CodeBERT
Loops through paper_code_with_files.json and generates embeddings for each code file.

This script performs INFERENCE ONLY - it does NOT fine-tune CodeBERT.
It simply uses the pretrained model to generate embeddings for your code files.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import logging
import sys

# Handle imports
sys.path.insert(0, str(Path(__file__).parent))
from code_encoder_model import CodeEncoder

# Import FAISS index manager for building retrieval index
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from retrieval.faiss_index import FAISSIndexManager
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_embedding(
    encoder: CodeEncoder,
    code_text: str,
    max_length: int = 512
) -> np.ndarray:
    """
    Generate embedding for a single code text using CodeBERT.
    
    Args:
        encoder: CodeEncoder instance (pretrained CodeBERT)
        code_text: Code content as string
        max_length: Maximum sequence length for tokenization
    
    Returns:
        Embedding vector as numpy array (embedding_dim,)
    """
    # Tokenize
    inputs = encoder.tokenizer(
        code_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    # Move to device
    input_ids = inputs["input_ids"].to(encoder.device)
    attention_mask = inputs["attention_mask"].to(encoder.device)
    
    # Generate embedding (inference mode - no gradients)
    encoder.model.eval()
    with torch.no_grad():
        embedding = encoder(input_ids, attention_mask)
    
    # Convert to numpy and squeeze batch dimension
    embedding_np = embedding.cpu().numpy().squeeze(0)
    
    return embedding_np


def generate_embeddings_batch(
    encoder: CodeEncoder,
    code_texts: List[str],
    max_length: int = 512
) -> np.ndarray:
    """
    Generate embeddings for a batch of code texts (more efficient).
    
    Args:
        encoder: CodeEncoder instance
        code_texts: List of code content strings
        max_length: Maximum sequence length
    
    Returns:
        Embeddings array (batch_size, embedding_dim)
    """
    # Tokenize batch
    inputs = encoder.tokenizer(
        code_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
    # Move to device
    input_ids = inputs["input_ids"].to(encoder.device)
    attention_mask = inputs["attention_mask"].to(encoder.device)
    
    # Generate embeddings (inference mode)
    encoder.model.eval()
    with torch.no_grad():
        embeddings = encoder(input_ids, attention_mask)
    
    # Convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    
    return embeddings_np


def process_paper_code_with_files(
    json_path: str,
    output_path: Optional[str] = None,
    model_name: str = "microsoft/codebert-base",
    batch_size: int = 8,
    max_length: int = 512,
    device: Optional[str] = None
) -> List[Dict]:
    """
    Process paper_code_with_files.json and generate embeddings for each code file.
    
    Args:
        json_path: Path to paper_code_with_files.json
        output_path: Optional path to save embeddings JSON
        model_name: CodeBERT model name
        batch_size: Batch size for processing (larger = faster but more memory)
        max_length: Maximum sequence length for tokenization
        device: Device to use ('cuda', 'cpu', or None for auto)
    
    Returns:
        List of dictionaries with embeddings and metadata
    """
    logger.info("=" * 60)
    logger.info("Generating Code Embeddings from Pretrained CodeBERT")
    logger.info("=" * 60)
    logger.info(f"Input JSON: {json_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max length: {max_length}")
    logger.info("=" * 60)
    
    # Load JSON file
    json_path_obj = Path(json_path)
    if not json_path_obj.exists():
        raise FileNotFoundError(f"File not found: {json_path}")
    
    logger.info(f"Loading JSON file: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} papers with repositories")
    
    # Load CodeBERT encoder (pretrained, no fine-tuning)
    logger.info(f"\nLoading CodeBERT model: {model_name}")
    logger.info("NOTE: This is INFERENCE ONLY - the model will NOT be fine-tuned.")
    encoder = CodeEncoder(
        model_name=model_name,
        max_length=max_length,
        device=device
    )
    encoder.model.eval()  # Ensure eval mode (no training)
    
    # Collect all code files with metadata
    code_items = []
    for entry_idx, entry in enumerate(data, 1):
        paper = entry.get("paper", {})
        repositories = entry.get("repositories", [])
        
        if not paper:
            continue
        
        arxiv_id = paper.get("arxiv_id", "")
        paper_title = paper.get("title", "")
        
        for repo in repositories:
            if not repo.get("cloned", False):
                continue
            
            repo_name = repo.get("name", "")
            repo_url = repo.get("url", "")
            code_files = repo.get("code_files", [])
            
            for code_file in code_files:
                code_content = code_file.get("content", "")
                if not code_content.strip():
                    continue
                
                code_items.append({
                    "code_text": code_content,
                    "metadata": {
                        "arxiv_id": arxiv_id,
                        "paper_title": paper_title,
                        "repo_name": repo_name,
                        "repo_url": repo_url,
                        "code_file_path": code_file.get("path", ""),
                        "code_language": code_file.get("language", ""),
                        "code_file_extension": code_file.get("extension", ""),
                        "code_file_lines": code_file.get("lines", 0),
                        "entry_index": entry_idx,
                    }
                })
    
    logger.info(f"\nFound {len(code_items)} code files to process")
    
    # Process in batches
    results = []
    num_batches = (len(code_items) + batch_size - 1) // batch_size
    
    logger.info(f"\nGenerating embeddings (processing {num_batches} batches)...")
    
    for batch_idx in tqdm(range(0, len(code_items), batch_size), desc="Processing batches"):
        batch_items = code_items[batch_idx:batch_idx + batch_size]
        batch_code_texts = [item["code_text"] for item in batch_items]
        batch_metadata = [item["metadata"] for item in batch_items]
        
        # Generate embeddings for batch
        try:
            embeddings = generate_embeddings_batch(
                encoder=encoder,
                code_texts=batch_code_texts,
                max_length=max_length
            )
            
            # Store results
            for i, (embedding, metadata) in enumerate(zip(embeddings, batch_metadata)):
                results.append({
                    "embedding": embedding.tolist(),  # Convert to list for JSON
                    "embedding_dim": len(embedding),
                    "metadata": metadata
                })
        
        except Exception as e:
            logger.warning(f"Error processing batch {batch_idx // batch_size + 1}: {e}")
            # Fallback to individual processing for this batch
            for item in batch_items:
                try:
                    embedding = generate_embedding(
                        encoder=encoder,
                        code_text=item["code_text"],
                        max_length=max_length
                    )
                    results.append({
                        "embedding": embedding.tolist(),
                        "embedding_dim": len(embedding),
                        "metadata": item["metadata"]
                    })
                except Exception as e2:
                    logger.warning(f"Error processing individual file: {e2}")
                    continue
        
        # Clear cache periodically
        if encoder.device == "cuda" and (batch_idx // batch_size) % 10 == 0:
            torch.cuda.empty_cache()
    
    logger.info(f"\n✓ Generated embeddings for {len(results)} code files")
    
    # Save results if output path provided
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving embeddings to: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"✓ Saved {len(results)} embeddings to {output_path}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total code files processed: {len(results)}")
    if results:
        logger.info(f"Embedding dimension: {results[0]['embedding_dim']}")
        logger.info(f"Sample metadata:")
        sample_meta = results[0]["metadata"]
        logger.info(f"  ArXiv ID: {sample_meta.get('arxiv_id', 'N/A')}")
        logger.info(f"  Paper: {sample_meta.get('paper_title', 'N/A')[:50]}...")
        logger.info(f"  Repo: {sample_meta.get('repo_name', 'N/A')}")
        logger.info(f"  File: {sample_meta.get('code_file_path', 'N/A')}")
    logger.info("=" * 60)
    
    return results


def build_faiss_index_from_embeddings(
    embeddings_json_path: str,
    faiss_index_path: str,
    faiss_metadata_path: str,
    embedding_dim: int = 768,
    index_type: str = "FlatIP",
    use_gpu: bool = False
) -> None:
    """
    Build FAISS index from embeddings JSON file.
    
    Args:
        embeddings_json_path: Path to embeddings JSON file (output from process_paper_code_with_files)
        faiss_index_path: Path to save FAISS index (.index file)
        faiss_metadata_path: Path to save metadata (.pkl file)
        embedding_dim: Embedding dimension (default: 768 for CodeBERT-base)
        index_type: FAISS index type ("FlatIP", "FlatL2", "IVFFlat")
        use_gpu: Whether to use GPU for FAISS
    """
    if not FAISS_AVAILABLE:
        raise ImportError(
            "FAISS is not available. Please install it: pip install faiss-cpu or faiss-gpu"
        )
    
    logger.info("=" * 60)
    logger.info("Building FAISS Index from Embeddings")
    logger.info("=" * 60)
    logger.info(f"Loading embeddings from: {embeddings_json_path}")
    
    # Load embeddings JSON
    with open(embeddings_json_path, "r", encoding="utf-8") as f:
        embeddings_data = json.load(f)
    
    logger.info(f"Loaded {len(embeddings_data)} embeddings")
    
    # Extract embeddings and metadata
    embeddings_list = []
    metadata_list = []
    
    for item in tqdm(embeddings_data, desc="Preparing data"):
        embedding = np.array(item["embedding"], dtype=np.float32)
        embeddings_list.append(embedding)
        
        # Format metadata for FAISS (match expected format)
        meta = item["metadata"]
        metadata_list.append({
            "paper_id": meta.get("arxiv_id", ""),
            "paper_title": meta.get("paper_title", ""),
            "repo_name": meta.get("repo_name", ""),
            "repo_url": meta.get("repo_url", ""),
            "file_path": meta.get("code_file_path", ""),
            "code_language": meta.get("code_language", ""),
            "code_file_extension": meta.get("code_file_extension", ""),
            "code_file_lines": meta.get("code_file_lines", 0),
            "code_snippet": f"File: {meta.get('code_file_path', 'N/A')}",  # Could include actual snippet
        })
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    logger.info(f"Embeddings array shape: {embeddings_array.shape}")
    
    # Verify embedding dimension
    if embeddings_array.shape[1] != embedding_dim:
        logger.warning(
            f"Embedding dimension mismatch: expected {embedding_dim}, "
            f"got {embeddings_array.shape[1]}. Using actual dimension."
        )
        embedding_dim = embeddings_array.shape[1]
    
    # Create FAISS index manager
    logger.info(f"\nCreating FAISS index (type: {index_type}, dim: {embedding_dim})...")
    index_manager = FAISSIndexManager(
        embedding_dim=embedding_dim,
        index_type=index_type,
        use_gpu=use_gpu
    )
    
    # Add embeddings to index
    logger.info("Adding embeddings to FAISS index...")
    index_manager.add_embeddings(embeddings_array, metadata_list)
    
    # Save index
    logger.info(f"Saving FAISS index to: {faiss_index_path}")
    logger.info(f"Saving metadata to: {faiss_metadata_path}")
    index_manager.save(faiss_index_path, faiss_metadata_path)
    
    # Print statistics
    stats = index_manager.get_stats()
    logger.info("\n" + "=" * 60)
    logger.info("FAISS Index Build Complete!")
    logger.info("=" * 60)
    logger.info(f"Total vectors: {stats['total_vectors']}")
    logger.info(f"Embedding dimension: {stats['embedding_dim']}")
    logger.info(f"Index type: {stats['index_type']}")
    logger.info(f"Index saved to: {faiss_index_path}")
    logger.info(f"Metadata saved to: {faiss_metadata_path}")
    logger.info("=" * 60)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate embeddings from pretrained CodeBERT for code files in paper_code_with_files.json"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="data/raw/papers/paper_code_with_files.json",
        help="Path to paper_code_with_files.json"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/code_embeddings.json",
        help="Path to save embeddings JSON (default: data/processed/code_embeddings.json)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/codebert-base",
        help="CodeBERT model name (default: microsoft/codebert-base)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8, increase for faster processing if you have GPU memory)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512, reduce if OOM)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cuda', 'cpu', or None for auto)"
    )
    parser.add_argument(
        "--build_faiss",
        action="store_true",
        help="Build FAISS index from embeddings after generation"
    )
    parser.add_argument(
        "--faiss_index_path",
        type=str,
        default=None,
        help="Path to save FAISS index (default: <output_path directory>/faiss_index.index)"
    )
    parser.add_argument(
        "--faiss_metadata_path",
        type=str,
        default=None,
        help="Path to save FAISS metadata (default: <output_path directory>/faiss_metadata.pkl)"
    )
    parser.add_argument(
        "--faiss_index_type",
        type=str,
        default="FlatIP",
        choices=["FlatIP", "FlatL2", "IVFFlat"],
        help="FAISS index type (default: FlatIP for cosine similarity)"
    )
    parser.add_argument(
        "--faiss_use_gpu",
        action="store_true",
        help="Use GPU for FAISS index (requires faiss-gpu)"
    )
    
    args = parser.parse_args()
    
    # Generate embeddings
    results = process_paper_code_with_files(
        json_path=args.json_path,
        output_path=args.output_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    )
    
    print("\n" + "=" * 60)
    print("✅ Embedding Generation Complete!")
    print("=" * 60)
    print(f"Generated {len(results)} embeddings")
    print(f"Saved to: {args.output_path}")
    print("\nNOTE: This script performs INFERENCE ONLY.")
    print("The pretrained CodeBERT model was NOT fine-tuned.")
    print("It simply generated embeddings using the pretrained weights.")
    
    # Build FAISS index if requested
    if args.build_faiss:
        if not FAISS_AVAILABLE:
            print("\n⚠️  FAISS not available. Skipping FAISS index build.")
            print("   Install FAISS: pip install faiss-cpu (or faiss-gpu)")
        else:
            # Determine FAISS output paths
            output_dir = Path(args.output_path).parent
            faiss_index_path = args.faiss_index_path or str(output_dir / "faiss_index.index")
            faiss_metadata_path = args.faiss_metadata_path or str(output_dir / "faiss_metadata.pkl")
            
            # Get embedding dimension from results
            embedding_dim = results[0]["embedding_dim"] if results else 768
            
            print("\n" + "=" * 60)
            print("Building FAISS Index...")
            print("=" * 60)
            
            build_faiss_index_from_embeddings(
                embeddings_json_path=args.output_path,
                faiss_index_path=faiss_index_path,
                faiss_metadata_path=faiss_metadata_path,
                embedding_dim=embedding_dim,
                index_type=args.faiss_index_type,
                use_gpu=args.faiss_use_gpu
            )
            
            print("\n✅ FAISS Index Ready for Retrieval!")
            print(f"   Index: {faiss_index_path}")
            print(f"   Metadata: {faiss_metadata_path}")
            print("\nYou can now use this index with the retrieval system:")
            print("   from src.retrieval.faiss_index import FAISSIndexManager")
            print("   manager = FAISSIndexManager(embedding_dim=768)")
            print(f"   manager.load('{faiss_index_path}', '{faiss_metadata_path}')")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

