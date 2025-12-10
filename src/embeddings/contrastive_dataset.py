"""
Day 4 Step 3: Create Contrastive Learning Dataset
PyTorch Dataset class for tokenizing paper-code pairs
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional
import logging
from transformers import AutoTokenizer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from code_encoder_model import load_codebert
from paper_code_parser import parse_paper_code_pairs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContrastiveDataset(Dataset):
    """
    PyTorch Dataset for contrastive learning of paper-code pairs

    Takes (paper_text, code_text) pairs and tokenizes them using CodeBERT tokenizer
    """

    def __init__(
        self,
        pairs: Optional[List[Tuple[str, str, Dict]]] = None,
        json_path: Optional[str] = None,
        tokenizer=None,
        model_name: str = "microsoft/codebert-base",
        max_length: int = 512,
    ):
        """
        Initialize contrastive dataset

        Args:
            pairs: List of (paper_text, code_text, metadata) tuples from parser
                  If None, will load from json_path
            json_path: Path to parsed_pairs.json (used if pairs is None)
            tokenizer: CodeBERT tokenizer (loaded if None)
            model_name: Model name for tokenizer
            max_length: Maximum sequence length for tokenization
        """
        self.max_length = max_length

        # Load tokenizer
        if tokenizer is None:
            logger.info(f"Loading tokenizer: {model_name}")
            _, tokenizer, _ = load_codebert(model_name)
        self.tokenizer = tokenizer

        # Load pairs
        if pairs is None:
            if json_path is None:
                raise ValueError("Must provide either pairs or json_path")
            logger.info(f"Loading pairs from {json_path}")
            import json

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            pairs = [
                (item["paper_text"], item["code_text"], item["metadata"])
                for item in data
            ]

        self.pairs = pairs
        logger.info(f"Initialized dataset with {len(self.pairs)} pairs")
        logger.info(f"Max sequence length: {max_length}")

    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
                - paper_input_ids: Token IDs for paper text
                - paper_attention_mask: Attention mask for paper
                - code_input_ids: Token IDs for code text
                - code_attention_mask: Attention mask for code
                - metadata: Original metadata dict
        """
        paper_text, code_text, metadata = self.pairs[idx]

        # Tokenize paper text
        paper_encoded = self.tokenizer(
            paper_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize code text
        code_encoded = self.tokenizer(
            code_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "paper_input_ids": paper_encoded["input_ids"].squeeze(
                0
            ),  # Remove batch dim
            "paper_attention_mask": paper_encoded["attention_mask"].squeeze(0),
            "code_input_ids": code_encoded["input_ids"].squeeze(0),
            "code_attention_mask": code_encoded["attention_mask"].squeeze(0),
            "metadata": metadata,
        }


class ContrastiveDataCollator:
    """
    Data collator for batching ContrastiveDataset samples
    Handles proper batching of tokenized sequences
    """

    def __init__(self, pad_token_id: int = 0):
        """
        Initialize collator

        Args:
            pad_token_id: Padding token ID (usually 0)
        """
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples

        Args:
            batch: List of samples from ContrastiveDataset

        Returns:
            Batched tensors with shape (batch_size, seq_len)
        """
        # Extract all fields
        paper_input_ids = [item["paper_input_ids"] for item in batch]
        paper_attention_mask = [item["paper_attention_mask"] for item in batch]
        code_input_ids = [item["code_input_ids"] for item in batch]
        code_attention_mask = [item["code_attention_mask"] for item in batch]
        metadata = [item["metadata"] for item in batch]

        # Stack tensors (they're already padded to max_length)
        paper_input_ids = torch.stack(paper_input_ids)
        paper_attention_mask = torch.stack(paper_attention_mask)
        code_input_ids = torch.stack(code_input_ids)
        code_attention_mask = torch.stack(code_attention_mask)

        return {
            "paper_input_ids": paper_input_ids,
            "paper_attention_mask": paper_attention_mask,
            "code_input_ids": code_input_ids,
            "code_attention_mask": code_attention_mask,
            "metadata": metadata,  # Keep as list for easy access
        }


def create_dataset(
    json_path: str = "data/processed/parsed_pairs.json",
    model_name: str = "microsoft/codebert-base",
    max_length: int = 512,
) -> ContrastiveDataset:
    """
    Convenience function to create ContrastiveDataset

    Args:
        json_path: Path to parsed_pairs.json
        model_name: Model name for tokenizer
        max_length: Max sequence length

    Returns:
        ContrastiveDataset instance
    """
    return ContrastiveDataset(
        json_path=json_path, model_name=model_name, max_length=max_length
    )


if __name__ == "__main__":
    """
    Test script for Day 4 Step 3
    Tests dataset creation and batch loading
    """
    import sys
    from pathlib import Path
    from torch.utils.data import DataLoader

    # Add parent to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("=" * 60)
    print("Day 4 Step 3: Testing ContrastiveDataset")
    print("=" * 60)

    # Check if parsed_pairs.json exists, if not parse first
    parsed_path = Path("data/processed/parsed_pairs.json")
    if not parsed_path.exists():
        print("\nWARNING: parsed_pairs.json not found. Running parser first...")
        from src.embeddings.paper_code_parser import parse_paper_code_pairs

        pairs = parse_paper_code_pairs(
            json_path="data/raw/papers/paper_code_with_files.json",
            output_path=str(parsed_path),
        )
        print(f"✓ Parsed {len(pairs)} pairs")

    # Create dataset
    print("\n1. Creating ContrastiveDataset...")
    try:
        dataset = create_dataset(json_path=str(parsed_path), max_length=512)
        print(f"✓ Dataset created with {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        sys.exit(1)

    # Test single sample
    print("\n2. Testing single sample...")
    try:
        sample = dataset[0]
        print(f"✓ Sample retrieved")
        print(f"  Paper input_ids shape: {sample['paper_input_ids'].shape}")
        print(f"  Code input_ids shape: {sample['code_input_ids'].shape}")
        print(f"  Paper attention_mask shape: {sample['paper_attention_mask'].shape}")
        print(f"  Metadata keys: {list(sample['metadata'].keys())}")
    except Exception as e:
        print(f"✗ Error getting sample: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Test DataLoader
    print("\n3. Testing DataLoader with batches...")
    try:
        collator = ContrastiveDataCollator()
        dataloader = DataLoader(
            dataset, batch_size=4, shuffle=False, collate_fn=collator
        )

        # Get first batch
        batch = next(iter(dataloader))
        print(f"✓ Batch created")
        print(f"  Batch size: {batch['paper_input_ids'].shape[0]}")
        print(f"  Paper input_ids shape: {batch['paper_input_ids'].shape}")
        print(f"  Code input_ids shape: {batch['code_input_ids'].shape}")
        print(f"  Paper attention_mask shape: {batch['paper_attention_mask'].shape}")
        print(f"  Code attention_mask shape: {batch['code_attention_mask'].shape}")
        print(f"  Metadata items: {len(batch['metadata'])}")

        # Verify tokenization
        print("\n4. Verifying tokenization...")
        tokenizer = dataset.tokenizer
        paper_text = dataset.pairs[0][0]
        decoded = tokenizer.decode(
            batch["paper_input_ids"][0], skip_special_tokens=False
        )
        print(f"  Original paper text (first 100 chars): {paper_text[:100]}...")
        print(f"  Decoded tokens (first 100 chars): {decoded[:100]}...")
        print("✓ Tokenization verified")

    except Exception as e:
        print(f"✗ Error testing DataLoader: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
