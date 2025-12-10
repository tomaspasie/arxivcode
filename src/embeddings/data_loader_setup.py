"""
Day 4 Step 4: Setup DataLoader for Training
Creates train/val splits and DataLoaders ready for training
"""

import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging
import json

from contrastive_dataset import ContrastiveDataset, ContrastiveDataCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_data_loaders(
    json_path: str = "data/processed/parsed_pairs.json",
    model_name: str = "microsoft/codebert-base",
    max_length: int = 512,
    batch_size: int = 8,
    train_split: float = 0.8,
    shuffle_train: bool = True,
    num_workers: int = 0,
    seed: Optional[int] = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders

    Args:
        json_path: Path to parsed_pairs.json
        model_name: Model name for tokenizer
        max_length: Maximum sequence length
        batch_size: Batch size for training
        train_split: Fraction of data for training (rest for validation)
        shuffle_train: Whether to shuffle training data
        num_workers: Number of worker processes (0 for single process)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info("=" * 60)
    logger.info("Day 4 Step 4: Setting up DataLoaders")
    logger.info("=" * 60)

    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Create full dataset
    logger.info(f"Loading dataset from {json_path}")
    full_dataset = ContrastiveDataset(
        json_path=json_path, model_name=model_name, max_length=max_length
    )

    logger.info(f"Full dataset size: {len(full_dataset)} samples")

    # Split into train and validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    logger.info(f"Splitting dataset: {train_size} train, {val_size} val")

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")

    # Create data collator
    collator = ContrastiveDataCollator()

    # Create DataLoaders
    logger.info(
        f"Creating DataLoaders (batch_size={batch_size}, num_workers={num_workers})"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Pin memory if using GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info("✓ DataLoaders created successfully")
    logger.info("=" * 60)

    return train_loader, val_loader


def save_dataset_info(
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_path: str = "data/processed/dataset_info.json",
    config: Optional[Dict] = None,
):
    """
    Save dataset information and configuration for reproducibility

    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        output_path: Path to save info JSON
        config: Optional configuration dictionary to save
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get batch info
    train_info = get_batch_info(train_loader)
    val_info = get_batch_info(val_loader)

    # Create info dictionary
    info = {
        "train": {
            "num_batches": train_info["num_batches"],
            "batch_size": train_info["batch_size"],
            "total_samples": train_info["num_batches"] * train_info["batch_size"],
        },
        "val": {
            "num_batches": val_info["num_batches"],
            "batch_size": val_info["batch_size"],
            "total_samples": val_info["num_batches"] * val_info["batch_size"],
        },
        "data_shapes": {
            "paper_input_ids": train_info["paper_input_ids_shape"],
            "code_input_ids": train_info["code_input_ids_shape"],
            "paper_attention_mask": train_info["paper_attention_mask_shape"],
            "code_attention_mask": train_info["code_attention_mask_shape"],
        },
    }

    if config:
        info["config"] = config

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Saved dataset info to {output_path}")
    return info


def get_batch_info(loader: DataLoader) -> dict:
    """
    Get information about a DataLoader

    Args:
        loader: DataLoader to inspect

    Returns:
        Dictionary with batch information
    """
    # Get a sample batch
    sample_batch = next(iter(loader))

    info = {
        "num_batches": len(loader),
        "batch_size": sample_batch["paper_input_ids"].shape[0],
        "paper_input_ids_shape": list(sample_batch["paper_input_ids"].shape),
        "code_input_ids_shape": list(sample_batch["code_input_ids"].shape),
        "paper_attention_mask_shape": list(sample_batch["paper_attention_mask"].shape),
        "code_attention_mask_shape": list(sample_batch["code_attention_mask"].shape),
        "num_metadata_items": len(sample_batch["metadata"]),
    }

    return info


def test_data_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_test_batches: int = 3,
):
    """
    Test DataLoaders by iterating through a few batches

    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_test_batches: Number of batches to test
    """
    logger.info("=" * 60)
    logger.info("Testing DataLoaders")
    logger.info("=" * 60)

    # Test training loader
    logger.info("\n1. Testing training loader...")
    train_batches = 0
    for i, batch in enumerate(train_loader):
        if i >= num_test_batches:
            break
        train_batches += 1
        logger.info(f"  Batch {i+1}: shape={batch['paper_input_ids'].shape}")

    logger.info(f"✓ Training loader: {train_batches} batches tested")

    # Test validation loader
    logger.info("\n2. Testing validation loader...")
    val_batches = 0
    for i, batch in enumerate(val_loader):
        if i >= num_test_batches:
            break
        val_batches += 1
        logger.info(f"  Batch {i+1}: shape={batch['paper_input_ids'].shape}")

    logger.info(f"✓ Validation loader: {val_batches} batches tested")

    # Print batch info
    logger.info("\n3. Batch information:")
    train_info = get_batch_info(train_loader)
    val_info = get_batch_info(val_loader)

    logger.info(f"  Train batches: {train_info['num_batches']}")
    logger.info(f"  Train batch size: {train_info['batch_size']}")
    logger.info(f"  Val batches: {val_info['num_batches']}")
    logger.info(f"  Val batch size: {val_info['batch_size']}")

    logger.info("=" * 60)
    logger.info("DataLoader testing complete! ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    """
    Test script for Day 4 Step 4
    Creates and tests DataLoaders
    """
    import sys
    from pathlib import Path

    # Add parent to path
    sys.path.insert(0, str(Path(__file__).parent))

    print("=" * 60)
    print("Day 4 Step 4: Testing DataLoader Setup")
    print("=" * 60)

    # Check if parsed_pairs.json exists
    parsed_path = Path("data/processed/parsed_pairs.json")
    if not parsed_path.exists():
        print("\n⚠️  parsed_pairs.json not found. Running parser first...")
        from paper_code_parser import parse_paper_code_pairs

        pairs = parse_paper_code_pairs(
            json_path="data/raw/papers/paper_code_with_files.json",
            output_path=str(parsed_path),
        )
        print(f"✓ Parsed {len(pairs)} pairs")

    # Create DataLoaders
    print("\n1. Creating DataLoaders...")
    try:
        train_loader, val_loader = create_data_loaders(
            json_path=str(parsed_path),
            batch_size=8,
            train_split=0.8,
            num_workers=0,
        )
        print("✓ DataLoaders created")
    except Exception as e:
        print(f"✗ Error creating DataLoaders: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Test DataLoaders
    print("\n2. Testing DataLoaders...")
    try:
        test_data_loaders(train_loader, val_loader, num_test_batches=3)
    except Exception as e:
        print(f"✗ Error testing DataLoaders: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Save dataset info
    print("\n3. Saving dataset information...")
    try:
        config = {
            "json_path": str(parsed_path),
            "model_name": "microsoft/codebert-base",
            "max_length": 512,
            "batch_size": 8,
            "train_split": 0.8,
            "num_workers": 0,
            "seed": 42,
        }
        dataset_info = save_dataset_info(
            train_loader,
            val_loader,
            output_path="data/processed/dataset_info.json",
            config=config,
        )
        print("✓ Dataset info saved")
        print(f"  Train samples: {dataset_info['train']['total_samples']}")
        print(f"  Val samples: {dataset_info['val']['total_samples']}")
    except Exception as e:
        print(f"⚠️  Warning: Could not save dataset info: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("Day 4 Step 4: COMPLETE ✓")
    print("=" * 60)
    print("\n📁 Files created/saved:")
    print("  1. data/processed/parsed_pairs.json - Parsed paper-code text pairs")
    print("  2. data/processed/dataset_info.json - Dataset statistics & config")
    print("\n✅ Ready for Day 5: InfoNCE Loss & Training Loop")
