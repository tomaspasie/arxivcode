#!/usr/bin/env python3
"""
Simple training script for the paper comprehension model.
Uses Mistral-7B with QLoRA for efficient fine-tuning.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.config import get_config, DataConfig
from models.trainer import train

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""

    # Use the paper_understanding preset (Mistral-7B, optimized for paper QA)
    logger.info("Loading 'paper_understanding' configuration...")
    qlora_config = get_config("paper_understanding")

    # Data configuration
    data_config = DataConfig(
        train_data_path="data/processed/train.json",
        eval_data_path="data/processed/eval.json",
        input_field="input",
        output_field="output",
    )

    # Show configuration
    logger.info("=" * 60)
    logger.info("Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {qlora_config.model_name}")
    logger.info(f"LoRA rank: {qlora_config.lora_r}")
    logger.info(f"LoRA alpha: {qlora_config.lora_alpha}")
    logger.info(f"Max sequence length: {qlora_config.max_seq_length}")
    logger.info(f"Batch size: {qlora_config.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {qlora_config.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {qlora_config.per_device_train_batch_size * qlora_config.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {qlora_config.learning_rate}")
    logger.info(f"Epochs: {qlora_config.num_train_epochs}")
    logger.info(f"Output directory: {qlora_config.output_dir}")
    logger.info("=" * 60)

    # Start training
    logger.info("Starting training...")
    trainer = train(
        qlora_config=qlora_config,
        data_config=data_config,
    )

    logger.info("Training complete!")
    logger.info(f"Model saved to: {qlora_config.output_dir}/final")
    logger.info(f"Logs available at: {qlora_config.logging_dir}")
    logger.info("\nTo view training progress with TensorBoard:")
    logger.info(f"  tensorboard --logdir {qlora_config.logging_dir}")


if __name__ == "__main__":
    main()
