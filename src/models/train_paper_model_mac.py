#!/usr/bin/env python3
"""
Mac-optimized training script for the paper comprehension model.
Uses Phi-2 (2.7B) with LoRA - smaller model that fits in available disk space.
MPS (Metal Performance Shaders) enabled for GPU acceleration on Mac.
"""

import os
import sys
import logging
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function optimized for Mac."""

    # Force CPU to avoid MPS memory issues
    import os
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = "cpu"
    logger.info("Using CPU for training (MPS disabled due to memory constraints)")

    # Configuration
    model_name = "microsoft/phi-2"  # Smaller model (2.7B, ~5GB) - fits on Mac
    train_data = "data/processed/train.json"
    eval_data = "data/processed/eval.json"
    output_dir = "output/checkpoints"
    logging_dir = "output/logs"

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Mac-Optimized Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name} (2.7B parameters)")
    logger.info(f"Device: {device}")
    logger.info(f"Training data: {train_data}")
    logger.info(f"Eval data: {eval_data}")
    logger.info(f"Model size: ~5GB (fits in available disk space)")
    logger.info("=" * 60)

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model (without quantization for Mac)
    logger.info("Loading model (this may take a few minutes)...")
    # Use CPU to avoid MPS buffer size limit
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map={"": "cpu"},  # Use CPU instead of MPS to avoid buffer limit
        low_cpu_mem_usage=True,
        trust_remote_code=True,  # Phi-2 requires this
    )
    logger.info("Model loaded on CPU (MPS has buffer size limitations for this model)")

    # Apply LoRA
    logger.info("Applying LoRA configuration...")
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA scaling
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": train_data,
            "eval": eval_data,
        }
    )

    # Tokenization function
    def tokenize_function(examples):
        texts = [
            f"{inp}\n{out}"
            for inp, out in zip(examples["input"], examples["output"])
        ]
        return tokenizer(
            texts,
            truncation=True,
            max_length=1024,  # Reduced for Mac memory
            padding="max_length",
        )

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names,
    )

    logger.info(f"Training samples: {len(tokenized_dataset['train'])}")
    logger.info(f"Evaluation samples: {len(tokenized_dataset['eval'])}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments (Mac-optimized for CPU)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,  # Reduced to 1 epoch for CPU training
        per_device_train_batch_size=1,  # Small batch for Mac
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Reduced for CPU
        learning_rate=2e-4,
        weight_decay=0.001,
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        logging_steps=10,
        logging_dir=logging_dir,
        save_steps=500,  # Less frequent saves for CPU
        eval_steps=500,
        save_total_limit=2,
        fp16=False,  # CPU training
        gradient_checkpointing=True,
        group_by_length=True,
        report_to="none",  # Disable tensorboard to save disk space
        eval_strategy="steps",  # Changed from evaluation_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,  # No multiprocessing for CPU
        use_cpu=True,  # Force CPU usage
        no_cuda=True,  # Disable CUDA
    )

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info("Saving final model...")
    final_output_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    logger.info("Training complete!")
    logger.info(f"Model saved to: {final_output_dir}")
    logger.info(f"\nTo view training progress with TensorBoard:")
    logger.info(f"  tensorboard --logdir {logging_dir}")


if __name__ == "__main__":
    main()
