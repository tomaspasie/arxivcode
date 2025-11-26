"""
Training script for fine-tuning models with QLoRA.
"""

import os
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from typing import Optional
import logging

from .model_loader import load_model_and_tokenizer, get_bnb_config, get_lora_config
from .config import QLoRAConfig, DataConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_dataset(
    data_config: DataConfig,
    tokenizer,
    qlora_config: QLoRAConfig,
):
    """
    Load and prepare dataset for training.

    Args:
        data_config: Data configuration
        tokenizer: Model tokenizer
        qlora_config: QLoRA configuration

    Returns:
        Prepared dataset
    """
    logger.info(f"Loading dataset from {data_config.train_data_path}")

    # Load dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": data_config.train_data_path,
            "eval": data_config.eval_data_path if data_config.eval_data_path else None,
        }
    )

    # Tokenization function
    def tokenize_function(examples):
        # Handle both single-field and instruction-tuning formats
        if data_config.output_field:
            # Instruction tuning format
            texts = [
                f"{inp}\n{out}"
                for inp, out in zip(
                    examples[data_config.input_field],
                    examples[data_config.output_field]
                )
            ]
        else:
            # Simple text format
            texts = examples[data_config.input_field]

        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=qlora_config.max_seq_length,
            padding="max_length",
            return_tensors=None,
        )
        return tokenized

    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=data_config.preprocessing_num_workers,
        remove_columns=dataset["train"].column_names,
    )

    # Limit samples if specified
    if data_config.max_train_samples:
        tokenized_dataset["train"] = tokenized_dataset["train"].select(
            range(min(data_config.max_train_samples, len(tokenized_dataset["train"])))
        )
    if data_config.max_eval_samples and "eval" in tokenized_dataset:
        tokenized_dataset["eval"] = tokenized_dataset["eval"].select(
            range(min(data_config.max_eval_samples, len(tokenized_dataset["eval"])))
        )

    logger.info(f"Training samples: {len(tokenized_dataset['train'])}")
    if "eval" in tokenized_dataset:
        logger.info(f"Evaluation samples: {len(tokenized_dataset['eval'])}")

    return tokenized_dataset


def get_training_arguments(qlora_config: QLoRAConfig) -> TrainingArguments:
    """
    Create TrainingArguments from QLoRAConfig.

    Args:
        qlora_config: QLoRA configuration

    Returns:
        TrainingArguments object
    """
    return TrainingArguments(
        output_dir=qlora_config.output_dir,
        num_train_epochs=qlora_config.num_train_epochs,
        per_device_train_batch_size=qlora_config.per_device_train_batch_size,
        per_device_eval_batch_size=qlora_config.per_device_eval_batch_size,
        gradient_accumulation_steps=qlora_config.gradient_accumulation_steps,
        learning_rate=qlora_config.learning_rate,
        weight_decay=qlora_config.weight_decay,
        warmup_ratio=qlora_config.warmup_ratio,
        max_grad_norm=qlora_config.max_grad_norm,
        optim=qlora_config.optim,
        lr_scheduler_type=qlora_config.lr_scheduler_type,
        logging_steps=qlora_config.logging_steps,
        logging_dir=qlora_config.logging_dir,
        save_steps=qlora_config.save_steps,
        eval_steps=qlora_config.eval_steps,
        save_total_limit=qlora_config.save_total_limit,
        fp16=qlora_config.fp16,
        bf16=qlora_config.bf16,
        gradient_checkpointing=qlora_config.gradient_checkpointing,
        group_by_length=qlora_config.group_by_length,
        report_to=qlora_config.report_to,
        do_eval=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )


def train(
    qlora_config: Optional[QLoRAConfig] = None,
    data_config: Optional[DataConfig] = None,
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Main training function.

    Args:
        qlora_config: QLoRA configuration (uses default if None)
        data_config: Data configuration (uses default if None)
        resume_from_checkpoint: Path to checkpoint to resume from
    """
    # Use default configs if not provided
    if qlora_config is None:
        qlora_config = QLoRAConfig()
    if data_config is None:
        data_config = DataConfig()

    # Create output directories
    os.makedirs(qlora_config.output_dir, exist_ok=True)
    os.makedirs(qlora_config.logging_dir, exist_ok=True)

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    bnb_config = get_bnb_config(
        load_in_4bit=qlora_config.load_in_4bit,
        bnb_4bit_quant_type=qlora_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=qlora_config.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=qlora_config.bnb_4bit_use_double_quant,
    )

    lora_config = get_lora_config(
        r=qlora_config.lora_r,
        lora_alpha=qlora_config.lora_alpha,
        target_modules=qlora_config.lora_target_modules,
        lora_dropout=qlora_config.lora_dropout,
        bias=qlora_config.lora_bias,
    )

    model, tokenizer = load_model_and_tokenizer(
        model_name=qlora_config.model_name,
        use_qlora=qlora_config.use_qlora,
        bnb_config=bnb_config,
        lora_config=lora_config,
        trust_remote_code=qlora_config.trust_remote_code,
    )

    # Prepare dataset
    logger.info("Preparing dataset...")
    dataset = prepare_dataset(data_config, tokenizer, qlora_config)

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Create training arguments
    training_args = get_training_arguments(qlora_config)

    # Create trainer
    logger.info("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("eval"),
        data_collator=data_collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final model
    logger.info("Saving final model...")
    final_output_dir = os.path.join(qlora_config.output_dir, "final")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    logger.info(f"Training complete! Model saved to {final_output_dir}")

    return trainer


if __name__ == "__main__":
    import argparse
    from .config import PRESETS

    parser = argparse.ArgumentParser(description="Train model with QLoRA")
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        help="Use a preset configuration",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (overrides preset)",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        help="Path to evaluation data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint",
    )

    args = parser.parse_args()

    # Get config
    if args.preset:
        from .config import get_config
        qlora_config = get_config(args.preset)
    else:
        qlora_config = QLoRAConfig()

    # Override with command line arguments
    if args.model:
        qlora_config.model_name = args.model
    if args.output_dir:
        qlora_config.output_dir = args.output_dir

    # Create data config
    data_config = DataConfig(
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
    )

    # Train
    train(
        qlora_config=qlora_config,
        data_config=data_config,
        resume_from_checkpoint=args.resume,
    )
