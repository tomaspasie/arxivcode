"""
Configuration file for model training with QLoRA.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA training."""

    # Model configuration
    model_name: str = "mistral-7b"  # or "llama-3-8b"
    max_seq_length: int = 2048
    trust_remote_code: bool = False

    # QLoRA/LoRA configuration
    use_qlora: bool = True
    lora_r: int = 16  # LoRA rank - higher = more parameters but better performance
    lora_alpha: int = 32  # LoRA scaling factor (usually 2x lora_r)
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None  # None = auto-detect
    lora_bias: str = "none"  # "none", "all", or "lora_only"

    # 4-bit quantization configuration
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4"
    bnb_4bit_compute_dtype: str = "bfloat16"  # "float16" or "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch size = 4 * 4 = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    max_grad_norm: float = 0.3

    # Optimizer and scheduler
    optim: str = "paged_adamw_32bit"  # Memory-efficient AdamW
    lr_scheduler_type: str = "cosine"

    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    output_dir: str = "./output/checkpoints"
    logging_dir: str = "./output/logs"

    # Other training arguments
    fp16: bool = False
    bf16: bool = True  # Use bfloat16 for better stability
    gradient_checkpointing: bool = True  # Reduce memory usage
    group_by_length: bool = True  # Group sequences of similar length
    report_to: str = "tensorboard"  # "tensorboard", "wandb", or "none"

    # Generation config (for evaluation)
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True

    def __post_init__(self):
        """Validate and set default values."""
        if self.lora_target_modules is None:
            # Default target modules for LLaMA and Mistral
            self.lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]


@dataclass
class DataConfig:
    """Configuration for dataset."""

    # Dataset paths
    train_data_path: str = "./data/processed/train.json"
    eval_data_path: Optional[str] = "./data/processed/eval.json"
    test_data_path: Optional[str] = "./data/processed/test.json"

    # Data processing
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    preprocessing_num_workers: int = 4

    # Dataset format
    # Expected format: {"text": "...", "label": "..."} or {"input": "...", "output": "..."}
    input_field: str = "text"
    output_field: Optional[str] = None  # For instruction tuning

    # Data split ratios (if single file provided)
    train_split: float = 0.9
    eval_split: float = 0.1


# Preset configurations for common use cases
PRESETS = {
    "paper_understanding": QLoRAConfig(
        model_name="mistral-7b",
        max_seq_length=2048,
        lora_r=16,
        lora_alpha=32,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
    ),
    "code_retrieval": QLoRAConfig(
        model_name="llama-3-8b",
        max_seq_length=4096,  # Longer for code
        lora_r=32,  # Higher rank for complex task
        lora_alpha=64,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
    ),
    "fast_prototype": QLoRAConfig(
        model_name="mistral-7b",
        max_seq_length=1024,
        lora_r=8,  # Lower rank for faster training
        lora_alpha=16,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=3e-4,
    ),
    "high_quality": QLoRAConfig(
        model_name="llama-3-8b",
        max_seq_length=2048,
        lora_r=64,  # Very high rank
        lora_alpha=128,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        warmup_ratio=0.1,
    ),
}


def get_config(preset: Optional[str] = None) -> QLoRAConfig:
    """
    Get configuration, optionally from a preset.

    Args:
        preset: Name of preset configuration (None for default)

    Returns:
        QLoRAConfig object
    """
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
        return PRESETS[preset]
    return QLoRAConfig()


if __name__ == "__main__":
    # Print all preset configurations
    print("Available QLoRA Presets:")
    print("=" * 80)
    for name, config in PRESETS.items():
        print(f"\n{name.upper()}:")
        print(f"  Model: {config.model_name}")
        print(f"  Sequence Length: {config.max_seq_length}")
        print(f"  LoRA Rank: {config.lora_r} (alpha: {config.lora_alpha})")
        print(f"  Epochs: {config.num_train_epochs}")
        print(f"  Batch Size: {config.per_device_train_batch_size} "
              f"(effective: {config.per_device_train_batch_size * config.gradient_accumulation_steps})")
        print(f"  Learning Rate: {config.learning_rate}")
