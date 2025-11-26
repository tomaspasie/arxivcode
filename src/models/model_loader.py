"""
Model loader for LLaMA-3-8B and Mistral-7B with QLoRA support.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Supported models configuration
SUPPORTED_MODELS = {
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",
}


def get_bnb_config(
    load_in_4bit: bool = True,
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: str = "bfloat16",
    bnb_4bit_use_double_quant: bool = True,
) -> BitsAndBytesConfig:
    """
    Create BitsAndBytes quantization configuration for QLoRA.

    Args:
        load_in_4bit: Whether to load model in 4-bit precision
        bnb_4bit_quant_type: Quantization type (nf4 or fp4)
        bnb_4bit_compute_dtype: Compute dtype for 4-bit base models
        bnb_4bit_use_double_quant: Whether to use nested quantization

    Returns:
        BitsAndBytesConfig object
    """
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )

    logger.info(f"BitsAndBytes config: 4bit={load_in_4bit}, type={bnb_4bit_quant_type}, "
                f"dtype={bnb_4bit_compute_dtype}, double_quant={bnb_4bit_use_double_quant}")

    return bnb_config


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: Optional[list] = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> LoraConfig:
    """
    Create LoRA configuration for parameter-efficient fine-tuning.

    Args:
        r: LoRA rank (attention dimension)
        lora_alpha: LoRA scaling factor
        target_modules: Modules to apply LoRA to (None = auto-detect)
        lora_dropout: Dropout probability for LoRA layers
        bias: Bias training strategy ("none", "all", "lora_only")
        task_type: Task type for PEFT

    Returns:
        LoraConfig object
    """
    # Default target modules for LLaMA and Mistral models
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )

    logger.info(f"LoRA config: r={r}, alpha={lora_alpha}, dropout={lora_dropout}, "
                f"target_modules={target_modules}")

    return lora_config


def load_model_and_tokenizer(
    model_name: str,
    use_qlora: bool = True,
    bnb_config: Optional[BitsAndBytesConfig] = None,
    lora_config: Optional[LoraConfig] = None,
    device_map: str = "auto",
    trust_remote_code: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer with optional QLoRA configuration.

    Args:
        model_name: Model identifier (key from SUPPORTED_MODELS or HF model path)
        use_qlora: Whether to apply QLoRA (4-bit quantization + LoRA)
        bnb_config: BitsAndBytes configuration (created if None and use_qlora=True)
        lora_config: LoRA configuration (created if None and use_qlora=True)
        device_map: Device mapping strategy
        trust_remote_code: Whether to trust remote code

    Returns:
        Tuple of (model, tokenizer)
    """
    # Resolve model path
    model_path = SUPPORTED_MODELS.get(model_name, model_name)
    logger.info(f"Loading model: {model_path}")

    # Create default configs if not provided
    if use_qlora:
        if bnb_config is None:
            bnb_config = get_bnb_config()
        if lora_config is None:
            lora_config = get_lora_config()

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        padding_side="right",  # Required for training
    )

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config if use_qlora else None,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16 if not use_qlora else None,
    )

    # Apply QLoRA if requested
    if use_qlora:
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)

        logger.info("Applying LoRA...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    logger.info("Model and tokenizer loaded successfully!")
    return model, tokenizer


def load_llama3_8b(use_qlora: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Convenience function to load LLaMA-3-8B."""
    return load_model_and_tokenizer("llama-3-8b", use_qlora=use_qlora)


def load_mistral_7b(use_qlora: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Convenience function to load Mistral-7B."""
    return load_model_and_tokenizer("mistral-7b", use_qlora=use_qlora)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Load model with QLoRA")
    parser.add_argument(
        "--model",
        type=str,
        default="mistral-7b",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model to load",
    )
    parser.add_argument(
        "--no-qlora",
        action="store_true",
        help="Disable QLoRA (load full precision model)",
    )
    parser.add_argument(
        "--test-generation",
        action="store_true",
        help="Test model generation with a sample prompt",
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        use_qlora=not args.no_qlora,
    )

    # Test generation if requested
    if args.test_generation:
        logger.info("\nTesting generation...")
        prompt = "Explain what a neural network is:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"\nPrompt: {prompt}")
        logger.info(f"Generated: {generated_text}")
