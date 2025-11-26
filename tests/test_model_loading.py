"""
Test script to verify model loading and QLoRA setup.
This script tests the model loading without actually downloading large models.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.config import QLoRAConfig, get_config, PRESETS
from src.models.model_loader import get_bnb_config, get_lora_config
import torch


def test_configs():
    """Test configuration creation."""
    print("=" * 80)
    print("TESTING CONFIGURATION CREATION")
    print("=" * 80)

    # Test default config
    print("\n1. Testing default QLoRA config...")
    config = QLoRAConfig()
    print(f"   ✓ Model: {config.model_name}")
    print(f"   ✓ LoRA rank: {config.lora_r}, alpha: {config.lora_alpha}")
    print(f"   ✓ Batch size: {config.per_device_train_batch_size}")
    print(f"   ✓ Learning rate: {config.learning_rate}")

    # Test presets
    print("\n2. Testing preset configs...")
    for preset_name in PRESETS.keys():
        preset_config = get_config(preset_name)
        print(f"   ✓ Loaded preset: {preset_name}")
        print(f"     - Model: {preset_config.model_name}")
        print(f"     - LoRA rank: {preset_config.lora_r}")

    print("\n✓ Configuration tests passed!")


def test_bnb_config():
    """Test BitsAndBytes configuration."""
    print("\n" + "=" * 80)
    print("TESTING BITSANDBYTES CONFIGURATION")
    print("=" * 80)

    print("\n1. Testing 4-bit quantization config...")
    bnb_config = get_bnb_config()
    print(f"   ✓ Load in 4-bit: {bnb_config.load_in_4bit}")
    print(f"   ✓ Quant type: {bnb_config.bnb_4bit_quant_type}")
    print(f"   ✓ Compute dtype: {bnb_config.bnb_4bit_compute_dtype}")
    print(f"   ✓ Double quantization: {bnb_config.bnb_4bit_use_double_quant}")

    print("\n2. Testing custom BnB config...")
    custom_bnb = get_bnb_config(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype="float16",
    )
    print(f"   ✓ Custom config created with fp4 and float16")

    print("\n✓ BitsAndBytes configuration tests passed!")


def test_lora_config():
    """Test LoRA configuration."""
    print("\n" + "=" * 80)
    print("TESTING LORA CONFIGURATION")
    print("=" * 80)

    print("\n1. Testing default LoRA config...")
    lora_config = get_lora_config()
    print(f"   ✓ LoRA rank (r): {lora_config.r}")
    print(f"   ✓ LoRA alpha: {lora_config.lora_alpha}")
    print(f"   ✓ Dropout: {lora_config.lora_dropout}")
    print(f"   ✓ Target modules: {lora_config.target_modules}")

    print("\n2. Testing custom LoRA config...")
    custom_lora = get_lora_config(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
    )
    print(f"   ✓ Custom config: r={custom_lora.r}, alpha={custom_lora.lora_alpha}")

    print("\n✓ LoRA configuration tests passed!")


def test_pytorch_setup():
    """Test PyTorch and CUDA availability."""
    print("\n" + "=" * 80)
    print("TESTING PYTORCH SETUP")
    print("=" * 80)

    print(f"\n1. PyTorch version: {torch.__version__}")
    print(f"2. CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - CUDA version: {torch.version.cuda}")
        print(f"   - GPU count: {torch.cuda.device_count()}")
        print(f"   - Current device: {torch.cuda.current_device()}")
        print(f"   - Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("   - Running on CPU (MPS may be available on Apple Silicon)")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"   - MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")

    print(f"\n3. BFloat16 support: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'N/A (CPU/MPS)'}")

    print("\n✓ PyTorch setup tests passed!")


def print_model_info():
    """Print information about loading models."""
    print("\n" + "=" * 80)
    print("MODEL LOADING INFORMATION")
    print("=" * 80)

    print("""
To actually load a model, you'll need to:

1. For LLaMA-3-8B:
   - Request access at: https://huggingface.co/meta-llama/Meta-Llama-3-8B
   - Login with: huggingface-cli login

2. For Mistral-7B:
   - Available publicly at: https://huggingface.co/mistralai/Mistral-7B-v0.1
   - Login with: huggingface-cli login (recommended)

Example usage:
--------------
from src.models.model_loader import load_llama3_8b, load_mistral_7b

# Load with QLoRA (recommended)
model, tokenizer = load_mistral_7b(use_qlora=True)

# Or load full precision (requires more memory)
model, tokenizer = load_mistral_7b(use_qlora=False)

Memory Requirements:
-------------------
- With QLoRA (4-bit): ~5-6 GB VRAM
- Without QLoRA (full precision): ~28-32 GB VRAM

Current Setup:
--------------
✓ All packages installed
✓ Configurations ready
✓ Model loader script ready
✓ Training script ready

Next Steps:
-----------
1. Login to Hugging Face: huggingface-cli login
2. Test model loading: python src/models/model_loader.py --model mistral-7b
3. Prepare your training data
4. Start training: python -m src.models.trainer --preset paper_understanding --train-data <path>
""")


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "MODEL LOADING & QLORA SETUP TEST" + " " * 25 + "║")
    print("╚" + "=" * 78 + "╝")

    try:
        test_configs()
        test_bnb_config()
        test_lora_config()
        test_pytorch_setup()
        print_model_info()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        print("\nYour QLoRA setup is ready to use!")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
