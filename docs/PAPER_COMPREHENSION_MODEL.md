# Paper Comprehension Model Setup

Guide for setting up and training the paper comprehension model that enables question-answering on collected research papers.

## Overview

The paper comprehension model allows users to:
- Ask questions about collected papers (249 papers from your dataset)
- Get answers based on paper content, methodology, and code
- Understand relationships between papers and their implementations

## Model Architecture

We use **LLaMA-3-8B** or **Mistral-7B** with QLoRA for:
- Memory efficiency (5-6GB VRAM vs 28-32GB full precision)
- Fast fine-tuning on paper-specific QA tasks
- Maintains 99% of full model performance

## Quick Start

### 1. Install Dependencies

All required packages are in `requirements.txt`:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

Includes: torch, transformers, peft, bitsandbytes, accelerate, datasets

### 2. Authenticate with Hugging Face

```bash
pip install huggingface_hub
huggingface-cli login
```

For LLaMA-3, request access: https://huggingface.co/meta-llama/Meta-Llama-3-8B

### 3. Test Model Loading

```bash
python examples/test_model_loading.py
```

This verifies your setup without downloading models.

### 4. Load a Model

```python
from src.models.model_loader import load_mistral_7b, load_llama3_8b

# Load with QLoRA (recommended)
model, tokenizer = load_mistral_7b(use_qlora=True)

# Test generation
prompt = "Explain the BERT model architecture:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Configuration Presets

Located in `src/models/config.py`:

| Preset | Model | LoRA Rank | Context | Best For |
|--------|-------|-----------|---------|----------|
| `paper_understanding` | Mistral-7B | 16 | 2048 | Default for paper QA |
| `code_retrieval` | LLaMA-3-8B | 32 | 4096 | Code + paper understanding |
| `fast_prototype` | Mistral-7B | 8 | 1024 | Quick testing |
| `high_quality` | LLaMA-3-8B | 64 | 2048 | Production deployment |

**Recommended**: Start with `paper_understanding` preset.

## Training Data Format

### Instruction-Tuning Format (Recommended)

Create `data/processed/train.json`:
```json
[
    {
        "input": "What is the main contribution of the BERT paper?",
        "output": "BERT introduces bidirectional pre-training for language understanding by using masked language modeling..."
    },
    {
        "input": "How does the Transformer architecture work?",
        "output": "The Transformer uses self-attention mechanisms to process sequences in parallel..."
    }
]
```

### Data Preparation Steps

1. Extract paper abstracts/summaries from your collected data
2. Generate QA pairs from paper content
3. Split into train/eval sets (90/10)

Example script structure:
```python
import json
# Load your paper collection
papers = json.load(open('data/raw/papers/paper_code_pairs.json'))

# Create QA pairs
qa_pairs = []
for paper in papers:
    # Generate questions about the paper
    qa_pairs.append({
        "input": f"Summarize the {paper['title']} paper",
        "output": paper['abstract']
    })
    # Add more QA variations...

# Save
json.dump(qa_pairs, open('data/processed/train.json', 'w'))
```

## Training

### Using Preset Configuration

```bash
python -m src.models.trainer \
    --preset paper_understanding \
    --train-data data/processed/train.json \
    --eval-data data/processed/eval.json \
    --output-dir output/paper_qa_model
```

### Custom Configuration

```bash
python -m src.models.trainer \
    --model llama-3-8b \
    --train-data data/processed/train.json \
    --eval-data data/processed/eval.json \
    --output-dir output/custom_model
```

### Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir output/logs

# View in browser at http://localhost:6006
```

### Resume from Checkpoint

```bash
python -m src.models.trainer \
    --preset paper_understanding \
    --train-data data/processed/train.json \
    --resume output/paper_qa_model/checkpoint-500
```

## Memory Requirements

| Configuration | VRAM | LoRA Rank | Training Time (per epoch) |
|---------------|------|-----------|---------------------------|
| Fast | 4-5 GB | r=8 | ~2-3 hours |
| Default | 5-6 GB | r=16 | ~3-4 hours |
| Advanced | 6-7 GB | r=32 | ~4-5 hours |
| Maximum | 8-10 GB | r=64 | ~5-6 hours |

**Your System**: Apple Silicon with MPS - expect ~6-8GB with default settings.

## QLoRA Configuration Details

### Key Parameters

```python
# LoRA Configuration
lora_r = 16                    # Rank: higher = more capacity
lora_alpha = 32                # Scaling: usually 2x rank
lora_dropout = 0.05            # Regularization
target_modules = [             # Which transformer layers to adapt
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# 4-bit Quantization
load_in_4bit = True            # Enable QLoRA
bnb_4bit_quant_type = "nf4"   # NormalFloat4 (recommended)
bnb_4bit_compute_dtype = "bfloat16"
bnb_4bit_use_double_quant = True

# Training
batch_size = 4                 # Per-device
gradient_accumulation = 4      # Effective batch = 16
learning_rate = 2e-4           # Higher for LoRA
num_epochs = 3
max_seq_length = 2048
```

View all presets: `python -m src.models.config`

## Code Structure

```
src/models/
├── model_loader.py     # Load LLaMA/Mistral with QLoRA
├── config.py          # Training configurations & presets
├── trainer.py         # Training pipeline
└── __init__.py

examples/
└── test_model_loading.py    # Verify setup

docs/
└── PAPER_COMPREHENSION_MODEL.md  # This file
```

## API Usage (After Training)

```python
from src.models.model_loader import load_model_and_tokenizer
from peft import PeftModel
import torch

# Load base model with QLoRA
base_model, tokenizer = load_model_and_tokenizer(
    "mistral-7b",
    use_qlora=True
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(
    base_model,
    "output/paper_qa_model/final"
)

# Ask questions
def ask_question(question: str):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Use the model
answer = ask_question("What is the main contribution of BERT?")
print(answer)
```

## Supported Models

| Model | HuggingFace Path | Access | Size |
|-------|-----------------|--------|------|
| Mistral-7B | `mistralai/Mistral-7B-v0.1` | Open | 7B |
| Mistral-7B-Instruct | `mistralai/Mistral-7B-Instruct-v0.1` | Open | 7B |
| LLaMA-3-8B | `meta-llama/Meta-Llama-3-8B` | Gated | 8B |
| LLaMA-3-8B-Instruct | `meta-llama/Meta-Llama-3-8B-Instruct` | Gated | 8B |

**Recommendation**: Start with Mistral-7B (no access request needed).

## Troubleshooting

### Out of Memory
- Reduce batch size: edit config or use `per_device_train_batch_size=2`
- Lower LoRA rank: use `fast_prototype` preset
- Reduce sequence length
- Ensure gradient checkpointing is enabled (default)

### Slow Training
- Increase batch size if memory allows
- Use smaller LoRA rank for faster iterations
- Reduce max_seq_length for shorter papers

### Model Access Errors
```bash
# Login again
huggingface-cli login

# For LLaMA, request access at:
# https://huggingface.co/meta-llama/Meta-Llama-3-8B
```

### Import Errors
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Next Steps

1. **Prepare training data**: Create QA pairs from your 249 collected papers
2. **Choose model**: Mistral-7B (easier) or LLaMA-3-8B (better quality)
3. **Start training**: Use `paper_understanding` preset
4. **Evaluate**: Test on held-out papers
5. **Deploy**: Integrate into retrieval system

## Resources

- QLoRA Paper: https://arxiv.org/abs/2305.14314
- PEFT Documentation: https://huggingface.co/docs/peft
- Transformers: https://huggingface.co/docs/transformers
- Your paper collection: `data/raw/papers/paper_code_pairs.json`
