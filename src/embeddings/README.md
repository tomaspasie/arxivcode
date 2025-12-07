# Code Encoder Training Workflow

This guide explains how to train a contrastive learning model that learns to align paper descriptions with their corresponding code implementations.

## Overview

The workflow trains two encoders (paper encoder and code encoder) using CodeBERT to create embeddings where:
- **Matching pairs** (paper + its code) are close together
- **Non-matching pairs** are far apart

These embeddings are then used for semantic code retrieval: given a paper query, find the most relevant code.

---

## Prerequisites

1. **Data**: You need `data/raw/papers/paper_code_pairs.json` (created by data collection pipeline)
2. **Environment**: Python 3.11 with required packages (see main `requirements.txt`)
3. **GPU** (optional but recommended): Training is faster on GPU

---

## Step-by-Step Workflow

### Step 1: Parse Paper-Code Pairs

**Command:**
```bash
python src/embeddings/paper_code_parser.py
```

**What it does:**
- Reads `data/raw/papers/paper_code_pairs.json`
- Extracts paper text (title + optional abstract) and code text (repo name + description)
- Filters out empty or invalid pairs
- Saves to `data/processed/parsed_pairs.json`

**Output:**
- `data/processed/parsed_pairs.json` - Clean text pairs ready for training

**What you'll see:**
```
Day 4 Step 2: Parsing paper_code_pairs.json
Loading JSON file: data/raw/papers/paper_code_pairs.json
Loaded 249 paper-code pairs
Parsing complete!
  Total pairs processed: 249
  Text pairs created: 249
  Skipped: 0
```

---

### Step 2: Setup DataLoaders

**Command:**
```bash
python src/embeddings/data_loader_setup.py
```

**What it does:**
- Loads `parsed_pairs.json`
- Creates PyTorch Dataset that tokenizes paper and code text
- Splits data into train (80%) and validation (20%)
- Creates DataLoaders with batching, shuffling, etc.
- Saves dataset statistics to `data/processed/dataset_info.json`

**Output:**
- Train/val DataLoaders ready for training
- `data/processed/dataset_info.json` - Dataset statistics

**What you'll see:**
```
Day 4 Step 4: Setting up DataLoaders
Loading dataset from data/processed/parsed_pairs.json
Full dataset size: 249 samples
Splitting dataset: 199 train, 50 val
Creating DataLoaders (batch_size=8, num_workers=0)
✓ DataLoaders created successfully
```

---

### Step 3: Test InfoNCE Loss (Optional)

**Command:**
```bash
python src/embeddings/contrastive_loss.py
```

**What it does:**
- Tests the InfoNCE loss function with dummy data
- Shows how loss behaves with different embedding similarities
- Verifies gradients work correctly

**What you'll see:**
```
Testing InfoNCE Loss Function
Test 1: Random Embeddings (Untrained Model)
   Loss: 1.3863 (expected: ~1.39)
   Diagonal similarities: [0.12, 0.08, 0.15, 0.09]
   Off-diagonal mean: 0.02

Test 2: Partially Similar Embeddings
   Loss: 0.5234
   Gap (diagonal - off_diagonal): 0.89
```

**Why test?** Verifies the loss function works before training.

---

### Step 4: Train the Code Encoder

**Command:**
```bash
python src/embeddings/train_code_encoder.py
```

**What it does:**
1. **Loads models**: Creates two CodeBERT encoders (paper + code)
2. **Loads data**: Gets train/val DataLoaders from Step 2
3. **Training loop** (for each epoch):
   - For each batch:
     - Encodes papers → paper embeddings
     - Encodes codes → code embeddings
     - Computes InfoNCE loss (push positives together, pull negatives apart)
     - Updates model weights via backpropagation
   - Validates on validation set
   - Saves best model if validation loss improved
4. **Saves checkpoints**: Best model + periodic checkpoints

**Output:**
- `checkpoints/code_encoder/best_model.pt` - Best trained model
- `checkpoints/code_encoder/checkpoint_epoch_N.pt` - Periodic checkpoints
- `checkpoints/code_encoder/training_history.json` - Training curves

**What you'll see:**
```
Code Encoder Training
Creating DataLoaders...
Loading encoders: microsoft/codebert-base
Starting training for 3 epochs
Device: cuda
Train batches: 25
Val batches: 7

Epoch 0: Train Loss = 1.2345
Epoch 0: Val Loss = 1.1890
✓ Saved best model (val_loss: 1.1890)

Epoch 1: Train Loss = 0.8765
Epoch 1: Val Loss = 0.8234
✓ Saved best model (val_loss: 0.8234)

Epoch 2: Train Loss = 0.6543
Epoch 2: Val Loss = 0.7123
✓ Saved best model (val_loss: 0.7123)

Training complete!
Best validation loss: 0.7123
```

**Training time:** 
- CPU: ~2-4 hours for 3 epochs
- GPU: ~30-60 minutes for 3 epochs

---

### Step 5: Customize Training (Optional)

**Command with options:**
```bash
python src/embeddings/train_code_encoder.py \
    --batch_size 16 \
    --num_epochs 5 \
    --learning_rate 3e-5 \
    --temperature 0.1 \
    --checkpoint_dir checkpoints/my_experiment
```

**Parameters:**
- `--batch_size`: Larger = faster but more GPU memory (default: 8)
- `--num_epochs`: How many times to go through data (default: 3)
- `--learning_rate`: How fast to learn (default: 2e-5)
- `--temperature`: InfoNCE temperature (lower = harder negatives, default: 0.07)
- `--checkpoint_dir`: Where to save models

---

### Step 6: Resume Training (If Interrupted)

**Command:**
```bash
python src/embeddings/train_code_encoder.py \
    --resume_from checkpoints/code_encoder/checkpoint_epoch_1.pt
```

**What it does:**
- Loads model weights, optimizer state, and training history
- Continues from the saved epoch
- Useful if training was interrupted

---

## Quick Start (All Steps)

Run everything in sequence:

```bash
# Step 1: Parse data
python src/embeddings/paper_code_parser.py

# Step 2: Setup DataLoaders
python src/embeddings/data_loader_setup.py

# Step 3: Train (this takes time!)
python src/embeddings/train_code_encoder.py
```

Or use the bash script (if available):
```bash
./scripts/run_day4.sh  # Runs steps 1-2
python src/embeddings/train_code_encoder.py  # Step 3
```

---

## Understanding the Training Process

### What Happens During Training

1. **Forward Pass:**
   ```
   Paper text → Tokenizer → Paper Encoder → Paper Embedding (768 dims)
   Code text  → Tokenizer → Code Encoder  → Code Embedding (768 dims)
   ```

2. **Loss Computation:**
   - For each paper in batch, compute similarity with all codes
   - Positive pair (paper[i], code[i]) should have HIGH similarity
   - Negative pairs (paper[i], code[j≠i]) should have LOW similarity
   - Loss = how well the model distinguishes positives from negatives

3. **Backward Pass:**
   - Calculate gradients (what to change)
   - Update encoder weights to reduce loss
   - Repeat for next batch

### What "Good" Training Looks Like

- **Initial loss**: ~1.0-1.5 (model can't tell pairs apart)
- **Final loss**: ~0.3-0.7 (model learned to align pairs)
- **Loss decreases**: Should steadily decrease over epochs
- **Val loss tracks train loss**: If val loss increases, you're overfitting

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```bash
# Reduce batch size
python src/embeddings/train_code_encoder.py --batch_size 4
```

### Issue: Training Too Slow

**Solution:**
- Use GPU if available (automatically detected)
- Increase batch size if you have GPU memory
- Reduce number of epochs for testing

### Issue: Loss Not Decreasing

**Possible causes:**
- Learning rate too high → try `--learning_rate 1e-5`
- Learning rate too low → try `--learning_rate 5e-5`
- Data quality issues → check `parsed_pairs.json`

### Issue: Loss is 0.0000

**This is normal in tests** (see `contrastive_loss.py` test output). In real training, loss starts around 1.0-1.5.

---

## Next Steps

After training completes:

1. **Generate Embeddings** (Day 7):
   - Use trained model to create embeddings for all code snippets
   - Save embeddings for retrieval system

2. **Validation Metrics**:
   - Compute Recall@K (does correct code rank in top K?)
   - Compute MRR (Mean Reciprocal Rank)
   - Visual spot checks

3. **Integration**:
   - Connect to retrieval system (FAISS)
   - Use in end-to-end pipeline

---

## File Structure

```
src/embeddings/
├── paper_code_parser.py      # Step 1: Parse JSON → text pairs
├── contrastive_dataset.py     # Step 2: Create PyTorch Dataset
├── data_loader_setup.py      # Step 2: Create DataLoaders
├── code_encoder_model.py     # CodeBERT encoder wrapper
├── contrastive_loss.py        # InfoNCE loss function
└── train_code_encoder.py      # Step 3: Training loop

data/processed/
├── parsed_pairs.json          # Output of Step 1
└── dataset_info.json          # Output of Step 2

checkpoints/code_encoder/
├── best_model.pt              # Best trained model
├── checkpoint_epoch_N.pt      # Periodic checkpoints
└── training_history.json      # Training curves
```

---

## Key Concepts

- **Contrastive Learning**: Learn by comparing (positive vs negative pairs)
- **InfoNCE Loss**: Loss function that pushes positives together, pulls negatives apart
- **Embeddings**: Fixed-size vectors (768 numbers) representing text meaning
- **In-Batch Negatives**: Other pairs in the same batch serve as negatives (efficient!)

---

## Questions?

- Check training logs for detailed progress
- Review `training_history.json` for loss curves
- Test loss function separately if issues arise
- See main `README.md` for project overview

