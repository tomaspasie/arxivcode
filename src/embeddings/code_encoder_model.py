"""
Day 4 Step 1: Load CodeBERT Base Model
Loads microsoft/codebert-base and verifies it works correctly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeEncoder(nn.Module):
    """
    Encoder model for code/paper embeddings using CodeBERT
    Wraps AutoModel to provide consistent interface for training
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        """
        Initialize code encoder

        Args:
            model_name: Hugging Face model identifier
            max_length: Maximum sequence length
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Handle tokenizer padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get embedding dimension
        config = AutoConfig.from_pretrained(model_name)
        self.embedding_dim = config.hidden_size

        self.model.to(self.device)
        logger.info(f"Model loaded. Embedding dim: {self.embedding_dim}")

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for training

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            Embeddings (batch_size, embedding_dim)
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Pool embeddings (mean pooling over sequence length)
        embeddings = outputs.last_hidden_state
        # Mask out padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        embeddings = (embeddings * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(
            min=1e-9
        )

        return embeddings


def load_codebert(model_name: str = "microsoft/codebert-base"):
    """
    Load CodeBERT model and tokenizer

    Args:
        model_name: Hugging Face model identifier

    Returns:
        Tuple of (model, tokenizer, config)
    """
    logger.info("=" * 60)
    logger.info(f"Loading CodeBERT: {model_name}")
    logger.info("=" * 60)

    # Step 1: Load tokenizer
    logger.info("Step 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"✓ Tokenizer loaded. Vocab size: {len(tokenizer)}")

    # Step 2: Load model configuration
    logger.info("Step 2: Loading model configuration...")
    config = AutoConfig.from_pretrained(model_name)
    logger.info(f"✓ Config loaded. Hidden size: {config.hidden_size}")

    # Step 3: Load model
    logger.info("Step 3: Loading model...")
    model = AutoModel.from_pretrained(model_name)
    logger.info("✓ Model loaded")

    # Step 4: Check parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"✓ Total parameters: {total_params:,}")
    logger.info(f"✓ Trainable parameters: {trainable_params:,}")

    # Step 5: Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Step 4: Moving model to {device}...")
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    logger.info(f"✓ Model moved to {device}")

    # Step 6: Verify model works with a test input
    logger.info("Step 5: Verifying model with test input...")
    test_text = "def hello_world(): print('Hello, World!')"
    inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # Get [CLS] token embedding (first token)
        embedding = outputs.last_hidden_state[:, 0, :]
        logger.info(f"✓ Test embedding shape: {embedding.shape}")
        logger.info(f"✓ Model verification successful!")

    logger.info("=" * 60)
    logger.info("CodeBERT loaded and verified successfully!")
    logger.info("=" * 60)

    return model, tokenizer, config


if __name__ == "__main__":
    # Load CodeBERT
    model, tokenizer, config = load_codebert()

    # Print summary
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Model: microsoft/codebert-base")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Number of layers: {config.num_hidden_layers}")
    print(f"Attention heads: {config.num_attention_heads}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)
