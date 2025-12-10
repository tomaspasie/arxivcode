"""
Day 5 Step 2: Training Loop for Code Encoder
Implements contrastive learning training with InfoNCE loss
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict
import logging
import sys
from tqdm import tqdm
import json
import os

# Handle imports
sys.path.insert(0, str(Path(__file__).parent))
from code_encoder_model import CodeEncoder
from contrastive_loss import InfoNCELoss
from data_loader_setup import create_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContrastiveTrainer:
    """
    Trainer for contrastive learning of paper-code embeddings
    """

    def __init__(
        self,
        paper_encoder: CodeEncoder,
        code_encoder: CodeEncoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 2e-5,
        temperature: float = 0.07,
        device: Optional[str] = None,
        checkpoint_dir: str = "checkpoints/code_encoder",
    ):
        """
        Initialize trainer

        Args:
            paper_encoder: Encoder for paper text
            code_encoder: Encoder for code text
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
            learning_rate: Learning rate for optimizer
            temperature: Temperature for InfoNCE loss
            device: Device to run on ('cuda', 'cpu', or None for auto)
            checkpoint_dir: Directory to save checkpoints
        """
        self.paper_encoder = paper_encoder
        self.code_encoder = code_encoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Move models to device
        self.paper_encoder.to(self.device)
        self.code_encoder.to(self.device)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.paper_encoder.parameters())
            + list(self.code_encoder.parameters()),
            lr=learning_rate,
            weight_decay=0.01,
        )

        # Loss function
        self.criterion = InfoNCELoss(temperature=temperature)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

        # Clear GPU cache before training
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self._log_memory_usage("After initialization")
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Learning rate: {learning_rate}, Temperature: {temperature}")
    
    def _log_memory_usage(self, stage: str = ""):
        """Log current GPU memory usage"""
        if self.device == "cuda":
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            max_allocated = torch.cuda.max_memory_allocated(0) / 1e9
            logger.info(f"GPU Memory {stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Max={max_allocated:.2f}GB")

    def train_epoch(self) -> float:
        """Train for one epoch"""
        # Clear cache at start of epoch
        if self.device == "cuda":
            torch.cuda.empty_cache()
            self._log_memory_usage("Start of epoch")
        
        self.paper_encoder.train()
        self.code_encoder.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            leave=False,
        )
        for batch in pbar:
            # Move to device
            paper_input_ids = batch["paper_input_ids"].to(self.device)
            paper_attention_mask = batch["paper_attention_mask"].to(self.device)
            code_input_ids = batch["code_input_ids"].to(self.device)
            code_attention_mask = batch["code_attention_mask"].to(self.device)

            # Forward pass through encoders
            paper_embeddings = self.paper_encoder(paper_input_ids, paper_attention_mask)
            code_embeddings = self.code_encoder(code_input_ids, code_attention_mask)

            # Compute loss (uses in-batch negatives)
            loss = self.criterion(paper_embeddings, code_embeddings)
            
            # Store loss value before deleting tensors
            loss_value = loss.item()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Delete intermediate tensors to free memory
            del paper_embeddings, code_embeddings

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.paper_encoder.parameters())
                + list(self.code_encoder.parameters()),
                max_norm=1.0,
            )

            self.optimizer.step()

            # Delete batch tensors and loss, then clear cache to free memory
            del paper_input_ids, paper_attention_mask, code_input_ids, code_attention_mask, loss
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Update metrics
            total_loss += loss_value
            num_batches += 1
            self.global_step += 1

            # Update progress bar (log memory every 100 batches)
            if num_batches % 100 == 0 and self.device == "cuda":
                self._log_memory_usage(f"Batch {num_batches}")
            
            pbar.set_postfix(
                {"loss": loss_value, "avg_loss": total_loss / num_batches}
            )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self) -> float:
        """Validate on validation set"""
        if self.val_loader is None:
            return 0.0

        self.paper_encoder.eval()
        self.code_encoder.eval()

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for batch in pbar:
                paper_input_ids = batch["paper_input_ids"].to(self.device)
                paper_attention_mask = batch["paper_attention_mask"].to(self.device)
                code_input_ids = batch["code_input_ids"].to(self.device)
                code_attention_mask = batch["code_attention_mask"].to(self.device)

                paper_embeddings = self.paper_encoder(
                    paper_input_ids, paper_attention_mask
                )
                code_embeddings = self.code_encoder(code_input_ids, code_attention_mask)

                loss = self.criterion(paper_embeddings, code_embeddings)
                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({"val_loss": loss.item()})
                
                # Clear cache during validation too
                if self.device == "cuda":
                    torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def train(
        self,
        num_epochs: int,
        save_every: int = 1,
        save_best: bool = True,
    ):
        """
        Train for multiple epochs

        Args:
            num_epochs: Number of epochs
            save_every: Save checkpoint every N epochs
            save_best: Save checkpoint when validation loss improves
        """
        logger.info("=" * 60)
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            logger.info(f"Val batches: {len(self.val_loader)}")
        logger.info("=" * 60)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                logger.info(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")

                # Save best model
                if save_best and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
                    logger.info(f"✓ Saved best model (val_loss: {val_loss:.4f})")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch, is_best=False)

        logger.info("=" * 60)
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 60)

        # Save training history
        self.save_training_history()

        return self.train_losses, self.val_losses

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "paper_encoder_state_dict": self.paper_encoder.state_dict(),
            "code_encoder_state_dict": self.code_encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }

        if is_best:
            checkpoint_path = self.checkpoint_dir / "best_model.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.paper_encoder.load_state_dict(checkpoint["paper_encoder_state_dict"])
        self.code_encoder.load_state_dict(checkpoint["code_encoder_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def save_training_history(self):
        """Save training history to JSON"""
        history = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "num_epochs": len(self.train_losses),
        }

        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Saved training history to {history_path}")


def train_code_encoder(
    json_path: str = "data/processed/parsed_pairs.json",
    model_name: str = "microsoft/codebert-base",
    batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    temperature: float = 0.07,
    train_split: float = 0.8,
    checkpoint_dir: str = "checkpoints/code_encoder",
    resume_from: Optional[str] = None,
    max_length: int = 512,
):
    """
    Main training function

    Args:
        json_path: Path to parsed_pairs.json
        model_name: Model name for encoders
        batch_size: Batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        temperature: Temperature for InfoNCE loss
        train_split: Train/val split ratio
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
    """
    logger.info("=" * 60)
    logger.info("Code Encoder Training")
    logger.info("=" * 60)
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        logger.info(f"GPU Memory (before loading): Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

    # Create DataLoaders
    logger.info("Creating DataLoaders...")
    train_loader, val_loader = create_data_loaders(
        json_path=json_path,
        model_name=model_name,
        batch_size=batch_size,
        train_split=train_split,
        max_length=max_length,
    )

    # Load encoders (same model for paper and code)
    logger.info(f"Loading encoders: {model_name}")
    paper_encoder = CodeEncoder(model_name=model_name)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        logger.info(f"GPU Memory (after paper encoder): Allocated={allocated:.2f}GB")
    
    code_encoder = CodeEncoder(model_name=model_name)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        logger.info(f"GPU Memory (after code encoder): Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

    # Create trainer
    trainer = ContrastiveTrainer(
        paper_encoder=paper_encoder,
        code_encoder=code_encoder,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        temperature=temperature,
        checkpoint_dir=checkpoint_dir,
    )

    # Resume from checkpoint if specified
    if resume_from:
        trainer.load_checkpoint(Path(resume_from))

    # Train
    train_losses, val_losses = trainer.train(
        num_epochs=num_epochs,
        save_every=1,
        save_best=True,
    )

    logger.info("Training complete!")
    return trainer


if __name__ == "__main__":
    """
    Main training script
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Train code encoder with contrastive learning"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="data/processed/parsed_pairs.json",
        help="Path to parsed_pairs.json",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/codebert-base",
        help="Model name for encoders",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (default: 4 for Tesla P4 GPU with 7GB memory, use 2 if OOM)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512, reduce to 256 if OOM)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for InfoNCE loss",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints/code_encoder",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    # Train
    trainer = train_code_encoder(
        json_path=args.json_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        max_length=args.max_length,
    )

    print("\n" + "=" * 60)
    print("Training Complete! ✓")
    print("=" * 60)
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Best model: {args.checkpoint_dir}/best_model.pt")
    print("\nNext step: Generate embeddings (Day 7)")
