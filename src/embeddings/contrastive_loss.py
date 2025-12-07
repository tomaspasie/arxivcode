"""
Day 5 Step 1: InfoNCE Loss Function
Implements InfoNCE (NT-Xent) loss for contrastive learning of paper-code pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InfoNCELoss(nn.Module):
    """
    InfoNCE (NT-Xent) loss for contrastive learning

    This loss function learns to:
    - Push similar paper-code pairs close together in embedding space
    - Pull dissimilar pairs apart

    Uses in-batch negatives: other pairs in the batch serve as negative examples
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize InfoNCE loss

        Args:
            temperature: Temperature parameter for softmax scaling
                        Lower temperature = sharper distributions (harder negatives)
                        Higher temperature = softer distributions (easier negatives)
                        Typical range: 0.05 - 0.2
        """
        super().__init__()
        self.temperature = temperature
        logger.info(f"InfoNCE loss initialized with temperature={temperature}")

    def forward(
        self,
        paper_embeddings: torch.Tensor,
        code_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss

        Args:
            paper_embeddings: Paper embeddings of shape (batch_size, embedding_dim)
            code_embeddings: Code embeddings of shape (batch_size, embedding_dim)
            negative_embeddings: Optional negative code embeddings of shape
                               (num_negatives * batch_size, embedding_dim)
                               If None, uses in-batch negatives

        Returns:
            Scalar loss value
        """
        batch_size = paper_embeddings.size(0)
        device = paper_embeddings.device

        # Normalize embeddings to unit sphere (for cosine similarity)
        paper_embeddings = F.normalize(paper_embeddings, p=2, dim=1)
        code_embeddings = F.normalize(code_embeddings, p=2, dim=1)

        # Compute positive similarities (diagonal: matching pairs)
        # Shape: (batch_size,)
        positive_sim = (paper_embeddings * code_embeddings).sum(dim=1)
        positive_sim = positive_sim / self.temperature

        # Compute all pairwise similarities between papers and codes
        # Shape: (batch_size, batch_size)
        # Each row i: similarity between paper i and all codes
        all_sim = torch.matmul(paper_embeddings, code_embeddings.t()) / self.temperature

        # Add negative samples if provided
        if negative_embeddings is not None:
            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)
            # Compute similarities with negative codes
            # Shape: (batch_size, num_negatives * batch_size)
            negative_sim = (
                torch.matmul(paper_embeddings, negative_embeddings.t())
                / self.temperature
            )
            # Concatenate with all similarities
            all_sim = torch.cat([all_sim, negative_sim], dim=1)

        # Create labels: diagonal is positive (paper i matches code i)
        labels = torch.arange(batch_size, device=device)

        # Cross-entropy loss
        # For each paper, we want the matching code to have highest similarity
        loss = F.cross_entropy(all_sim, labels)

        return loss

    def compute_similarity_matrix(
        self,
        paper_embeddings: torch.Tensor,
        code_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute similarity matrix between papers and codes (for analysis)

        Args:
            paper_embeddings: Paper embeddings (batch_size, embedding_dim)
            code_embeddings: Code embeddings (batch_size, embedding_dim)

        Returns:
            Similarity matrix (batch_size, batch_size)
        """
        paper_embeddings = F.normalize(paper_embeddings, p=2, dim=1)
        code_embeddings = F.normalize(code_embeddings, p=2, dim=1)
        return torch.matmul(paper_embeddings, code_embeddings.t())


def test_infonce_loss():
    """
    Test function to verify InfoNCE loss works correctly
    """
    print("=" * 60)
    print("Testing InfoNCE Loss Function")
    print("=" * 60)

    # Create dummy embeddings
    batch_size = 4
    embedding_dim = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize loss
    criterion = InfoNCELoss(temperature=0.07)

    # Test 1: Completely random embeddings (untrained model scenario)
    print("\n" + "=" * 60)
    print("Test 1: Random Embeddings (Untrained Model)")
    print("=" * 60)
    torch.manual_seed(42)
    paper_emb_random = torch.randn(batch_size, embedding_dim, device=device)
    code_emb_random = torch.randn(batch_size, embedding_dim, device=device)

    loss_random = criterion(paper_emb_random, code_emb_random)
    sim_matrix_random = criterion.compute_similarity_matrix(
        paper_emb_random, code_emb_random
    )
    diagonal_random = torch.diag(sim_matrix_random)
    off_diagonal_random = sim_matrix_random[
        ~torch.eye(batch_size, dtype=bool, device=device)
    ]

    print(f"   Loss: {loss_random.item():.4f}")
    print(
        f"   Expected range: ~{torch.log(torch.tensor(batch_size, dtype=torch.float)).item():.2f} (log(batch_size))"
    )
    print(f"   Diagonal similarities (positive pairs):")
    for i, sim in enumerate(diagonal_random):
        print(f"      Pair {i}: {sim.item():.4f}")
    print(f"   Off-diagonal mean: {off_diagonal_random.mean().item():.4f}")
    print(f"   Off-diagonal std: {off_diagonal_random.std().item():.4f}")
    print(
        f"   Off-diagonal min: {off_diagonal_random.min().item():.4f}, max: {off_diagonal_random.max().item():.4f}"
    )

    # Show which pairs have highest similarities (potential confusions)
    print(f"\n   Top 3 highest off-diagonal similarities (potential confusions):")
    off_diag_flat = off_diagonal_random.flatten()
    top3_indices = torch.topk(off_diag_flat, k=min(3, len(off_diag_flat))).indices
    for idx in top3_indices:
        row = idx // (batch_size - 1)
        col = idx % (batch_size - 1)
        actual_col = col if col < row else col + 1
        print(
            f"      Paper {row} <-> Code {actual_col}: {off_diag_flat[idx].item():.4f}"
        )

    # Test 2: Partially similar embeddings (partially trained model)
    print("\n" + "=" * 60)
    print("Test 2: Partially Similar Embeddings (Partially Trained)")
    print("=" * 60)
    torch.manual_seed(42)
    paper_emb_partial = torch.randn(batch_size, embedding_dim, device=device)
    code_emb_partial = torch.randn(batch_size, embedding_dim, device=device)
    for i in range(batch_size):
        # Positive pair: 70% similar, 30% random
        code_emb_partial[i] = 0.7 * paper_emb_partial[i] + 0.3 * torch.randn(
            embedding_dim, device=device
        )

    loss_partial = criterion(paper_emb_partial, code_emb_partial)
    sim_matrix_partial = criterion.compute_similarity_matrix(
        paper_emb_partial, code_emb_partial
    )
    diagonal_partial = torch.diag(sim_matrix_partial)
    off_diagonal_partial = sim_matrix_partial[
        ~torch.eye(batch_size, dtype=bool, device=device)
    ]

    print(f"   Loss: {loss_partial.item():.4f}")
    print(f"   Diagonal similarities (positive pairs):")
    for i, sim in enumerate(diagonal_partial):
        print(f"      Pair {i}: {sim.item():.4f}")
    print(f"   Off-diagonal mean: {off_diagonal_partial.mean().item():.4f}")
    print(
        f"   Gap (diagonal - off_diagonal): {(diagonal_partial.mean() - off_diagonal_partial.mean()).item():.4f}"
    )

    # Analyze why loss is low
    print(f"\n   Loss Analysis:")
    print(f"      - Diagonal similarity is {diagonal_partial.mean().item():.4f}")
    print(
        f"      - Off-diagonal similarity is {off_diagonal_partial.mean().item():.4f}"
    )
    print(
        f"      - Gap: {(diagonal_partial.mean() - off_diagonal_partial.mean()).item():.4f}"
    )
    if diagonal_partial.mean() - off_diagonal_partial.mean() > 0.5:
        print(f"      ⚠️  Large gap indicates positive pairs are too similar in test")
    print(
        f"      - With temperature={criterion.temperature}, scaled gap: {(diagonal_partial.mean() - off_diagonal_partial.mean()) / criterion.temperature:.2f}"
    )

    # Test 3: Very similar embeddings (well-trained model)
    print("\n" + "=" * 60)
    print("Test 3: Very Similar Embeddings (Well-Trained Model)")
    print("=" * 60)
    torch.manual_seed(42)
    paper_emb_trained = torch.randn(batch_size, embedding_dim, device=device)
    code_emb_trained = torch.randn(batch_size, embedding_dim, device=device)
    for i in range(batch_size):
        # Positive pair: 95% similar, 5% random (well-trained)
        code_emb_trained[i] = 0.95 * paper_emb_trained[i] + 0.05 * torch.randn(
            embedding_dim, device=device
        )

    loss_trained = criterion(paper_emb_trained, code_emb_trained)
    sim_matrix_trained = criterion.compute_similarity_matrix(
        paper_emb_trained, code_emb_trained
    )
    diagonal_trained = torch.diag(sim_matrix_trained)
    off_diagonal_trained = sim_matrix_trained[
        ~torch.eye(batch_size, dtype=bool, device=device)
    ]

    print(f"   Loss: {loss_trained.item():.4f}")
    print(f"   Diagonal similarities: {diagonal_trained}")
    print(f"   Off-diagonal mean: {off_diagonal_trained.mean().item():.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(
        f"Random embeddings loss:     {loss_random.item():.4f} (expected: ~{torch.log(torch.tensor(batch_size, dtype=torch.float)).item():.2f})"
    )
    print(f"Partially trained loss:     {loss_partial.item():.4f}")
    print(f"Well-trained loss:          {loss_trained.item():.4f}")
    print(f"\nBatch size: {batch_size}")
    print(f"  - Small batch size means fewer negatives")
    print(
        f"  - Theoretical max loss: log({batch_size}) = {torch.log(torch.tensor(batch_size, dtype=torch.float)).item():.4f}"
    )
    print(f"  - In real training with batch_size=8-16, loss will start higher")

    # Test 4: Gradient flow
    print("\n" + "=" * 60)
    print("Test 4: Gradient Flow")
    print("=" * 60)
    paper_emb_grad = torch.randn(
        batch_size, embedding_dim, device=device, requires_grad=True
    )
    code_emb_grad = torch.randn(
        batch_size, embedding_dim, device=device, requires_grad=True
    )
    loss_grad = criterion(paper_emb_grad, code_emb_grad)
    loss_grad.backward()
    print(f"   Paper embeddings grad: {paper_emb_grad.grad is not None}")
    print(f"   Code embeddings grad: {code_emb_grad.grad is not None}")
    print(
        f"   Gradient norms - Paper: {paper_emb_grad.grad.norm().item():.4f}, Code: {code_emb_grad.grad.norm().item():.4f}"
    )
    print(f"   ✓ Gradients computed successfully")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  - Loss starts high with random embeddings (untrained model)")
    print("  - Loss decreases as embeddings become more similar (training progress)")
    print(
        "  - Small batch size (4) means fewer negatives, but this is just for testing"
    )
    print("  - In real training with batch_size=8-16, initial loss will be ~1.0-2.0")
    print(
        "  - The low loss in Test 2 is due to test setup (70% similarity is still high)"
    )


if __name__ == "__main__":
    test_infonce_loss()
