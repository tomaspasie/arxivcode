"""
Example: Using the ExplanationLLM class directly

This example demonstrates how to use the LLM explanation system
without going through the API.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.explanation_llm import ExplanationLLM


def main():
    # Load environment variables
    load_dotenv()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set in .env file")
        print("Please add your OpenAI API key to .env")
        return

    print("="*70)
    print("ExplanationLLM Example")
    print("="*70)

    # Initialize the LLM
    print("\n1. Initializing ExplanationLLM with GPT-4o...")
    llm = ExplanationLLM(model="gpt-4o", temperature=0.3)
    print("✅ LLM initialized")

    # Example 1: Contrastive Learning
    print("\n" + "="*70)
    print("Example 1: Contrastive Learning (SimCLR)")
    print("="*70)

    code_1 = """def contrastive_loss(z1, z2, temperature=0.5):
    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Compute similarity matrix with temperature scaling
    similarity = torch.matmul(z1, z2.T) / temperature

    # Create labels (diagonal elements are positive pairs)
    labels = torch.arange(z1.size(0)).to(z1.device)

    # NT-Xent loss
    return F.cross_entropy(similarity, labels)"""

    print("\nGenerating explanation...")
    explanation_1 = llm.generate_explanation(
        query="contrastive learning",
        code_snippet=code_1,
        paper_title="SimCLR: A Simple Framework for Contrastive Learning of Visual Representations",
        paper_context="SimCLR uses normalized temperature-scaled cross entropy loss (NT-Xent) for self-supervised learning."
    )

    print(f"\n📝 Explanation:\n{explanation_1}")

    # Example 2: Attention Mechanism
    print("\n" + "="*70)
    print("Example 2: Attention Mechanism (Transformer)")
    print("="*70)

    code_2 = """def scaled_dot_product_attention(query, key, value, mask=None):
    # Compute attention scores
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Apply attention weights to values
    output = torch.matmul(attention_weights, value)
    return output, attention_weights"""

    print("\nGenerating explanation...")
    explanation_2 = llm.generate_explanation(
        query="attention mechanism",
        code_snippet=code_2,
        paper_title="Attention Is All You Need",
        paper_context="The Transformer architecture uses scaled dot-product attention as its core mechanism for sequence modeling."
    )

    print(f"\n📝 Explanation:\n{explanation_2}")

    # Example 3: Batch Processing
    print("\n" + "="*70)
    print("Example 3: Batch Explanation (Multiple Code Snippets)")
    print("="*70)

    queries = [
        {
            "query": "dropout regularization",
            "code_snippet": """class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) > self.p).float()
            return x * mask / (1 - self.p)
        return x""",
            "paper_title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
            "paper_context": "Dropout randomly drops units during training to prevent co-adaptation of neurons."
        },
        {
            "query": "batch normalization",
            "code_snippet": """def batch_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta""",
            "paper_title": "Batch Normalization: Accelerating Deep Network Training",
            "paper_context": "Batch normalization normalizes layer inputs to reduce internal covariate shift."
        }
    ]

    print("\nGenerating batch explanations...")
    results = llm.batch_explain(queries)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['paper_title']}")
        print(f"   Query: {result['query']}")
        print(f"   📝 Explanation: {result['explanation']}")

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("✅ Successfully generated explanations for 4 code snippets")
    print(f"✅ Model used: {llm.model}")
    print(f"✅ Temperature: {llm.temperature}")
    print("\nNext steps:")
    print("  - Try with your own code snippets")
    print("  - Experiment with different temperatures")
    print("  - Use the API for production (see docs/LLM_EXPLANATION_API.md)")


if __name__ == "__main__":
    main()
