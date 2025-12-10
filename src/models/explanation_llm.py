"""LLM-based code explanation generator for research papers"""
from openai import OpenAI
import os
from typing import Optional


class ExplanationLLM:
    """Generates explanations for code snippets in the context of research papers"""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.3):
        """
        Initialize the ExplanationLLM.

        Args:
            model: OpenAI model to use (default: gpt-4o)
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        """
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

        # Improved prompt template with clear structure and constraints
        self.prompt_template = """You are a research code expert explaining implementations from academic papers.

Paper: {paper_title}
User Query: {query}
{paper_context_section}
Code:
```python
{code_snippet}
```

Provide a clear, technical explanation (2-3 sentences) that:
1. Identifies what algorithm/technique this code implements
2. Explains how it relates to the paper's key contributions
3. Highlights important implementation details or design choices

Be specific and avoid generic statements. Focus on the technical aspects that make this implementation noteworthy."""

    def generate_explanation(
        self,
        query: str,
        code_snippet: str,
        paper_title: str,
        paper_context: Optional[str] = None
    ) -> str:
        """
        Generate an explanation for a code snippet in the context of a research paper.

        Args:
            query: The user's question or search query
            code_snippet: The code to explain
            paper_title: Title of the research paper
            paper_context: Optional additional context from the paper (abstract, methods, etc.)

        Returns:
            Generated explanation as a string

        Raises:
            Exception: If the API call fails
        """
        # Format the paper context section if provided
        paper_context_section = ""
        if paper_context:
            paper_context_section = f"Paper Context: {paper_context}\n"

        # Build the prompt
        prompt = self.prompt_template.format(
            query=query,
            code_snippet=code_snippet,
            paper_title=paper_title,
            paper_context_section=paper_context_section
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=250,
                temperature=self.temperature
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            raise Exception(f"Failed to generate explanation: {str(e)}")

    def batch_explain(
        self,
        queries: list[dict]
    ) -> list[dict]:
        """
        Generate explanations for multiple code snippets.

        Args:
            queries: List of dicts with keys: query, code_snippet, paper_title, paper_context

        Returns:
            List of dicts with original query data plus 'explanation' key
        """
        results = []
        for q in queries:
            try:
                explanation = self.generate_explanation(
                    query=q.get("query", ""),
                    code_snippet=q.get("code_snippet", ""),
                    paper_title=q.get("paper_title", ""),
                    paper_context=q.get("paper_context")
                )
                results.append({**q, "explanation": explanation})
            except Exception as e:
                results.append({**q, "explanation": f"Error: {str(e)}"})

        return results


# Example usage
if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)

    # Initialize the LLM
    llm = ExplanationLLM()

    # Test example
    test_code = """
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    similarity = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(similarity, labels)
"""

    explanation = llm.generate_explanation(
        query="contrastive learning",
        code_snippet=test_code,
        paper_title="SimCLR: A Simple Framework for Contrastive Learning of Visual Representations",
        paper_context="This paper introduces SimCLR, a framework for contrastive self-supervised learning using normalized temperature-scaled cross entropy loss (NT-Xent)."
    )

    print("Generated Explanation:")
    print(explanation)
