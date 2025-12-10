"""FastAPI application for code explanation service"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.explanation_llm import ExplanationLLM

# Initialize FastAPI app
app = FastAPI(
    title="ArxivCode Explanation API",
    description="LLM-based code explanation service for research papers",
    version="1.0.0"
)

# Initialize the LLM (singleton)
llm = None


def get_llm():
    """Get or create the LLM instance"""
    global llm
    if llm is None:
        llm = ExplanationLLM()
    return llm


# Request/Response models
class ExplainRequest(BaseModel):
    """Request model for code explanation"""
    query: str = Field(..., description="User's search query or question")
    code_snippet: str = Field(..., description="Code snippet to explain")
    paper_title: str = Field(..., description="Title of the research paper")
    paper_context: Optional[str] = Field(
        None,
        description="Optional context from the paper (abstract, methods, etc.)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "contrastive learning",
                    "code_snippet": "def contrastive_loss(z1, z2, temperature=0.5):\n    z1 = F.normalize(z1, dim=1)\n    z2 = F.normalize(z2, dim=1)\n    similarity = torch.matmul(z1, z2.T) / temperature\n    labels = torch.arange(z1.size(0)).to(z1.device)\n    return F.cross_entropy(similarity, labels)",
                    "paper_title": "SimCLR: A Simple Framework for Contrastive Learning",
                    "paper_context": "Framework for contrastive self-supervised learning using NT-Xent loss"
                }
            ]
        }
    }


class ExplainResponse(BaseModel):
    """Response model for code explanation"""
    explanation: str = Field(..., description="Generated explanation")
    model: str = Field(..., description="Model used for generation")
    query: str = Field(..., description="Original query")
    paper_title: str = Field(..., description="Paper title")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str


# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API information"""
    return {
        "status": "ok",
        "message": "ArxivCode Explanation API is running. Visit /docs for API documentation."
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Verify OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY environment variable not set"
        )

    return {
        "status": "healthy",
        "message": "API is ready to process requests"
    }


@app.post("/explain", response_model=ExplainResponse)
async def explain_code(request: ExplainRequest):
    """
    Generate an explanation for a code snippet in the context of a research paper.

    Args:
        request: ExplainRequest with query, code_snippet, paper_title, and optional paper_context

    Returns:
        ExplainResponse with generated explanation and metadata

    Raises:
        HTTPException: If explanation generation fails
    """
    try:
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY environment variable not set. Please configure the API key."
            )

        # Get LLM instance
        explanation_llm = get_llm()

        # Generate explanation
        explanation = explanation_llm.generate_explanation(
            query=request.query,
            code_snippet=request.code_snippet,
            paper_title=request.paper_title,
            paper_context=request.paper_context
        )

        return {
            "explanation": explanation,
            "model": explanation_llm.model,
            "query": request.query,
            "paper_title": request.paper_title
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating explanation: {str(e)}"
        )


@app.post("/batch-explain")
async def batch_explain_code(requests: list[ExplainRequest]):
    """
    Generate explanations for multiple code snippets.

    Args:
        requests: List of ExplainRequest objects

    Returns:
        List of ExplainResponse objects
    """
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY environment variable not set"
            )

        explanation_llm = get_llm()

        # Convert to format expected by batch_explain
        queries = [
            {
                "query": req.query,
                "code_snippet": req.code_snippet,
                "paper_title": req.paper_title,
                "paper_context": req.paper_context
            }
            for req in requests
        ]

        # Generate explanations
        results = explanation_llm.batch_explain(queries)

        # Format responses
        responses = [
            {
                "explanation": r["explanation"],
                "model": explanation_llm.model,
                "query": r["query"],
                "paper_title": r["paper_title"]
            }
            for r in results
        ]

        return responses

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in batch explanation: {str(e)}"
        )


# Run with: uvicorn src.api.app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
