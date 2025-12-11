from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.dense_retrieval import DenseRetrieval
from src.models.explanation_llm import ExplanationLLM

app = FastAPI(title="ArXivCode API")

# Global instances
retriever = None
explanation_llm = None

@app.on_event("startup")
async def load_resources():
    global retriever, explanation_llm
    print("Loading retrieval system...")
    retriever = DenseRetrieval(
        embedding_model_name="microsoft/codebert-base",
        use_gpu=False
    )
    stats = retriever.get_statistics()
    print(f"Loaded {stats['total_vectors']} code snippets")
    
    print("Loading explanation LLM...")
    explanation_llm = ExplanationLLM(model="gpt-4o", temperature=0.3)
    print("Explanation LLM ready")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranker: bool = False
    hybrid_scoring: bool = True

class ExplainRequest(BaseModel):
    query: str
    code_snippet: str
    paper_title: str
    paper_context: str = ""

@app.get("/")
async def health_check():
    stats = retriever.get_statistics() if retriever else {}
    return {
        "status": "healthy",
        "total_snippets": stats.get('total_vectors', 0)
    }

@app.post("/search")
async def search(request: SearchRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    # Use the DenseRetrieval system
    results = retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        use_reranker=request.use_reranker,
        hybrid_scoring=request.hybrid_scoring
    )
    
    # Format results for frontend
    formatted_results = []
    for r in results:
        meta = r['metadata']
        formatted_results.append({
            "paper_title": meta.get('paper_title', 'Unknown'),
            "code_text": meta.get('code_text', ''),
            "function_name": meta.get('function_name', 'Unknown'),
            "file_path": meta.get('file_path', ''),
            "paper_url": meta.get('paper_url', ''),
            "repo_url": meta.get('repo_url', ''),
            "score": r['score'],
            "rank": r.get('rank', 0)
        })
    
    return {"query": request.query, "results": formatted_results}

@app.post("/explain")
async def explain(request: ExplainRequest):
    if not explanation_llm:
        raise HTTPException(status_code=503, detail="Explanation LLM not initialized")
    
    try:
        explanation = explanation_llm.generate_explanation(
            query=request.query,
            code_snippet=request.code_snippet,
            paper_title=request.paper_title,
            paper_context=request.paper_context
        )
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {str(e)}")

@app.get("/stats")
async def stats():
    if not retriever:
        return {"total_snippets": 0, "embedding_dim": 768, "model": "not loaded"}
    
    stats = retriever.get_statistics()
    return {
        "total_snippets": stats['total_vectors'],
        "embedding_dim": stats['embedding_dim'],
        "model": stats['embedding_model']
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
