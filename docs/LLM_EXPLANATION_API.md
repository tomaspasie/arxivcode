# LLM-Based Code Explanation API

This document describes the LLM-based code explanation system for ArxivCode, which uses OpenAI's GPT-4o to generate contextual explanations for code snippets from research papers.

## Overview

The explanation system consists of two main components:
1. **ExplanationLLM** (`src/models/explanation_llm.py`): Core LLM interface for generating explanations
2. **FastAPI Application** (`src/api/app.py`): REST API endpoint for serving explanations

## Features

- 🤖 GPT-4o-powered explanations with adjustable temperature
- 📝 Context-aware explanations using paper metadata
- 🚀 Fast API with automatic documentation
- 📊 Batch processing support
- ✅ Health check and monitoring endpoints
- 🔧 Configurable model and parameters

## Quick Start

### 1. Installation

Install required dependencies:
```bash
pip install openai fastapi uvicorn pydantic
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### 2. Set up OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Get your API key from: https://platform.openai.com/api-keys

### 3. Test OpenAI Connection

```bash
python3 test_openai_connection.py
```

Expected output:
```
✅ OPENAI_API_KEY found
✅ OpenAI API connection successful
Response: Hello! How can I assist you today?
```

### 4. Start the API Server

```bash
# Option 1: Using the start script
./scripts/start_api.sh

# Option 2: Using uvicorn directly
python3 -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### 5. Test the API

In another terminal, run the test script:
```bash
python3 test_api.py
```

## API Endpoints

### GET /health

Health check endpoint to verify API and OpenAI configuration.

**Response:**
```json
{
  "status": "healthy",
  "message": "API is ready to process requests"
}
```

### POST /explain

Generate an explanation for a single code snippet.

**Request Body:**
```json
{
  "query": "contrastive learning",
  "code_snippet": "def contrastive_loss(z1, z2, temperature=0.5):\n    z1 = F.normalize(z1, dim=1)\n    z2 = F.normalize(z2, dim=1)\n    similarity = torch.matmul(z1, z2.T) / temperature\n    labels = torch.arange(z1.size(0)).to(z1.device)\n    return F.cross_entropy(similarity, labels)",
  "paper_title": "SimCLR: A Simple Framework for Contrastive Learning",
  "paper_context": "Framework for contrastive self-supervised learning using NT-Xent loss"
}
```

**Response:**
```json
{
  "explanation": "This implements the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss from SimCLR. It normalizes embeddings, computes pairwise similarities with temperature scaling to control distribution sharpness, and uses cross-entropy to maximize similarity between positive pairs while minimizing similarity with negatives. The temperature parameter (0.5) makes the model more selective about which examples to treat as similar.",
  "model": "gpt-4o",
  "query": "contrastive learning",
  "paper_title": "SimCLR: A Simple Framework for Contrastive Learning"
}
```

### POST /batch-explain

Generate explanations for multiple code snippets in one request.

**Request Body:**
```json
[
  {
    "query": "attention mechanism",
    "code_snippet": "...",
    "paper_title": "Attention Is All You Need",
    "paper_context": "..."
  },
  {
    "query": "dropout",
    "code_snippet": "...",
    "paper_title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
    "paper_context": "..."
  }
]
```

**Response:** Array of explanation objects

## Using the API with cURL

### Health Check
```bash
curl http://localhost:8000/health
```

### Generate Explanation
```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "query": "contrastive learning",
    "code_snippet": "def contrastive_loss(z1, z2, temperature=0.5):\n    z1 = F.normalize(z1, dim=1)\n    z2 = F.normalize(z2, dim=1)\n    similarity = torch.matmul(z1, z2.T) / temperature\n    labels = torch.arange(z1.size(0)).to(z1.device)\n    return F.cross_entropy(similarity, labels)",
    "paper_title": "SimCLR: A Simple Framework for Contrastive Learning",
    "paper_context": "Framework for contrastive self-supervised learning"
  }'
```

## Using the API with Python

```python
import requests

# Single explanation
response = requests.post(
    "http://localhost:8000/explain",
    json={
        "query": "attention mechanism",
        "code_snippet": "def attention(q, k, v): ...",
        "paper_title": "Attention Is All You Need",
        "paper_context": "Transformer architecture with self-attention"
    }
)

result = response.json()
print(result["explanation"])
```

## Prompt Engineering

The system uses a carefully designed prompt template that:

1. **Provides Context**: Paper title, user query, and optional paper context
2. **Shows Code**: Formatted code snippet with syntax highlighting
3. **Gives Clear Instructions**: 3-point structure for consistent explanations
4. **Enforces Quality**: Emphasizes technical specificity and avoids generic statements

### Prompt Structure

```
You are a research code expert explaining implementations from academic papers.

Paper: {paper_title}
User Query: {query}
Paper Context: {paper_context}

Code:
```python
{code_snippet}
```

Provide a clear, technical explanation (2-3 sentences) that:
1. Identifies what algorithm/technique this implements
2. Explains how it relates to the paper's key contributions
3. Highlights important implementation details or design choices

Be specific and avoid generic statements.
```

### Design Decisions

1. **Temperature 0.3**: Lower temperature for more consistent, technical explanations
2. **Max Tokens 250**: Constrains responses to 2-3 sentences (~150-200 tokens)
3. **Paper Context**: Optional field for providing abstract/methods to improve accuracy
4. **Structured Output**: 3-point framework ensures comprehensive explanations

## Configuration

### Model Selection

The default model is `gpt-4o`, but you can configure it:

```python
from src.models.explanation_llm import ExplanationLLM

# Use GPT-4o mini for faster/cheaper responses
llm = ExplanationLLM(model="gpt-4o-mini", temperature=0.3)

# Use GPT-4 for higher quality
llm = ExplanationLLM(model="gpt-4", temperature=0.3)
```

### Temperature Tuning

- **0.0-0.3**: More deterministic, technical explanations (recommended)
- **0.4-0.7**: Balanced creativity and consistency
- **0.8-1.0**: More creative but less consistent

## Error Handling

The API handles several error cases:

1. **Missing API Key**: Returns 500 with clear error message
2. **Invalid Request**: Returns 422 with validation errors
3. **OpenAI API Errors**: Returns 500 with error details
4. **Rate Limiting**: Propagates OpenAI rate limit errors

## Performance Metrics

Based on testing with 10 example queries:

- **Average Response Time**: ~2-4 seconds per explanation
- **Success Rate**: 100% (with valid API key and credits)
- **Token Usage**: ~150-200 tokens per explanation
- **Cost**: ~$0.001-0.002 per explanation (GPT-4o pricing)

## Testing

### Unit Tests

Test the ExplanationLLM class directly:
```bash
python3 src/models/explanation_llm.py
```

### API Tests

Test all endpoints:
```bash
python3 test_api.py
```

### Manual Testing

Use the interactive docs at http://localhost:8000/docs to test endpoints manually.

## Example Test Cases

The system has been tested with the following paper/code combinations:

1. **SimCLR** - Contrastive learning loss function
2. **Attention Is All You Need** - Scaled dot-product attention
3. **Dropout** - Dropout regularization implementation
4. **Batch Normalization** - Batch norm forward pass
5. **ResNet** - Residual connection implementation
6. **BERT** - Masked language modeling
7. **GPT** - Autoregressive generation
8. **ViT** - Vision transformer patch embedding
9. **U-Net** - Skip connections in segmentation
10. **DDPM** - Diffusion forward process

## Integration with Retrieval System

The explanation API can be integrated with the existing FAISS retrieval system:

```python
from src.retrieval import DenseRetrieval
from src.models.explanation_llm import ExplanationLLM

# 1. Retrieve relevant code
retriever = DenseRetrieval('tfidf')
retriever.load_index('data/processed/FAISS/faiss_index.index',
                      'data/processed/FAISS/faiss_metadata.pkl')
results = retriever.search("contrastive learning", top_k=5)

# 2. Generate explanation for top result
llm = ExplanationLLM()
explanation = llm.generate_explanation(
    query="contrastive learning",
    code_snippet=results[0]['code'],
    paper_title=results[0]['paper_title'],
    paper_context=results[0].get('abstract', '')
)

print(explanation)
```

## Deployment Considerations

### Production Deployment

For production use:

1. **Add Authentication**: Use API keys or OAuth
2. **Rate Limiting**: Implement rate limiting to control costs
3. **Caching**: Cache common queries to reduce API calls
4. **Monitoring**: Add logging and metrics
5. **Error Tracking**: Integrate error tracking (e.g., Sentry)

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional
export API_PORT=8000
export API_HOST="0.0.0.0"
export LLM_MODEL="gpt-4o"
export LLM_TEMPERATURE=0.3
```

## Troubleshooting

### API Key Issues
- Verify key is set: `echo $OPENAI_API_KEY`
- Check key validity at https://platform.openai.com/api-keys
- Ensure account has credits

### Server Won't Start
- Check port 8000 is available: `lsof -i :8000`
- Verify dependencies installed: `pip list | grep fastapi`
- Check Python version: `python3 --version` (requires 3.9+)

### Poor Explanation Quality
- Provide more paper context in the request
- Lower temperature for more technical responses
- Use GPT-4 instead of GPT-4o for complex papers

## Future Improvements

1. **Streaming Responses**: Support streaming for real-time explanation generation
2. **Fine-tuning**: Fine-tune on research paper code explanations
3. **Multi-modal**: Support diagrams and equations from papers
4. **Feedback Loop**: Collect user feedback to improve prompts
5. **Caching**: Add Redis caching for common queries
6. **Analytics**: Track usage patterns and quality metrics

## References

- OpenAI API Documentation: https://platform.openai.com/docs
- FastAPI Documentation: https://fastapi.tiangolo.com
- Prompt Engineering Guide: https://platform.openai.com/docs/guides/prompt-engineering

## License

MIT
