from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import faiss

# Mock classes for missing pieces
class DummyEncoder:
    def encode(self, query):
        return np.random.rand(768)

class DummyLLM:
    def generate_explanation(self, query, code_snippet, paper_title, paper_context):
        return f"Dummy explanation for {query} in {paper_title}"

app = FastAPI(title="ArXivCode API")

# Load on startup
encoder = None
index = None
metadata = None
llm = None

@app.on_event("startup")
async def load_resources():
    global encoder, index, metadata, llm
    
    encoder = DummyEncoder()
    
    # SWAP SPOT: Replace this dummy data with real CodeBERT embeddings from Nicholas
    # When Nicholas provides the files, replace the np.random.rand(10, 768) with:
    # dummy_embeddings = np.load('data/processed/codebert_embeddings.npy').astype('float32')
    # And load metadata from: metadata = pickle.load(open('data/processed/codebert_metadata.pkl', 'rb'))
    dummy_embeddings = np.random.rand(10, 768).astype('float32')
    faiss.normalize_L2(dummy_embeddings)
    
    # Create FAISS index
    dim = 768
    index = faiss.IndexFlatIP(dim)
    index.add(dummy_embeddings)
    
    dummy_metadata = [
        {
            "paper_title": "Attention Is All You Need",
            "code_text": "def attention(query, key, value):\n    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))\n    attn_weights = F.softmax(scores, dim=-1)\n    return torch.matmul(attn_weights, value)",
            "function_name": "attention",
            "file_path": "models/transformer.py",
            "paper_url": "https://arxiv.org/abs/1706.03762",
            "repo_url": "https://github.com/tensorflow/models"
        },
        {
            "paper_title": "Batch Normalization",
            "code_text": "class BatchNorm1d(nn.Module):\n    def __init__(self, num_features):\n        super().__init__()\n        self.weight = nn.Parameter(torch.ones(num_features))\n        self.bias = nn.Parameter(torch.zeros(num_features))\n    \n    def forward(self, x):\n        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, self.training)",
            "function_name": "BatchNorm1d",
            "file_path": "layers/normalization.py",
            "paper_url": "https://arxiv.org/abs/1502.03167",
            "repo_url": "https://github.com/pytorch/pytorch"
        },
        {
            "paper_title": "Dropout Regularization",
            "code_text": "def dropout(x, p=0.5, training=True):\n    if training:\n        mask = torch.rand_like(x) > p\n        return x * mask / (1 - p)\n    return x",
            "function_name": "dropout",
            "file_path": "regularization/dropout.py",
            "paper_url": "https://arxiv.org/abs/1207.0580",
            "repo_url": "https://github.com/pytorch/pytorch"
        },
        {
            "paper_title": "Adam Optimizer",
            "code_text": "class Adam:\n    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999)):\n        self.params = list(params)\n        self.lr = lr\n        self.betas = betas\n        self.t = 0\n    \n    def step(self):\n        self.t += 1\n        for param in self.params:\n            # Adam update logic here\n            pass",
            "function_name": "Adam",
            "file_path": "optimizers/adam.py",
            "paper_url": "https://arxiv.org/abs/1412.6980",
            "repo_url": "https://github.com/pytorch/pytorch"
        },
        {
            "paper_title": "Cross-Entropy Loss",
            "code_text": "def cross_entropy_loss(logits, targets):\n    log_probs = F.log_softmax(logits, dim=-1)\n    return -log_probs.gather(1, targets.unsqueeze(1)).mean()",
            "function_name": "cross_entropy_loss",
            "file_path": "losses/classification.py",
            "paper_url": "https://arxiv.org/abs/1512.00567",
            "repo_url": "https://github.com/pytorch/pytorch"
        },
        {
            "paper_title": "LoRA Fine-tuning",
            "code_text": "class LoRA(nn.Module):\n    def __init__(self, weight, rank=8):\n        super().__init__()\n        self.A = nn.Parameter(torch.randn(rank, weight.shape[1]))\n        self.B = nn.Parameter(torch.randn(weight.shape[0], rank))\n        self.weight = weight\n    \n    def forward(self, x):\n        return F.linear(x, self.weight + self.B @ self.A)",
            "function_name": "LoRA",
            "file_path": "finetuning/lora.py",
            "paper_url": "https://arxiv.org/abs/2106.09685",
            "repo_url": "https://github.com/microsoft/LoRA"
        },
        {
            "paper_title": "Gradient Descent",
            "code_text": "def gradient_descent_step(params, grads, lr=0.01):\n    for param, grad in zip(params, grads):\n        param.data -= lr * grad",
            "function_name": "gradient_descent_step",
            "file_path": "optimizers/gd.py",
            "paper_url": "https://en.wikipedia.org/wiki/Gradient_descent",
            "repo_url": "https://github.com/scikit-learn/scikit-learn"
        },
        {
            "paper_title": "Self-Attention",
            "code_text": "def self_attention(x, W_q, W_k, W_v):\n    Q = x @ W_q\n    K = x @ W_k\n    V = x @ W_v\n    scores = Q @ K.T / math.sqrt(K.shape[-1])\n    attn = F.softmax(scores, dim=-1)\n    return attn @ V",
            "function_name": "self_attention",
            "file_path": "attention/self_attn.py",
            "paper_url": "https://arxiv.org/abs/1706.03762",
            "repo_url": "https://github.com/huggingface/transformers"
        },
        {
            "paper_title": "Layer Normalization",
            "code_text": "def layer_norm(x, gamma, beta, eps=1e-5):\n    mean = x.mean(dim=-1, keepdim=True)\n    var = x.var(dim=-1, keepdim=True, unbiased=False)\n    return gamma * (x - mean) / torch.sqrt(var + eps) + beta",
            "function_name": "layer_norm",
            "file_path": "normalization/layer_norm.py",
            "paper_url": "https://arxiv.org/abs/1607.06450",
            "repo_url": "https://github.com/pytorch/pytorch"
        },
        {
            "paper_title": "ReLU Activation",
            "code_text": "def relu(x):\n    return torch.maximum(x, torch.zeros_like(x))",
            "function_name": "relu",
            "file_path": "activations/relu.py",
            "paper_url": "https://en.wikipedia.org/wiki/Rectifier_(neural_networks)",
            "repo_url": "https://github.com/pytorch/pytorch"
        }
    ]
    metadata = dummy_metadata
    
    llm = DummyLLM()
    
    print(f"Loaded {index.ntotal} code snippets")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class ExplainRequest(BaseModel):
    query: str
    code_snippet: str
    paper_title: str
    paper_context: str = ""

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "total_snippets": index.ntotal if index else 0
    }

@app.post("/search")
async def search(request: SearchRequest):
    # Encode query with CodeBERT (same model as document embeddings)
    query_embedding = encoder.encode(request.query)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Search FAISS
    scores, indices = index.search(query_embedding, request.top_k)
    
    # Build results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(metadata):
            result = metadata[idx].copy()
            result["score"] = float(scores[0][i])
            results.append(result)
    
    return {"query": request.query, "results": results}

@app.post("/explain")
async def explain(request: ExplainRequest):
    # Placeholder for LLM integration (Pranati will fill in later)
    explanation = llm.generate_explanation(
        query=request.query,
        code_snippet=request.code_snippet,
        paper_title=request.paper_title,
        paper_context=request.paper_context
    )
    return {"explanation": explanation}

@app.get("/stats")
async def stats():
    return {
        "total_snippets": index.ntotal if index else 0,
        "embedding_dim": 768,
        "model": "microsoft/codebert-base"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
