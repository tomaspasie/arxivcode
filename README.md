# ArxivCode!

Machine learning project for paper-code understanding and retrieval. Enables question-answering on research papers using fine-tuned LLMs.

## Quick Setup

### 1. Environment Setup

```bash
# Create virtual environment (Python 3.11)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure GitHub Token

```bash
cp .env.example .env
# Edit .env and add your GitHub token
# Get token at: https://github.com/settings/tokens
```

### 3. Collect Papers

```bash
# Run complete collection pipeline
./scripts/collect_papers.sh
```

Output: `data/raw/papers/paper_code_pairs.json` (249 papers currently)

### 4. Setup Paper Comprehension Model

```bash
# Authenticate with Hugging Face
huggingface-cli login

# Test model setup
python tests/test_model_loading.py
```

See [Paper Comprehension Model Guide](docs/PAPER_COMPREHENSION_MODEL.md) for training details.

## Retrieval System Testing

The retrieval system uses FAISS for efficient similarity search across paper-code pairs. Test the system by running these commands in order:

```bash
# 1. Test imports
python -c "from src.retrieval import FAISSIndexManager, DenseRetrieval; print('✅ Imports work!')"

# 2. Test FAISS manager
python -m src.retrieval.faiss_index

# 3. Build index with real data
python -m src.retrieval.build_index --input data/raw/papers/paper_code_pairs.json

# 4. Test retrieval
python -m src.retrieval.test_retrieval
```

**Index Storage**: `data/processed/FAISS/`
- `faiss_index.index` - Vector similarity index
- `faiss_index_vectorizer.pkl` - TF-IDF vectorizer (for CPU-stable embeddings)
- `faiss_metadata.pkl` - Metadata for retrieved results

**Features**:
- TF-IDF embeddings for stable, CPU-friendly similarity search
- Repository-level code retrieval
- Query filtering by stars, year, and topics
- Validated with strong relevance on ML/AI queries

## Current Status

✅ **Phase 1: Data Collection** (Complete - Days 1-3)
- ArXiv API integration
- GitHub repository search
- Filtering & metadata collection
- 249 paper-code pairs collected

🔄 **Phase 2: Model Development & Retrieval** (In Progress - Days 4-7)
- **Code Understanding Model**: CodeBERT/StarCoder-Base with contrastive learning
- **Paper Comprehension Model**: LLaMA-3/Mistral with QLoRA fine-tuning
- **Retrieval System**: FAISS indexing with TF-IDF embeddings
- Dense retrieval pipeline and query testing

⏳ **Phase 3: System Integration** (Upcoming - Days 8-12)
- Backend API (Flask/FastAPI) for model inference
- Connect fine-tuned LLM to retrieval results
- Web interface with search functionality
- End-to-end testing and performance optimization

⏳ **Phase 4: Documentation & Delivery** (Final - Days 13-14)
- Technical documentation and finalization of README
- Final report and presentation slides
- Demo video and deployment instructions

## Documentation

- **[Data Collection Guide](docs/DATA_COLLECTION_GUIDE.md)** - Collection pipeline details
- **[Collection Methods Evaluation](docs/COLLECTION_METHODS_EVALUATION.md)** - Method comparison & rationale
- **[Paper Comprehension Model](docs/PAPER_COMPREHENSION_MODEL.md)** - Model training & deployment

## Requirements

- Python 3.11
- 8GB+ RAM (16GB recommended for model training)
- GitHub Personal Access Token (for data collection)
- Hugging Face account (for model access)

## License

MIT
