# ArXivCode Tutorial

This tutorial will guide you through setting up and running the ArXivCode project locally. ArXivCode is a system that bridges the gap between AI research papers and their code implementations, allowing you to search for theoretical concepts and retrieve relevant code snippets with explanations.

## Table of Contents

1. [Requirements](#requirements)
2. [Local Setup](#local-setup)
3. [Running the Application](#running-the-application)
4. [Advanced Usage](#advanced-usage)
   - [Data Collection Pipeline](#data-collection-pipeline)
   - [Code Extraction and Cleaning](#code-extraction-and-cleaning)
   - [Embedding Generation](#embedding-generation)

---

## Requirements

### System Requirements

- **Python**: 3.8 or higher (3.11 recommended)
- **RAM**: 16GB recommended (8GB minimum)
- **Disk Space**: At least 5GB free space for dependencies and data
- **Operating System**: macOS, Linux, or Windows (WSL recommended for Windows)
- **GPU** (for training/embedding): GPU with at least 8 GiB of memory is recommended if you plan to run training or generate embeddings yourself. CPU-only operation is possible but significantly slower.

### API Keys and Accounts

You'll need the following API keys and accounts:

1. **GitHub Personal Access Token** (for data collection)
   - Create at: https://github.com/settings/tokens
   - Required scopes: `public_repo` (read access to public repositories)

2. **OpenAI API Key** (for LLM explanations)
   - Get at: https://platform.openai.com/api-keys
   - Required for the explanation feature

3. **Hugging Face Account** (for model downloads)
   - Sign up at: https://huggingface.co/
   - Models will be downloaded automatically on first use

---

## Local Setup

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd arxivcode
```

### Step 2: Create Virtual Environment

It's recommended to use a virtual environment to isolate dependencies:

```bash
# Create virtual environment (Python 3.11 recommended)
python3.11 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### Step 3: Install Dependencies

Install all required packages from `requirements.txt`:

```bash
# Make sure virtual environment is activated
pip install --upgrade pip

# Install PyTorch first (with CUDA support if you have a GPU)
# For CPU only:
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# For CUDA 11.8 (Tesla P4 GPU or similar):
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install all other dependencies
pip install -r requirements.txt
```

**Note**: The first installation may take several minutes as it downloads large packages like PyTorch, transformers, and sentence-transformers.

### Step 4: Configure Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit the `.env` file and add your API keys:

```bash
# .env file
GITHUB_TOKEN=your_github_token_here
OPENAI_API_KEY=your_openai_api_key_here
```

**Security Note**: Never commit your `.env` file to version control. It's already included in `.gitignore`.

### Step 5: Download Large Data Files

The project uses Git LFS (Large File Storage) to manage large data files like pre-computed embeddings. You need to download these files before running the application:

```bash
# Install Git LFS (if not already installed)
brew install git-lfs

# Initialize Git LFS in the repository
git lfs install

# Download the large data files
git lfs pull
```

**Note**: This will download approximately 100MB of data files. The embeddings are essential for the retrieval system to work.

### Step 6: Download CodeBERT Model (First Run)

The CodeBERT model will be automatically downloaded on first use when you run the application. However, if you want to pre-download it:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('microsoft/codebert-base')"
```

This will download the model to your Hugging Face cache directory (typically `~/.cache/huggingface/`).

### Step 7: Verify Installation

Test that everything is installed correctly:

```bash
python -c "import torch; import transformers; import sentence_transformers; print('All imports successful!')"
```

---

## Running the Application

The ArXivCode system consists of two components:
1. **Backend API** (FastAPI) - Handles search and explanation requests
2. **Frontend UI** (Streamlit) - Provides the web interface

### Starting the Backend API

Open a terminal and run:

```bash
# Activate virtual environment
source venv/bin/activate

# Start the API server
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Or use the provided script:

```bash
./scripts/start_api.sh
```

The API will be available at:
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (interactive Swagger UI)

You should see output indicating that the retrieval system and explanation LLM are loading. The first startup may take a minute or two as it loads the CodeBERT model and FAISS index.

### Starting the Frontend UI

Open a **new terminal** (keep the API running) and run:

```bash
# Activate virtual environment
source venv/bin/activate

# Start the Streamlit frontend
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

The web interface will be available at: **http://localhost:8501**

### Using the Application

1. Open your browser and navigate to http://localhost:8501
2. Enter a search query in the search box (e.g., "how to implement LoRA", "transformer attention mechanism")
3. Adjust the number of results using the slider
4. Click "Search" to retrieve relevant code snippets
5. Click "Explain" on any result to get an AI-generated explanation
6. Use the "View Paper" and "View Repo" buttons to access the original sources

### Example Queries

- "how to implement LoRA"
- "transformer attention mechanism"
- "BERT fine-tuning"
- "flash attention"
- "PPO reinforcement learning"
- "vision transformer"
- "knowledge distillation"

---

## Advanced Usage

This section covers advanced operations for data collection, code extraction, embedding generation, and other development tasks.

### Data Collection Pipeline

The data collection pipeline downloads papers, associated GitHub repositories, and extracts code snippets. This is useful if you want to:
- Add new papers to the dataset
- Update existing code collections
- Build your own custom dataset

#### Running the Full Pipeline

The complete pipeline includes:
1. Collecting papers from Awesome lists
2. Collecting from curated paper lists
3. Merging and deduplicating collections
4. Downloading code from GitHub repositories
5. Generating QA datasets (optional)

```bash
# Activate virtual environment
source venv/bin/activate

# Run the full pipeline
python src/data_collection/run_full_pipeline.py
```

**Options**:

```bash
# Skip specific stages
python src/data_collection/run_full_pipeline.py --skip-code-download
python src/data_collection/run_full_pipeline.py --skip-awesome
python src/data_collection/run_full_pipeline.py --skip-curated

# Quick test mode (10 repos, skips code download)
python src/data_collection/run_full_pipeline.py --quick-test

# Limit number of repositories
python src/data_collection/run_full_pipeline.py --max-repos 50

# Set minimum GitHub stars threshold
python src/data_collection/run_full_pipeline.py --min-stars 100
```

**Output Files**:
- `data/raw/papers/paper_code_pairs.json` - Merged paper-code pairs
- `data/raw/code_repos/` - Cloned repositories
- `data/raw/papers/paper_code_with_files.json` - Papers with code files

#### Individual Collection Steps

You can also run individual collection steps:

**1. Collect from Awesome Lists**:

```bash
python -m src.data_collection.awesome_papers_collector
```

**2. Collect from Curated List**:

```bash
python -m src.data_collection.pwc_hf_collector
```

**3. Download Code from Repositories**:

```bash
python -m src.data_collection.code_downloader
```

The code downloader will:
- Clone GitHub repositories listed in `data/raw/papers/paper_code_pairs.json`
- Filter repositories by size (default: max 500MB)
- Extract code files based on file extensions
- Save repository paths to the dataset

**Configuration**:
- Set `GITHUB_TOKEN` in `.env` for higher rate limits
- Adjust `max_repo_size_mb` in the code downloader to filter large repos
- Modify `file_extensions` to include/exclude specific file types

### Code Extraction and Cleaning

After downloading repositories, you need to extract code snippets and clean the dataset.

#### Extract Code Snippets

Extract functions and methods from Python files:

```bash
python -m src.data_collection.extract_snippets
```

This script:
- Parses Python files using AST
- Extracts individual functions and class methods
- Saves each snippet with metadata (paper title, function name, file path, etc.)
- Outputs to `data/processed/code_snippets.json`

**Parameters** (modify in script):
- `min_lines`: Minimum lines for a function to be extracted (default: 50)
- `require_docstring`: Whether to require docstrings (default: True)

#### Clean the Dataset

Clean and filter the extracted code snippets:

```bash
python -m src.data_collection.clean_dataset
```

This script:
- Fixes generic "arXiv Query" titles by fetching real metadata
- Filters out irrelevant code (tests, utils, configs)
- Scores code-paper relevance
- Removes low-quality entries
- Outputs to `data/processed/code_snippets_cleaned.json`

**Filtering Criteria**:
- Removes test files (`*_test.py`, `tests/`, `testing/`)
- Removes configuration files (`setup.py`, `config.py`, `utils.py`)
- Removes utility/helper code
- Requires meaningful function names
- Keeps only paper-relevant implementations

**Expected Results**:
- Raw snippets: 37,000+ (before cleaning)
- Cleaned snippets: ~2,490 (after filtering)
- Reduction: ~93.3%

### Embedding Generation

Generate embeddings for code snippets using CodeBERT. This is required before the retrieval system can work.

> **Note**: A GPU with at least 8 GiB of memory is **strongly recommended** for generating embeddings. While CPU-only operation is possible, it will be significantly slower (potentially 10-50x slower depending on your dataset size). For training the code encoder, a GPU is essential.

#### Generate Embeddings

```bash
# Activate virtual environment
source venv/bin/activate

# Generate embeddings with default settings
python src/embeddings/generate_improved_embeddings.py
```

**Options**:

```bash
# Specify input/output paths
python src/embeddings/generate_improved_embeddings.py \
    --input data/processed/code_snippets_cleaned.json \
    --output data/processed/embeddings_v2

# Use different embedding strategy
python src/embeddings/generate_improved_embeddings.py \
    --strategy enhanced  # Options: code_only, code_abstract, enhanced, searchable

# Use GPU (if available)
python src/embeddings/generate_improved_embeddings.py --device cuda

# Adjust batch size
python src/embeddings/generate_improved_embeddings.py --batch-size 64

# Test existing embeddings
python src/embeddings/generate_improved_embeddings.py --test-only
```

**Embedding Strategies**:

1. **`enhanced`** (recommended): Combines paper title + function name + docstring + code
2. **`searchable`**: Optimized for matching user queries
3. **`code_abstract`**: Code + paper abstract
4. **`code_only`**: Just the code text

**Output Files**:
- `data/processed/embeddings_v2/code_embeddings.npy` - NumPy array of embeddings (768-dim vectors)
- `data/processed/embeddings_v2/metadata.json` - Metadata for each embedding
- `data/processed/embeddings_v2/config.json` - Configuration used for generation

**Embedding Statistics**:
- Model: `microsoft/codebert-base`
- Embedding Dimension: 768
- Storage: ~7.3 MB for 2,490 snippets
- Index Type: FAISS FlatIP (cosine similarity)

#### Build FAISS Index

After generating embeddings, build the FAISS index for fast retrieval:

```bash
python src/retrieval/build_index.py
```

This creates a FAISS index file that the retrieval system uses for fast similarity search.

### Training Code Encoder (Optional)

> **Note**: Training requires a GPU with at least 8 GiB of memory. CPU training is not recommended due to extremely long training times.

If you want to fine-tune the CodeBERT model on your dataset:

```bash
python src/embeddings/train_code_encoder.py \
    --train-data data/processed/train.json \
    --eval-data data/processed/eval.json \
    --output-dir models/code_encoder_finetuned
```

### Testing

Run tests to verify the system is working correctly:

```bash
# Test API endpoints
python -m pytest tests/test_api.py

# Test retrieval system
python src/retrieval/test_retrieval.py

# Test embedding generation
python tests/test_code_embeddings.py
```

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Make sure you're in the project root and virtual environment is activated
source venv/bin/activate
cd /path/to/arxivcode
```

**2. Model Download Issues**
- Check your internet connection
- Verify Hugging Face account access
- Clear cache: `rm -rf ~/.cache/huggingface/`

**3. API Key Errors**
- Verify `.env` file exists and contains correct keys
- Check key format (no quotes, no extra spaces)
- For GitHub: Ensure token has `public_repo` scope
- For OpenAI: Verify account has credits

**4. Git LFS Data Files Not Downloaded**
- If you get errors about loading embeddings or "Cannot load file containing pickled data"
- Install Git LFS: `brew install git-lfs`
- Initialize in repo: `git lfs install`
- Download files: `git lfs pull`
- Check that `data/processed/embeddings/code_embeddings.npy` is >100MB (not 134 bytes)

**5. Port Already in Use**
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process or use a different port
python -m uvicorn src.api.app:app --port 8001
```

**6. Out of Memory**
- Reduce batch size in embedding generation: `--batch-size 16`
- Use CPU instead of GPU: `--device cpu`
- Close other applications

**7. Missing Data Files**
- Ensure you've run the data collection pipeline
- Check that `data/processed/embeddings_v2/` exists
- Verify `code_embeddings.npy` and `metadata.json` are present

### Getting Help

- Check the main [README.md](README.md) for overview
- Review documentation in `docs/` directory
- Check existing issues on GitHub
- Review code comments in source files

---

## Next Steps

After completing the setup:

1. **Explore the Codebase**: Review the project structure in `src/`
2. **Customize Queries**: Try different search queries to understand retrieval quality
3. **Add Papers**: Use the data collection pipeline to add new papers
4. **Fine-tune Models**: Experiment with different embedding strategies
5. **Extend Functionality**: Add new features to the API or frontend

---

## Summary

This tutorial covered:
- ✅ System requirements and API key setup
- ✅ Local environment setup with virtual environment
- ✅ Installing dependencies from `requirements.txt`
- ✅ Configuring environment variables
- ✅ Downloading large data files with Git LFS
- ✅ Running the backend API and frontend UI
- ✅ Advanced data collection pipeline
- ✅ Code extraction and cleaning
- ✅ Embedding generation with CodeBERT
- ✅ Troubleshooting common issues

You should now be able to run ArXivCode locally and explore its features. For more detailed information, refer to the documentation in the `docs/` directory.
