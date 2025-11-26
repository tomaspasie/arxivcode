# Data Collection Guide

## Quick Start
_Make sure you follow the setup in the `README.md` before running the below commands_

```bash
# 1. Get GitHub token at: https://github.com/settings/tokens
#    Required scopes: public_repo, read:org

# 2. Add to .env file
echo "GITHUB_TOKEN=your_token_here" > .env

# 3. Run complete collection workflow
./scripts/collect_papers.sh
```

## Goal

Collect 200+ high-quality paper-code pairs:
- **Categories**: cs.CL (Computational Linguistics), cs.LG (Machine Learning), cs.CV (Computer Vision)
- **Date Range**: 2013-2025
- **Min GitHub Stars**: 50+

**Output**: `data/raw/papers/paper_code_pairs.json`

## Two-Stage Collection Approach

### Stage 1: Curated Baseline (Starting Point)

**Purpose**: Establish high-quality foundation with landmark papers

**Script**: `pwc_hf_collector.py`

**How it works**:
1. Loads manually curated list of 163 papers from `curated_papers_list.py`
2. Validates each paper's GitHub repository via GitHub API
3. Filters out repos that no longer exist or have <50 stars
4. Saves validated papers with full metadata

**Result**: ~153 papers

**Papers include**:
- Transformers: BERT, GPT-3, LLaMA, Mistral, T5
- Efficient Models: LoRA, QLoRA, FlashAttention, Mamba
- Vision: CLIP, Stable Diffusion, SAM, ControlNet
- And many more landmark papers

**Maintenance**: Update `curated_papers_list.py` occasionally when major new papers emerge

### Stage 2: Automated Expansion (Periodic)

**Purpose**: Discover new papers automatically from community sources

**Script**: `awesome_papers_collector.py` ⭐

**How it works**:
1. Scrapes 10+ "Awesome" GitHub lists (e.g., Awesome-LLM, ML-Papers-of-the-Week)
2. Extracts ArXiv IDs and GitHub URLs from markdown files
3. Fetches paper metadata from ArXiv API
4. Validates GitHub repos via API
5. Filters by year (2020-2025) and stars (50+)

**Result**: ~100 papers per run

**Sources scraped**:
- dair-ai/ML-Papers-of-the-Week
- Hannibal046/Awesome-LLM
- mlabonne/llm-course
- eugeneyan/applied-ml
- opendilab/awesome-RLHF
- And 5+ more community-curated lists

**Run frequency**: Weekly or monthly to catch new papers

### Stage 3: Merge Collections

**Script**: `merge_collections.py`

**How it works**:
1. Loads papers from both stages
2. Deduplicates by ArXiv ID
3. Generates unified dataset with statistics

**Result**: 249 unique papers (153 + 100 - 4 duplicates)

## Usage

### Complete Workflow (All Stages)

```bash
./scripts/collect_papers.sh
```

This runs:
1. Curated collection
2. Awesome-list scraping
3. Merge and deduplication

### Individual Stages

```bash
# Stage 1: Curated baseline
python src/data_collection/pwc_hf_collector.py

# Stage 2: Automated scraping (run periodically)
python src/data_collection/awesome_papers_collector.py

# Stage 3: Merge
python src/data_collection/merge_collections.py
```

### Periodic Updates (Skip Baseline)

If you already have the curated papers and just want to scrape new ones:

```bash
./scripts/collect_papers.sh --skip-curated
```

## Output Files

All files saved to `data/raw/papers/`:

- **paper_code_pairs.json** (197KB) - Final merged collection
  - 249 paper-code pairs
  - Full metadata: title, authors, abstract, year, category
  - Repository info: stars, forks, language, topics

- **collection_stats.json** - Statistics
  - Total papers, repositories
  - Papers by year and category
  - Average/min/max stars

- **curated_papers_backup.json** - Backup of curated collection
- **awesome_papers.json** - Papers from awesome lists

## Statistics

**Current Collection**: 249 papers

**By Year**:
- 2025: 28 papers
- 2024: 14 papers
- 2023: 84 papers
- 2022: 24 papers
- 2021: 28 papers
- 2020: 22 papers
- 2013-2019: 49 papers

**By Category**:
- cs.LG (Machine Learning): 138 papers
- cs.CL (Computational Linguistics): 65 papers
- cs.CV (Computer Vision): 30 papers
- Others: 16 papers

**GitHub Stars**:
- Average: 11,470 stars
- Max: 192,543 (TensorFlow)
- Min: 12 stars

## Requirements

- Python 3.11
- GitHub Personal Access Token

**Rate Limits**:
- Without token: ~60 requests/hour (not recommended)
- With token: ~5,000 requests/hour

**Estimated time**:
- Stage 1 (Curated): 5-10 minutes
- Stage 2 (Awesome scraping): 10-15 minutes
- Stage 3 (Merge): <1 minute
- **Total**: ~20 minutes

## Maintenance

### Adding New Curated Papers

Edit `src/data_collection/curated_papers_list.py`:

```python
papers.extend([
    {
        "arxiv_id": "2401.12345",
        "title": "Your New Paper",
        "github_urls": ["https://github.com/owner/repo"],
        "year": 2024,
        "category": "cs.LG"
    },
])
```

Then re-run the collection workflow.

### Adding New Awesome Lists

Edit `src/data_collection/awesome_papers_collector.py`:

```python
self.awesome_lists = [
    # ... existing lists ...
    {
        'repo': 'username/awesome-ml-list',
        'category': 'cs.LG',
        'description': 'Description of the list'
    },
]
```

## Troubleshooting

### GitHub Rate Limit Exceeded

**Error**: `GitHub rate limit exceeded`

**Solution**:
1. Check your token is in `.env`
2. Wait for rate limit reset (check with `gh api rate_limit`)
3. Token needs `public_repo` scope

### No Papers Found from Awesome Lists

**Possible causes**:
- Lists don't have ArXiv links + GitHub links together
- Papers outside date range (2020-2025)
- Repos have <50 stars

**Solution**: Check logs for which lists were scraped successfully

### Merge Produces Fewer Papers Than Expected

This is normal - duplicates are removed. Check logs:
```
Duplicates removed: 4
```

## Next Steps

After collecting papers:
1. Verify collection: `cat data/raw/papers/collection_stats.json`
2. Proceed to data processing: See [PROCESSING_GUIDE.md](PROCESSING_GUIDE.md)
3. Set up periodic collection: Add to cron/scheduler to run weekly

## Collection Method Summary

| Method | Papers | Speed | Sustainability |
|--------|--------|-------|----------------|
| Curated Baseline | 153 | Fast (5-10 min) | Manual updates needed |
| Awesome Scraping | ~100 | Medium (10-15 min) | **Fully automated** |
| **Combined** | **249** | **~20 min** | **Sustainable** |

The two-stage approach gives you:
- ✅ High-quality baseline (curated)
- ✅ Automatic discovery of new papers (awesome lists)
- ✅ Sustainable long-term solution
- ✅ 200+ papers achieved!
