#!/bin/bash
# Complete paper collection workflow
# Usage: ./scripts/collect_papers.sh [--skip-curated]

set -e  # Exit on error

echo "=================================="
echo "Paper-Code Pair Collection"
echo "=================================="
echo ""

# Activate virtual environment
source venv/bin/activate

# Check if we should skip curated collection
SKIP_CURATED=false
if [[ "$1" == "--skip-curated" ]]; then
    SKIP_CURATED=true
    echo "⏭️  Skipping curated collection..."
fi

# Stage 1: Curated collection (initial baseline)
if [ "$SKIP_CURATED" = false ]; then
    echo "📚 Stage 1: Collecting curated high-impact papers..."
    python src/data_collection/pwc_hf_collector.py
    echo ""
fi

# Stage 2: Automated awesome-list scraping
echo "🤖 Stage 2: Scraping Awesome lists for new papers..."
python src/data_collection/awesome_papers_collector.py
echo ""

# Merge collections
echo "🔀 Stage 3: Merging collections and removing duplicates..."
python src/data_collection/merge_collections.py
echo ""

echo "✅ Collection complete!"
echo "📄 Output: data/raw/papers/paper_code_pairs.json"
echo "📊 Stats: data/raw/papers/collection_stats.json"
