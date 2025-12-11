# Results Generation Script

This script generates all the metrics, evaluations, and case studies needed for Section 5 (Results) of the Final Report.

## Usage

```bash
# From project root
python results/generate_results.py
```

## Requirements

1. **Data files must exist:**
   - `data/processed/embeddings_v2/code_embeddings.npy`
   - `data/processed/embeddings_v2/metadata.json`
   - `data/processed/code_snippets_cleaned.json` (optional, for additional stats)

2. **Optional (for explanation generation):**
   - `OPENAI_API_KEY` environment variable set
   - Without this, the script will still run but skip explanation generation

## Output

The script generates two files in `results/`:

1. **`evaluation_results.json`**: Complete results in JSON format
2. **`results_summary.md`**: Markdown summary formatted for the report

## What It Generates

### 5.1 System Performance
- Papers indexed count
- Code snippets count
- Average retrieval time (measured)
- Average explanation time (measured)

### 5.2 Retrieval Accuracy
- Evaluates 30 test queries
- Categorizes by query type (Architecture, Implementation, Conceptual)
- Calculates relevant/partial/not relevant percentages
- **Note**: Uses heuristic estimation. For true accuracy, manual evaluation is recommended.

### 5.3 Case Studies
- **Success**: "multi-head attention" query
- **Partial**: "learning rate schedule" query  
- **Failure**: "why use layer norm" query

### 5.4 Baseline Comparison
- Compares ArXivCode with:
  - Manual GitHub search
  - GPT-4 Zero-Shot
  - GitHub Search

### 5.5 Error Analysis
- Query ambiguity percentage
- Missing papers percentage
- Code complexity issues
- Conceptual gap analysis

## Customization

You can modify the test queries in the `ResultsGenerator.__init__()` method to evaluate different queries.

## Notes

- The script uses heuristic estimation for relevance. For publication-quality results, manual evaluation of retrieved results is recommended.
- Explanation generation requires OpenAI API access and will incur costs (~$0.06 per explanation).
- Retrieval time measurements are based on actual system performance.
- The script automatically handles missing LLM access gracefully.

