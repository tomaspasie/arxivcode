"""
Complete end-to-end pipeline for paper comprehension dataset creation.

Pipeline:
1. Collect papers from Awesome lists (primary, auto-updating)
2. Fallback to curated list if needed
3. Merge and deduplicate
4. Download code from GitHub repos
5. Generate instruction-tuning QA dataset

Usage:
    python src/data_collection/run_full_pipeline.py
    python src/data_collection/run_full_pipeline.py --skip-code-download
    python src/data_collection/run_full_pipeline.py --skip-awesome
"""

import sys
import logging
import argparse
from pathlib import Path

# Import collectors
from awesome_papers_collector import AwesomePapersCollector
from pwc_hf_collector import PapersWithCodeCollector
from merge_collections import merge_collections, load_collection, save_merged
from code_downloader import CodeDownloader
from paper_qa_dataset_generator import PaperQADatasetGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_pipeline(
    min_stars: int = 50,
    skip_awesome: bool = False,
    skip_curated: bool = False,
    skip_code_download: bool = False,
    skip_qa_generation: bool = False,
    max_repos_download: int = None,
):
    """
    Run the complete paper comprehension dataset creation pipeline.

    Args:
        min_stars: Minimum GitHub stars for repos
        skip_awesome: Skip awesome list collection
        skip_curated: Skip curated list collection
        skip_code_download: Skip downloading code from repos
        skip_qa_generation: Skip QA dataset generation
        max_repos_download: Limit number of repos to download (None = all)
    """
    logger.info("="*80)
    logger.info("PAPER COMPREHENSION DATASET PIPELINE")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Minimum stars: {min_stars}")
    logger.info(f"  Skip awesome lists: {skip_awesome}")
    logger.info(f"  Skip curated list: {skip_curated}")
    logger.info(f"  Skip code download: {skip_code_download}")
    logger.info(f"  Skip QA generation: {skip_qa_generation}")
    logger.info("="*80)

    # Stage 1: Collect from Awesome Lists (Primary Source)
    if not skip_awesome:
        logger.info("\n[Stage 1/5] Collecting papers from Awesome lists...")
        logger.info("This is the PRIMARY source for automatic updates")

        try:
            awesome_collector = AwesomePapersCollector()
            pairs = awesome_collector.collect_all(
                min_stars=min_stars
            )
            awesome_collector.save_results(
                pairs,
                filename="awesome_collection.json"
            )
            logger.info("✅ Awesome list collection complete")
        except Exception as e:
            logger.error(f"❌ Awesome list collection failed: {e}")
            logger.info("Continuing with curated fallback...")
    else:
        logger.info("\n[Stage 1/5] Skipping Awesome list collection")

    # Stage 2: Collect from Curated List (Backup/Fallback)
    if not skip_curated:
        logger.info("\n[Stage 2/5] Collecting from curated list (backup)...")
        logger.info("This is the BACKUP source for high-quality baseline")

        try:
            curated_collector = PapersWithCodeCollector()
            pairs = curated_collector.collect_paper_code_pairs(
                min_stars=min_stars
            )
            curated_collector.save_results(
                pairs,
                filename="curated_collection.json"
            )
            logger.info("✅ Curated list collection complete")
        except Exception as e:
            logger.error(f"❌ Curated list collection failed: {e}")
    else:
        logger.info("\n[Stage 2/5] Skipping curated list collection")

    # Stage 3: Merge Collections
    logger.info("\n[Stage 3/5] Merging collections and removing duplicates...")

    try:
        data_dir = Path("data/raw/papers")
        collections = []

        # Load awesome collection if it exists
        awesome_file = data_dir / "awesome_collection.json"
        if awesome_file.exists():
            awesome_data = load_collection(awesome_file)
            collections.append(awesome_data)
            logger.info(f"Loaded {len(awesome_data)} papers from awesome collection")
        else:
            logger.info("Awesome collection file not found, skipping...")

        # Load curated collection if it exists
        curated_file = data_dir / "curated_collection.json"
        if curated_file.exists():
            curated_data = load_collection(curated_file)
            collections.append(curated_data)
            logger.info(f"Loaded {len(curated_data)} papers from curated collection")
        else:
            logger.info("Curated collection file not found, skipping...")

        if not collections:
            logger.error("❌ No collections found to merge")
            return

        # Merge collections
        merged = merge_collections(collections)
        
        # Save merged collection
        save_merged(merged, data_dir, filename="paper_code_pairs.json")
        
        logger.info("✅ Collections merged successfully")
    except Exception as e:
        logger.error(f"❌ Merge failed: {e}")
        logger.info("Check if at least one collection exists")
        return

    # Stage 4: Download Code from Repositories
    if not skip_code_download:
        logger.info("\n[Stage 4/5] Downloading code from GitHub repositories...")
        logger.info("This provides actual code files for the code understanding model")

        try:
            downloader = CodeDownloader(
                output_dir="data/raw/code_repos",
                max_repo_size_mb=500
            )
            downloader.process_paper_code_pairs(
                input_file="data/raw/papers/paper_code_pairs.json",
                output_file="data/raw/papers/paper_code_with_files.json",
                max_repos=max_repos_download
            )
            logger.info("✅ Code download complete")
        except Exception as e:
            logger.error(f"❌ Code download failed: {e}")
            logger.info("Continuing without code files...")
    else:
        logger.info("\n[Stage 4/5] Skipping code download")

    # Stage 5: Generate QA Dataset for Paper Comprehension
    if not skip_qa_generation:
        logger.info("\n[Stage 5/5] Generating instruction-tuning QA dataset...")
        logger.info("This creates the training data for YOUR paper comprehension model")

        try:
            qa_generator = PaperQADatasetGenerator(seed=42)
            qa_generator.generate_dataset(
                input_file="data/raw/papers/paper_code_pairs.json",
                output_train="data/processed/train.json",
                output_eval="data/processed/eval.json",
                train_ratio=0.9
            )
            logger.info("✅ QA dataset generation complete")
        except Exception as e:
            logger.error(f"❌ QA generation failed: {e}")
            return
    else:
        logger.info("\n[Stage 5/5] Skipping QA dataset generation")

    # Final Summary
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*80)
    logger.info("\nGenerated Files:")
    logger.info("  📄 data/raw/papers/paper_code_pairs.json - Merged paper-code pairs")
    if not skip_code_download:
        logger.info("  📁 data/raw/code_repos/ - Cloned repositories")
        logger.info("  📄 data/raw/papers/paper_code_with_files.json - Papers with code files")
    if not skip_qa_generation:
        logger.info("  📄 data/processed/train.json - Training QA pairs")
        logger.info("  📄 data/processed/eval.json - Evaluation QA pairs")

    logger.info("\nNext Steps:")
    logger.info("  1. Fine-tune model:")
    logger.info("     python -m src.models.trainer --preset paper_understanding \\")
    logger.info("       --train-data data/processed/train.json \\")
    logger.info("       --eval-data data/processed/eval.json")
    logger.info("\n  2. Share with team:")
    logger.info("     - paper_code_with_files.json → Partner 2 (Code Model)")
    logger.info("     - Your fine-tuned model → Partner 3 (Retrieval)")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Complete paper comprehension dataset pipeline"
    )
    parser.add_argument(
        '--min-stars',
        type=int,
        default=50,
        help='Minimum GitHub stars (default: 50)'
    )
    parser.add_argument(
        '--skip-awesome',
        action='store_true',
        help='Skip Awesome list collection (use curated only)'
    )
    parser.add_argument(
        '--skip-curated',
        action='store_true',
        help='Skip curated list collection (use Awesome only)'
    )
    parser.add_argument(
        '--skip-code-download',
        action='store_true',
        help='Skip downloading code from repos'
    )
    parser.add_argument(
        '--skip-qa-generation',
        action='store_true',
        help='Skip QA dataset generation'
    )
    parser.add_argument(
        '--max-repos',
        type=int,
        help='Maximum repos to download (for testing)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode (10 repos, skip code download)'
    )

    args = parser.parse_args()

    # Quick test mode
    if args.quick_test:
        logger.info("🚀 QUICK TEST MODE")
        run_pipeline(
            min_stars=args.min_stars,
            skip_awesome=False,
            skip_curated=True,
            skip_code_download=True,
            skip_qa_generation=False,
            max_repos_download=10
        )
    else:
        run_pipeline(
            min_stars=args.min_stars,
            skip_awesome=args.skip_awesome,
            skip_curated=args.skip_curated,
            skip_code_download=args.skip_code_download,
            skip_qa_generation=args.skip_qa_generation,
            max_repos_download=args.max_repos
        )


if __name__ == '__main__':
    main()
