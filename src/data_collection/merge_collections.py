"""
Merge multiple paper-code collections into a single deduplicated dataset
"""

import json
from pathlib import Path
from typing import List, Dict, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_collection(filepath: Path) -> List[Dict]:
    """Load a paper-code collection from JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_collections(collections: List[List[Dict]]) -> List[Dict]:
    """
    Merge multiple collections, deduplicating by ArXiv ID

    Args:
        collections: List of paper-code pair lists

    Returns:
        Merged and deduplicated list
    """
    merged = []
    seen_arxiv_ids = set()

    for collection in collections:
        for pair in collection:
            arxiv_id = pair['paper']['arxiv_id']

            # Skip duplicates
            if arxiv_id in seen_arxiv_ids:
                logger.debug(f"Skipping duplicate: {arxiv_id}")
                continue

            merged.append(pair)
            seen_arxiv_ids.add(arxiv_id)

    return merged


def save_merged(pairs: List[Dict], output_dir: Path, filename: str = "paper_code_pairs.json"):
    """Save merged collection"""
    output_path = output_dir / filename

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(pairs)} paper-code pairs to {output_path}")

    # Save statistics
    stats = {
        'total_pairs': len(pairs),
        'total_repositories': sum(len(p.get('repositories', [])) for p in pairs),
        'papers_by_year': {},
        'papers_by_category': {},
        'avg_stars': 0
    }

    all_stars = []
    for pair in pairs:
        paper = pair.get('paper', {})

        # Year stats
        year = paper.get('year', '')
        if year:
            stats['papers_by_year'][str(year)] = stats['papers_by_year'].get(str(year), 0) + 1

        # Category stats
        category = paper.get('category', 'unknown')
        stats['papers_by_category'][category] = stats['papers_by_category'].get(category, 0) + 1

        # Star stats
        for repo in pair.get('repositories', []):
            stars = repo.get('stars', 0)
            if stars:
                all_stars.append(stars)

    if all_stars:
        stats['avg_stars'] = sum(all_stars) / len(all_stars)
        stats['max_stars'] = max(all_stars)
        stats['min_stars'] = min(all_stars)

    stats_path = output_dir / "collection_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved statistics to {stats_path}")
    return output_path


def main():
    """Merge curated and awesome-list collections"""
    data_dir = Path("data/raw/papers")

    # Load collections
    logger.info("Loading collections...")

    curated = load_collection(data_dir / "curated_papers_backup.json")
    logger.info(f"Loaded {len(curated)} curated papers")

    awesome = load_collection(data_dir / "awesome_papers.json")
    logger.info(f"Loaded {len(awesome)} awesome-list papers")

    # Merge
    logger.info("\nMerging collections...")
    merged = merge_collections([curated, awesome])

    logger.info(f"\nMerge complete!")
    logger.info(f"  Curated papers: {len(curated)}")
    logger.info(f"  Awesome-list papers: {len(awesome)}")
    logger.info(f"  Total unique papers: {len(merged)}")
    logger.info(f"  Duplicates removed: {len(curated) + len(awesome) - len(merged)}")

    # Save
    output_path = save_merged(merged, data_dir)

    # Display sample
    print(f"\n{'='*60}")
    print(f"Merged Collection Complete!")
    print(f"{'='*60}")
    print(f"Total paper-code pairs: {len(merged)}")
    print(f"Output file: {output_path}")
    print(f"{'='*60}\n")

    # Show year distribution
    year_dist = {}
    for pair in merged:
        year = pair['paper'].get('year', 'unknown')
        year_dist[str(year)] = year_dist.get(str(year), 0) + 1

    print("Papers by year:")
    for year in sorted(year_dist.keys(), reverse=True):
        print(f"  {year}: {year_dist[year]}")

    # Show top papers by stars
    print("\nTop 5 papers by max stars:")
    sorted_pairs = sorted(merged, key=lambda p: max((r.get('stars', 0) for r in p['repositories']), default=0), reverse=True)
    for i, pair in enumerate(sorted_pairs[:5], 1):
        paper = pair['paper']
        max_stars = max((r.get('stars', 0) for r in pair['repositories']), default=0)
        print(f"  {i}. {paper['title'][:60]}... ({max_stars} stars)")


if __name__ == "__main__":
    main()
