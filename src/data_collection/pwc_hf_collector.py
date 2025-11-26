"""
Papers With Code + GitHub Collector using Hugging Face Datasets
Alternative approach that doesn't rely on ArXiv API
"""

import json
import time
import os
import sys
import re
from typing import List, Dict
from pathlib import Path
from github import Github
from dotenv import load_dotenv
import logging

# Add parent directory to path to import curated_papers_list
sys.path.insert(0, str(Path(__file__).parent))
from curated_papers_list import get_curated_papers

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PapersWithCodeCollector:
    """
    Collector that uses Papers With Code data to find paper-code pairs
    """

    def __init__(self, github_token: str = None, output_dir: str = "data/raw/papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GitHub API
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.github = Github(self.github_token) if self.github_token else Github()

    def validate_and_fetch_repos(self, paper: Dict, min_stars: int = 50) -> List[Dict]:
        """
        Validate GitHub URLs and fetch repository metadata

        Args:
            paper: Paper dictionary with github_urls
            min_stars: Minimum number of stars

        Returns:
            List of repository dictionaries
        """
        repos = []
        github_urls = paper.get("github_urls", [])

        for url in github_urls:
            try:
                # Extract owner/repo from URL

                match = re.search(r"github\.com/([\w-]+)/([\w.-]+)", url, re.IGNORECASE)
                if not match:
                    continue

                owner, repo_name = match.groups()
                repo = self.github.get_repo(f"{owner}/{repo_name}")

                # Check if repo meets minimum stars requirement
                if repo.stargazers_count >= min_stars:
                    repo_data = {
                        "url": repo.html_url,
                        "name": repo.full_name,
                        "description": repo.description,
                        "stars": repo.stargazers_count,
                        "forks": repo.forks_count,
                        "language": repo.language,
                        "created_at": (
                            repo.created_at.isoformat() if repo.created_at else None
                        ),
                        "updated_at": (
                            repo.updated_at.isoformat() if repo.updated_at else None
                        ),
                        "topics": (
                            repo.get_topics() if hasattr(repo, "get_topics") else []
                        ),
                    }
                    repos.append(repo_data)
                    logger.info(f"  ✓ {repo.full_name} ({repo.stargazers_count} stars)")
                else:
                    logger.info(
                        f"  ✗ {owner}/{repo_name} has only {repo.stargazers_count} stars (min: {min_stars})"
                    )

                time.sleep(1)  # Rate limiting

            except Exception as e:
                logger.warning(f"  Error fetching repo from {url}: {e}")
                continue

        return repos

    def collect_paper_code_pairs(self, min_stars: int = 50) -> List[Dict]:
        """
        Collect paper-code pairs from curated list

        Args:
            min_stars: Minimum GitHub stars

        Returns:
            List of paper-code pair dictionaries
        """
        logger.info("Starting curated paper-code collection")
        logger.info(f"Filters - Min Stars: >={min_stars}")

        papers = get_curated_papers()
        logger.info(f"Processing {len(papers)} curated papers")

        paper_code_pairs = []

        for i, paper_info in enumerate(papers, 1):
            logger.info(
                f"\n[{i}/{len(papers)}] Processing: {paper_info['title'][:60]}..."
            )

            repos = self.validate_and_fetch_repos(paper_info, min_stars=min_stars)

            if repos:
                # Create full paper metadata (simplified version)
                paper = {
                    "arxiv_id": paper_info["arxiv_id"],
                    "title": paper_info["title"],
                    "year": paper_info["year"],
                    "category": paper_info["category"],
                    "url": f"https://arxiv.org/abs/{paper_info['arxiv_id']}",
                }

                pair = {
                    "paper": paper,
                    "repositories": repos,
                }
                paper_code_pairs.append(pair)
                logger.info(f"  ✓ Added pair with {len(repos)} repositories")
            else:
                logger.info(f"  ✗ No repositories met criteria")

        logger.info(f"\nTotal paper-code pairs collected: {len(paper_code_pairs)}")
        return paper_code_pairs

    def save_results(
        self, pairs: List[Dict], filename: str = "paper_code_pairs.json"
    ) -> Path:
        """Save collected paper-code pairs to JSON file"""
        output_path = self.output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(pairs)} paper-code pairs to {output_path}")

        # Save statistics
        stats = {
            "total_pairs": len(pairs),
            "total_repositories": sum(len(p.get("repositories", [])) for p in pairs),
            "papers_by_year": {},
            "avg_stars": 0,
        }

        all_stars = []
        for pair in pairs:
            paper = pair.get("paper", {})
            year = paper.get("year", "")
            if year:
                stats["papers_by_year"][str(year)] = (
                    stats["papers_by_year"].get(str(year), 0) + 1
                )

            for repo in pair.get("repositories", []):
                stars = repo.get("stars", 0)
                if stars:
                    all_stars.append(stars)

        if all_stars:
            stats["avg_stars"] = sum(all_stars) / len(all_stars)
            stats["max_stars"] = max(all_stars)
            stats["min_stars"] = min(all_stars)

        stats_path = self.output_dir / "collection_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved statistics to {stats_path}")
        return output_path


def main():
    """Main execution function"""
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        logger.warning("No GITHUB_TOKEN found. API rate limits will be lower.")

    collector = PapersWithCodeCollector(github_token=github_token)

    # Collect paper-code pairs from curated list
    pairs = collector.collect_paper_code_pairs(min_stars=50)

    # Save results
    if pairs:
        output_path = collector.save_results(pairs)

        print(f"\n{'='*60}")
        print(f"Collection Complete!")
        print(f"{'='*60}")
        print(f"Paper-code pairs collected: {len(pairs)}")
        print(f"Output file: {output_path}")
        print(f"{'='*60}\n")

        # Display sample pairs
        print("Sample paper-code pairs:")
        for i, pair in enumerate(pairs[:5], 1):
            paper = pair["paper"]
            repos = pair["repositories"]
            print(f"\n{i}. {paper['title']}")
            print(f"   ArXiv: {paper['arxiv_id']}")
            print(f"   Year: {paper.get('year', 'N/A')}")
            print(f"   Repositories: {len(repos)}")
            if repos:
                top_repo = max(repos, key=lambda r: r.get("stars", 0))
                print(f"   Top repo: {top_repo['name']} ({top_repo['stars']} stars)")
    else:
        print("No pairs collected!")


if __name__ == "__main__":
    main()
