"""
Automated collector that scrapes "Awesome" GitHub lists to find paper-code pairs
This provides a sustainable middle ground between manual curation and full automation
"""

import requests
import json
import re
import os
import time
from typing import List, Dict, Set
from pathlib import Path
from github import Github
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AwesomePapersCollector:
    """
    Collects paper-code pairs from community-curated "Awesome" lists on GitHub
    """

    def __init__(self, github_token: str = None, output_dir: str = "data/raw/papers"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GitHub API
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.github = Github(self.github_token) if self.github_token else Github()

        # Awesome lists to scrape
        self.awesome_lists = [
            {
                'repo': 'dair-ai/ML-Papers-of-the-Week',
                'category': 'cs.LG',
                'description': 'Top ML papers of the week'
            },
            {
                'repo': 'papers-we-love/papers-we-love',
                'category': 'cs.LG',
                'description': 'Papers from the computer science community'
            },
            {
                'repo': 'eugeneyan/applied-ml',
                'category': 'cs.LG',
                'description': 'Applied ML papers and repositories'
            },
            {
                'repo': 'mlabonne/llm-course',
                'category': 'cs.CL',
                'description': 'LLM course with papers and implementations'
            },
            {
                'repo': 'Hannibal046/Awesome-LLM',
                'category': 'cs.CL',
                'description': 'Curated list of LLM papers and resources'
            },
            {
                'repo': 'thunlp/PLMpapers',
                'category': 'cs.CL',
                'description': 'Pre-trained Language Model papers'
            },
            {
                'repo': 'sebastianruder/NLP-progress',
                'category': 'cs.CL',
                'description': 'NLP progress tracking with papers'
            },
            {
                'repo': 'huggingface/diffusion-models-class',
                'category': 'cs.CV',
                'description': 'Diffusion models with papers'
            },
            {
                'repo': 'jbhuang0604/awesome-computer-vision',
                'category': 'cs.CV',
                'description': 'Computer vision resources and papers'
            },
            {
                'repo': 'opendilab/awesome-RLHF',
                'category': 'cs.LG',
                'description': 'RLHF papers and implementations'
            },
        ]

    def scrape_awesome_list(self, repo_name: str, category: str) -> List[Dict]:
        """
        Scrape an awesome list repository for paper-code pairs

        Args:
            repo_name: GitHub repository name (e.g., 'dair-ai/ML-Papers-of-the-Week')
            category: ArXiv category to assign

        Returns:
            List of paper-code pairs found
        """
        pairs = []
        seen_arxiv_ids = set()

        try:
            logger.info(f"\nScraping: {repo_name}")
            repo = self.github.get_repo(repo_name)

            # Get all markdown files in the repo
            contents = repo.get_contents("")
            markdown_files = []

            # Find README and other markdown files
            for content in contents:
                if content.name.lower().endswith('.md'):
                    markdown_files.append(content)

            # Also check subdirectories
            try:
                for content in contents:
                    if content.type == "dir" and content.name in ['papers', 'docs', 'resources']:
                        subcontents = repo.get_contents(content.path)
                        for subcontent in subcontents:
                            if subcontent.name.lower().endswith('.md'):
                                markdown_files.append(subcontent)
            except:
                pass

            # Parse each markdown file
            for md_file in markdown_files:
                try:
                    content_text = md_file.decoded_content.decode('utf-8', errors='ignore')

                    # Extract ArXiv URLs and associated GitHub URLs
                    # Pattern: Look for lines with both ArXiv and GitHub links
                    lines = content_text.split('\n')

                    for i, line in enumerate(lines):
                        # Find ArXiv IDs
                        arxiv_matches = re.findall(r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)', line, re.IGNORECASE)

                        for arxiv_id in arxiv_matches:
                            if arxiv_id in seen_arxiv_ids:
                                continue

                            # Look for GitHub URLs in the same line or nearby lines
                            github_urls = []

                            # Check current line and next 3 lines
                            context_lines = [line] + lines[i+1:i+4] if i+3 < len(lines) else [line]
                            for context_line in context_lines:
                                github_matches = re.findall(
                                    r'https?://(?:www\.)?github\.com/([\w-]+)/([\w.-]+)',
                                    context_line,
                                    re.IGNORECASE
                                )
                                for owner, repo_name_match in github_matches:
                                    # Skip the awesome list repo itself
                                    if f"{owner}/{repo_name_match}" != repo_name:
                                        github_urls.append(f"https://github.com/{owner}/{repo_name_match}")

                            if not github_urls:
                                # Try to extract from paper title - search GitHub
                                continue

                            # Get paper metadata from ArXiv
                            try:
                                arxiv_api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
                                arxiv_response = requests.get(arxiv_api_url, timeout=10)

                                if arxiv_response.status_code == 200:
                                    import xml.etree.ElementTree as ET
                                    root = ET.fromstring(arxiv_response.content)

                                    title_elem = root.find('.//{http://www.w3.org/2005/Atom}title')
                                    published_elem = root.find('.//{http://www.w3.org/2005/Atom}published')

                                    if title_elem is not None:
                                        paper_title = title_elem.text.strip()

                                        # Extract year
                                        year = 2020  # Default
                                        if published_elem is not None:
                                            year_match = re.search(r'(\d{4})', published_elem.text)
                                            if year_match:
                                                year = int(year_match.group(1))

                                        # Get GitHub repo metadata
                                        repo_data_list = []
                                        for github_url in github_urls[:3]:  # Limit to 3 repos per paper
                                            try:
                                                repo_match = re.search(r'github\.com/([\w-]+)/([\w.-]+)', github_url)
                                                if repo_match:
                                                    owner, repo_name_gh = repo_match.groups()
                                                    gh_repo = self.github.get_repo(f"{owner}/{repo_name_gh}")

                                                    repo_data_list.append({
                                                        'url': gh_repo.html_url,
                                                        'name': gh_repo.full_name,
                                                        'description': gh_repo.description,
                                                        'stars': gh_repo.stargazers_count,
                                                        'forks': gh_repo.forks_count,
                                                        'language': gh_repo.language,
                                                        'created_at': gh_repo.created_at.isoformat() if gh_repo.created_at else None,
                                                        'updated_at': gh_repo.updated_at.isoformat() if gh_repo.updated_at else None,
                                                        'topics': gh_repo.get_topics() if hasattr(gh_repo, 'get_topics') else []
                                                    })

                                                    time.sleep(0.5)  # Rate limiting
                                            except Exception as e:
                                                logger.debug(f"Error fetching GitHub repo {github_url}: {e}")
                                                continue

                                        if repo_data_list:
                                            pair = {
                                                'paper': {
                                                    'arxiv_id': arxiv_id,
                                                    'title': paper_title,
                                                    'year': year,
                                                    'category': category,
                                                    'url': f"https://arxiv.org/abs/{arxiv_id}"
                                                },
                                                'repositories': repo_data_list
                                            }

                                            pairs.append(pair)
                                            seen_arxiv_ids.add(arxiv_id)

                                            max_stars = max(r['stars'] for r in repo_data_list)
                                            logger.info(f"  ✓ {paper_title[:60]}... ({len(repo_data_list)} repos, max {max_stars} stars)")

                                time.sleep(1)  # ArXiv rate limiting

                            except Exception as e:
                                logger.debug(f"Error fetching ArXiv data for {arxiv_id}: {e}")
                                continue

                except Exception as e:
                    logger.debug(f"Error parsing {md_file.name}: {e}")
                    continue

            logger.info(f"  Found {len(pairs)} paper-code pairs from {repo_name}")

        except Exception as e:
            logger.error(f"Error scraping {repo_name}: {e}")

        return pairs

    def collect_all(self, min_stars: int = 50, target_count: int = 200,
                    year_start: int = 2020, year_end: int = 2025) -> List[Dict]:
        """
        Collect papers from all awesome lists

        Args:
            min_stars: Minimum stars for GitHub repositories
            target_count: Target number of papers
            year_start: Start year filter
            year_end: End year filter

        Returns:
            List of paper-code pairs
        """
        all_pairs = []
        seen_arxiv_ids = set()

        logger.info(f"Collecting papers from {len(self.awesome_lists)} awesome lists...")
        logger.info(f"Filters: min_stars={min_stars}, years={year_start}-{year_end}")

        for awesome_list in self.awesome_lists:
            if len(all_pairs) >= target_count:
                break

            pairs = self.scrape_awesome_list(
                repo_name=awesome_list['repo'],
                category=awesome_list['category']
            )

            # Filter and add pairs
            for pair in pairs:
                if len(all_pairs) >= target_count:
                    break

                arxiv_id = pair['paper']['arxiv_id']
                if arxiv_id in seen_arxiv_ids:
                    continue

                # Filter by year
                year = pair['paper'].get('year', 2020)
                if year < year_start or year > year_end:
                    continue

                # Filter by stars
                max_stars = max((r.get('stars', 0) for r in pair['repositories']), default=0)
                if max_stars < min_stars:
                    continue

                all_pairs.append(pair)
                seen_arxiv_ids.add(arxiv_id)

            time.sleep(2)  # Rate limit between repos

        logger.info(f"\nTotal paper-code pairs collected: {len(all_pairs)}")
        return all_pairs

    def save_results(self, pairs: List[Dict], filename: str = "awesome_papers.json") -> Path:
        """Save collected papers to JSON"""
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(pairs, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(pairs)} paper-code pairs to {output_path}")

        # Save statistics
        stats = {
            'total_pairs': len(pairs),
            'total_repositories': sum(len(p.get('repositories', [])) for p in pairs),
            'papers_by_year': {},
            'avg_stars': 0
        }

        all_stars = []
        for pair in pairs:
            paper = pair.get('paper', {})
            year = paper.get('year', '')
            if year:
                stats['papers_by_year'][str(year)] = stats['papers_by_year'].get(str(year), 0) + 1

            for repo in pair.get('repositories', []):
                stars = repo.get('stars', 0)
                if stars:
                    all_stars.append(stars)

        if all_stars:
            stats['avg_stars'] = sum(all_stars) / len(all_stars)
            stats['max_stars'] = max(all_stars)
            stats['min_stars'] = min(all_stars)

        stats_path = self.output_dir / "awesome_papers_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        return output_path


def main():
    """Main execution"""
    github_token = os.getenv('GITHUB_TOKEN')
    if not github_token:
        logger.warning("No GITHUB_TOKEN found. API rate limits will be lower.")

    collector = AwesomePapersCollector(github_token=github_token)

    # Collect papers
    pairs = collector.collect_all(
        min_stars=50,
        target_count=100,  # Collect 100 from awesome lists
        year_start=2020,
        year_end=2025
    )

    # Save results
    if pairs:
        output_path = collector.save_results(pairs)

        print(f"\n{'='*60}")
        print(f"Awesome Lists Collection Complete!")
        print(f"{'='*60}")
        print(f"Paper-code pairs collected: {len(pairs)}")
        print(f"Output file: {output_path}")
        print(f"{'='*60}\n")

        # Display sample pairs
        print("Sample paper-code pairs:")
        for i, pair in enumerate(pairs[:5], 1):
            paper = pair['paper']
            repos = pair['repositories']
            print(f"\n{i}. {paper['title']}")
            print(f"   ArXiv: {paper['arxiv_id']}")
            print(f"   Year: {paper.get('year', 'N/A')}")
            print(f"   Repositories: {len(repos)}")
            if repos:
                top_repo = max(repos, key=lambda r: r.get('stars', 0))
                print(f"   Top repo: {top_repo['name']} ({top_repo['stars']} stars)")
    else:
        print("No pairs collected!")


if __name__ == "__main__":
    main()
