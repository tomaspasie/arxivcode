"""
Day 4 Step 2: Parse paper_code_with_files.json
Extracts paper abstracts and code file contents from the JSON file
Creates contrastive pairs: (abstract, code_file_content)
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperCodeParser:
    """
    Parser for paper_code_with_files.json that extracts paper abstracts and code file contents
    Creates one pair per code file: (paper_abstract, code_file_content)
    """

    def __init__(self, json_path: str):
        """
        Initialize parser

        Args:
            json_path: Path to paper_code_with_files.json
        """
        self.json_path = Path(json_path)
        self.pairs = []

    def load_json(self) -> List[Dict]:
        """
        Load paper_code_with_files.json file

        Returns:
            List of paper-code pair dictionaries with code files
        """
        logger.info(f"Loading JSON file: {self.json_path}")
        if not self.json_path.exists():
            raise FileNotFoundError(f"File not found: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} papers with repositories")
        return data

    def extract_paper_text(self, paper: Dict) -> str:
        """
        Extract paper abstract text

        Args:
            paper: Paper dictionary with title, arxiv_id, abstract, etc.

        Returns:
            Paper text: "title + abstract"
        """
        title = paper.get("title") or ""
        title = title.strip() if title else ""
        
        abstract = paper.get("abstract") or ""
        abstract = abstract.strip() if abstract else ""

        # Build paper text: title + abstract
        if abstract:
            paper_text = f"{title} {abstract}"
        else:
            paper_text = title

        return paper_text.strip()

    def extract_code_file_content(self, code_file: Dict) -> str:
        """
        Extract code file content

        Args:
            code_file: Code file dictionary with path, content, language, etc.

        Returns:
            Code file content as string
        """
        content = code_file.get("content") or ""
        return content.strip()

    def parse(self) -> List[Tuple[str, str, Dict]]:
        """
        Parse JSON file and extract paper-code text pairs
        Creates one pair per code file: (paper_abstract, code_file_content)

        Returns:
            List of tuples: (paper_text, code_text, metadata)
            metadata contains: paper_id, repo_name, arxiv_id, code_file_path, etc.
        """
        logger.info("=" * 60)
        logger.info("Day 4 Step 2: Parsing paper_code_with_files.json")
        logger.info("=" * 60)

        # Load JSON
        data = self.load_json()

        pairs = []
        skipped_papers = 0
        skipped_files = 0
        total_code_files = 0

        for i, entry in enumerate(data, 1):
            paper = entry.get("paper", {})
            repositories = entry.get("repositories", [])

            # Skip if paper is missing required fields
            if not paper:
                logger.warning(f"Entry {i}: Skipping - missing paper")
                skipped_papers += 1
                continue

            # Extract paper text (title + abstract)
            paper_text = self.extract_paper_text(paper)

            if not paper_text:
                logger.warning(f"Entry {i}: Skipping - empty paper text (no title/abstract)")
                skipped_papers += 1
                continue

            # Process each repository
            for repo in repositories:
                # Skip if repository wasn't cloned or has no code files
                if not repo.get("cloned", False):
                    logger.debug(f"Entry {i}: Repo {repo.get('name', 'unknown')} not cloned, skipping")
                    continue

                code_files = repo.get("code_files", [])
                if not code_files:
                    logger.debug(f"Entry {i}: Repo {repo.get('name', 'unknown')} has no code files, skipping")
                    continue

                # Create a pair for each code file
                for code_file in code_files:
                    code_text = self.extract_code_file_content(code_file)

                    if not code_text:
                        skipped_files += 1
                        continue

                    total_code_files += 1

                    # Create metadata
                    metadata = {
                        "arxiv_id": paper.get("arxiv_id") or "",
                        "paper_title": paper.get("title") or "",
                        "repo_name": repo.get("name") or "",
                        "repo_url": repo.get("url") or "",
                        "code_file_path": code_file.get("path") or "",
                        "code_language": code_file.get("language") or "",
                        "code_file_extension": code_file.get("extension") or "",
                        "code_file_lines": code_file.get("lines", 0),
                        "pair_index": i,
                    }

                    pairs.append((paper_text, code_text, metadata))

                if i % 10 == 0:
                    logger.info(
                        f"Processed {i}/{len(data)} papers... ({len(pairs)} text pairs created from {total_code_files} code files)"
                    )

        logger.info("=" * 60)
        logger.info(f"Parsing complete!")
        logger.info(f"  Total papers processed: {len(data)}")
        logger.info(f"  Papers skipped: {skipped_papers}")
        logger.info(f"  Code files processed: {total_code_files}")
        logger.info(f"  Code files skipped: {skipped_files}")
        logger.info(f"  Text pairs created: {len(pairs)}")
        logger.info("=" * 60)

        self.pairs = pairs
        return pairs

    def filter_empty(
        self, pairs: List[Tuple[str, str, Dict]]
    ) -> List[Tuple[str, str, Dict]]:
        """
        Filter out pairs with empty text fields

        Args:
            pairs: List of (paper_text, code_text, metadata) tuples

        Returns:
            Filtered list
        """
        filtered = [
            (paper_text, code_text, metadata)
            for paper_text, code_text, metadata in pairs
            if paper_text.strip() and code_text.strip()
        ]

        logger.info(
            f"Filtered {len(pairs)} -> {len(filtered)} pairs (removed empty entries)"
        )
        return filtered

    def save_parsed_pairs(
        self, output_path: str, pairs: Optional[List[Tuple[str, str, Dict]]] = None
    ):
        """
        Save parsed pairs to JSON file

        Args:
            output_path: Path to save output JSON
            pairs: Pairs to save (uses self.pairs if None)
        """
        if pairs is None:
            pairs = self.pairs

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to list of dictionaries for JSON serialization
        output_data = [
            {
                "paper_text": paper_text,
                "code_text": code_text,
                "metadata": metadata,
            }
            for paper_text, code_text, metadata in pairs
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(output_data)} parsed pairs to {output_path}")


def parse_paper_code_pairs(
    json_path: str = "data/raw/papers/paper_code_with_files.json",
    output_path: Optional[str] = None,
) -> List[Tuple[str, str, Dict]]:
    """
    Convenience function to parse paper_code_with_files.json

    Args:
        json_path: Path to paper_code_with_files.json
        output_path: Optional path to save parsed pairs

    Returns:
        List of (paper_text, code_text, metadata) tuples
    """
    parser = PaperCodeParser(json_path)
    pairs = parser.parse()
    pairs = parser.filter_empty(pairs)

    if output_path:
        parser.save_parsed_pairs(output_path, pairs)

    return pairs


if __name__ == "__main__":
    # Example usage
    import sys

    # Parse paper_code_with_files.json (includes abstracts and code files)
    logger.info("Parsing paper_code_with_files.json...")
    pairs = parse_paper_code_pairs(
        json_path="data/raw/papers/paper_code_with_files.json",
        output_path="data/processed/parsed_pairs.json",
    )

    # Print summary
    print("\n" + "=" * 60)
    print("PARSING SUMMARY")
    print("=" * 60)
    print(f"Total pairs: {len(pairs)}")
    if pairs:
        print(f"\nSample pair:")
        paper_text, code_text, metadata = pairs[0]
        print(f"  Paper: {paper_text[:100]}...")
        print(f"  Code: {code_text[:100]}...")
        print(f"  ArXiv ID: {metadata['arxiv_id']}")
        print(f"  Repo: {metadata['repo_name']}")
    print("=" * 60)
