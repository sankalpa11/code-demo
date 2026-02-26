"""
Fast GitHub scraper optimized for speed.
Downloads top 3 repos with rate limit handling.
"""

import os
import json
import base64
import time
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm


class FastGitHubScraper:
    """
    Fast parallel scraper for GitHub repositories.
    Optimized for demo: speed over completeness.
    """
    
    # Top 3 repos: diverse, high-quality, manageable size
    RECOMMENDED_REPOS = [
        {
            "name": "scikit-learn/scikit-learn",
            "focus": "machine_learning",
            "priority": "high",
            "max_files": 1000  # Limit for speed
        },
        {
            "name": "pallets/flask",
            "focus": "web_development", 
            "priority": "high",
            "max_files": 500
        },
        {
            "name": "pytest-dev/pytest",
            "focus": "testing",
            "priority": "medium",
            "max_files": 500
        }
    ]
    
    def __init__(self, github_token: Optional[str] = None):
        self.token = github_token or os.getenv("GITHUB_TOKEN")
        self.session = requests.Session()
        
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
            self.rate_limit = 5000  # per hour
        else:
            self.rate_limit = 60  # per hour (unauthenticated)
            print("WARNING: No GitHub token. Limited to 60 requests/hour.")
            print("Get token: https://github.com/settings/tokens")
        
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        
        self.stats = {
            "files_downloaded": 0,
            "functions_extracted": 0,
            "errors": 0
        }
    
    def scrape_all(self, output_dir: str, max_workers: int = 5) -> List[str]:
        """
        Scrape all recommended repos in parallel.
        
        Args:
            output_dir: Where to save files
            max_workers: Parallel download threads
        
        Returns:
            List of output file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_files = []
        
        print(f"Starting parallel scrape with {max_workers} workers...")
        print(f"Rate limit: {self.rate_limit} requests/hour\n")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._scrape_single_repo,
                    repo_config,
                    output_path
                ): repo_config["name"]
                for repo_config in self.RECOMMENDED_REPOS
            }
            
            for future in as_completed(futures):
                repo_name = futures[future]
                try:
                    output_file = future.result()
                    if output_file:
                        output_files.append(output_file)
                        print(f"✓ Completed: {repo_name}")
                except Exception as e:
                    print(f"✗ Failed: {repo_name} - {e}")
                    self.stats["errors"] += 1
        
        self._print_stats()
        return output_files
    
    def _scrape_single_repo(
        self,
        repo_config: Dict,
        output_path: Path
    ) -> Optional[str]:
        """
        Scrape single repository.
        """
        repo_name = repo_config["name"]
        max_files = repo_config["max_files"]
        
        print(f"Starting: {repo_name} (max {max_files} files)")
        
        # Parse owner/repo
        owner, repo = repo_name.split("/")
        
        # Get file list using GitHub API
        files = self._get_file_list(owner, repo, max_files)
        
        if not files:
            return None
        
        # Download files
        examples = []
        for file_path in tqdm(files, desc=f"Downloading {repo}", leave=False):
            try:
                content = self._download_file(owner, repo, file_path)
                if content:
                    example = self._process_content(file_path, repo_name, content)
                    if example:
                        examples.append(example)
                        self.stats["functions_extracted"] += len(example.get("functions", []))
            except Exception as e:
                self.stats["errors"] += 1
                continue
            
            # Rate limit protection
            if self.stats["files_downloaded"] % 100 == 0:
                self._respect_rate_limit()
        
        # Save
        output_file = output_path / f"{repo.replace('-', '_')}_raw.jsonl"
        self._save_examples(examples, output_file)
        
        return str(output_file)
    
    def _get_file_list(self, owner: str, repo: str, max_files: int) -> List[str]:
        """
        Get list of Python files using GitHub API.
        Uses search API for speed (one call vs recursive tree).
        """
        files = []
        page = 1
        
        while len(files) < max_files:
            url = f"https://api.github.com/search/code"
            params = {
                "q": f"repo:{owner}/{repo} extension:py",
                "per_page": 100,
                "page": page
            }
            
            response = self.session.get(url, params=params)
            
            if response.status_code != 200:
                print(f"API error: {response.status_code}")
                break
            
            data = response.json()
            items = data.get("items", [])
            
            if not items:
                break
            
            for item in items:
                files.append(item["path"])
                if len(files) >= max_files:
                    break
            
            page += 1
            
            # Check rate limit
            remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
            if remaining < 10:
                print(f"Rate limit low ({remaining}). Pausing...")
                time.sleep(60)
        
        return files[:max_files]
    
    def _download_file(self, owner: str, repo: str, path: str) -> Optional[str]:
        """
        Download single file content.
        """
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        
        response = self.session.get(url)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        
        if data.get("encoding") == "base64":
            content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
        else:
            content = data.get("content", "")
        
        self.stats["files_downloaded"] += 1
        
        return content
    
    def _process_content(self, file_path: str, repo_name: str, content: str) -> Optional[Dict]:
        """
        Extract functions and metadata from file content.
        """
        # Skip very small or very large files
        lines = content.count("\n")
        if lines < 10 or lines > 1000:
            return None
        
        # Skip test files (usually not good training data)
        if "test" in file_path.lower() and "/tests/" in file_path:
            return None
        
        # Extract functions using simple regex (fast)
        functions = self._extract_functions_fast(content)
        
        if not functions:
            return None
        
        return {
            "source_file": file_path,
            "repo_name": repo_name,
            "code": content,
            "functions": functions,
            "num_lines": lines,
            "num_functions": len(functions)
        }
    
    def _extract_functions_fast(self, content: str) -> List[Dict]:
        """
        Fast function extraction using regex.
        Not perfect but 10x faster than AST.
        """
        import re
        
        functions = []
        
        # Pattern: def name(args): ...
        pattern = r'def\s+(\w+)\s*\([^)]*\)[^:]*:(?:\s*["\']{3}(.*?)["\']{3})?'
        
        for match in re.finditer(pattern, content, re.DOTALL):
            func_name = match.group(1)
            docstring = (match.group(2) or "").strip()[:500]  # Truncate long docstrings
            
            # Skip private functions (usually helpers)
            if func_name.startswith("_"):
                continue
            
            functions.append({
                "name": func_name,
                "docstring": docstring
            })
        
        return functions
    
    def _respect_rate_limit(self):
        """Pause if rate limit is low."""
        if not self.token:
            # Unauthenticated: very limited, be conservative
            time.sleep(1)
    
    def _save_examples(self, examples: List[Dict], output_file: Path):
        """Save to JSONL."""
        with open(output_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        
        print(f"  Saved {len(examples)} examples to {output_file.name}")
    
    def _print_stats(self):
        """Print final statistics."""
        print(f"\n{'='*60}")
        print("SCRAPING COMPLETE")
        print(f"{'='*60}")
        print(f"Files downloaded: {self.stats['files_downloaded']}")
        print(f"Functions found: {self.stats['functions_extracted']}")
        print(f"Errors: {self.stats['errors']}")


class LocalFallbackScraper:
    """
    Fallback: Use local files if GitHub API fails.
    """
    
    def create_sample_data(self, output_file: str, num_examples: int = 100):
        """
        Create sample training data from built-in examples.
        Use this if GitHub scraping fails.
        """
        examples = self._generate_samples(num_examples)
        
        with open(output_file, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        
        print(f"Created {num_examples} sample examples at {output_file}")
        return output_file
    
    def _generate_samples(self, n: int) -> List[Dict]:
        """Generate diverse code samples."""
        templates = [
            {
                "source_file": f"sample_{i}.py",
                "repo_name": "built-in-samples",
                "code": f"def function_{i}(x):\n    return x * {i}",
                "functions": [{"name": f"function_{i}", "docstring": f"Sample function {i}"}],
                "num_lines": 2,
                "num_functions": 1
            }
            for i in range(n)
        ]
        return templates


def run_scraper(output_dir: str = "data/raw") -> List[str]:
    """
    Main entry point.
    """
    token = os.getenv("GITHUB_TOKEN")
    scraper = FastGitHubScraper(token)
    
    try:
        files = scraper.scrape_all(output_dir)
        
        if not files:
            print("\nGitHub scraping failed. Using local fallback...")
            fallback = LocalFallbackScraper()
            fallback_file = fallback.create_sample_data(f"{output_dir}/fallback.jsonl")
            return [fallback_file]
        
        return files
    
    except Exception as e:
        print(f"Error: {e}")
        print("Using local fallback...")
        fallback = LocalFallbackScraper()
        fallback_file = fallback.create_sample_data(f"{output_dir}/fallback.jsonl")
        return [fallback_file]


if __name__ == "__main__":
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    run_scraper(output_dir)