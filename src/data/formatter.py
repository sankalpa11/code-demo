"""
Convert scraped code to instruction-response pairs for training.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional


class CodeFormatter:
    """
    Format raw code into training pairs for Coder and Reviewer agents.
    """
    
    # Instruction templates
    CODER_TEMPLATES = [
        "Write a Python function to {description}",
        "Implement {description} in Python",
        "Create a function that {description}",
        "Write code to {description}",
        "Implement the following: {description}",
    ]
    
    def __init__(self):
        self.stats = {"coder": 0, "reviewer": 0}
    
    def format_file(self, input_file: str, output_dir: str):
        """
        Convert raw JSONL to Coder and Reviewer training files.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load raw data
        with open(input_file) as f:
            examples = [json.loads(line) for line in f]
        
        print(f"Loaded {len(examples)} examples from {input_file}")
        
        coder_pairs = []
        reviewer_pairs = []
        
        for ex in examples:
            # Create Coder pair (instruction -> code)
            coder = self._create_coder_pair(ex)
            if coder:
                coder_pairs.append(coder)
            
            # Create Reviewer pair (code -> review)
            reviewer = self._create_reviewer_pair(ex)
            if reviewer:
                reviewer_pairs.append(reviewer)
        
        # Save
        base_name = Path(input_file).stem.replace("_raw", "")
        
        coder_file = output_path / f"{base_name}_coder.jsonl"
        reviewer_file = output_path / f"{base_name}_reviewer.jsonl"
        
        self._save(coder_pairs, coder_file)
        self._save(reviewer_pairs, reviewer_file)
        
        return str(coder_file), str(reviewer_file)
    
    def _create_coder_pair(self, ex: Dict) -> Optional[Dict]:
        """Create instruction-code pair."""
        funcs = ex.get("functions", [])
        if not funcs:
            return None
        
        # Pick function with docstring
        func = random.choice([f for f in funcs if f.get("docstring")] or funcs)
        
        docstring = func.get("docstring", "")
        func_name = func.get("name", "")
        
        # Create description
        if docstring:
            desc = docstring.split(".")[0][:100]
        else:
            desc = func_name.replace("_", " ")
        
        instruction = random.choice(self.CODER_TEMPLATES).format(description=desc)
        
        # Extract function code
        code = self._extract_function(ex["code"], func_name)
        
        self.stats["coder"] += 1
        
        return {
            "instruction": instruction,
            "code": code or ex["code"][:500],
            "source": ex["source_file"]
        }
    
    def _create_reviewer_pair(self, ex: Dict) -> Optional[Dict]:
        """Create code-review pair."""
        code = ex["code"]
        
        # Skip very short files
        if len(code) < 200:
            return None
        
        # Truncate long files
        if len(code) > 1500:
            code = code[:1500] + "\n# ..."
        
        # Generate review
        review = self._generate_review(code, ex)
        
        self.stats["reviewer"] += 1
        
        return {
            "code": code,
            "review": review,
            "source": ex["source_file"]
        }
    
    def _extract_function(self, full_code: str, func_name: str) -> str:
        """Extract single function from file."""
        pattern = rf"(def\s+{func_name}\s*\([^)]*\).*?)(?=\ndef\s+|\nclass\s+|$)"
        match = re.search(pattern, full_code, re.DOTALL)
        return match.group(1).strip()[:1000] if match else ""
    
    def _generate_review(self, code: str, ex: Dict) -> str:
        """Generate code review."""
        issues = []
        positives = []
        
        # Check docstrings
        if '"""' in code or "'''" in code:
            positives.append("Has docstrings")
        else:
            issues.append("Missing docstrings")
        
        # Check type hints
        if "->" in code or ":" in code:
            positives.append("Uses type hints")
        else:
            issues.append("Add type hints")
        
        # Check error handling
        if "try:" in code:
            positives.append("Has error handling")
        else:
            issues.append("Add error handling")
        
        # Complexity
        if ex.get("num_lines", 0) > 100:
            issues.append("Consider splitting long function")
        
        # Build review
        parts = []
        if positives:
            parts.append("Strengths: " + ", ".join(positives))
        if issues:
            parts.append("Improve: " + ", ".join(issues))
        
        return " | ".join(parts) if parts else "Good code structure"
    
    def _save(self, pairs: List[Dict], filepath: Path):
        """Save to JSONL."""
        with open(filepath, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"Saved {len(pairs)} pairs to {filepath}")


def merge_and_split(
    coder_files: List[str],
    reviewer_files: List[str],
    output_dir: str,
    train_ratio: float = 0.9
):
    """
    Merge all files and split into train/val.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Merge Coder
    all_coder = []
    for f in coder_files:
        with open(f) as fp:
            all_coder.extend([json.loads(l) for l in fp])
    
    # Merge Reviewer
    all_reviewer = []
    for f in reviewer_files:
        with open(f) as fp:
            all_reviewer.extend([json.loads(l) for l in fp])
    
    # Shuffle
    random.shuffle(all_coder)
    random.shuffle(all_reviewer)
    
    # Split
    coder_split = int(len(all_coder) * train_ratio)
    reviewer_split = int(len(all_reviewer) * train_ratio)
    
    # Save final files
    files = {
        "coder_train.jsonl": all_coder[:coder_split],
        "coder_val.jsonl": all_coder[coder_split:],
        "reviewer_train.jsonl": all_reviewer[:reviewer_split],
        "reviewer_val.jsonl": all_reviewer[reviewer_split:],
    }
    
    for name, data in files.items():
        with open(output_path / name, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"{name}: {len(data)} examples")
    
    return output_path


if __name__ == "__main__":
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/scikit_learn_raw.jsonl"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    
    formatter = CodeFormatter()
    formatter.format_file(input_file, output_dir)