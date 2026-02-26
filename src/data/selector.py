"""
IFD (Instruction Following Difficulty) + KMeans selection.
Selects optimal training subset from large codebase.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class ScoredExample:
    """Single example with IFD score."""
    instruction: str
    code: str
    cluster: int = 0
    ifd_score: float = 0.0


class IFDKMeansSelector:
    """
    Selects high-quality, diverse training data.
    
    1. KMeans clustering (preserve problem type diversity)
    2. IFD scoring (select complex, valuable examples)
    """
    
    def __init__(
        self,
        base_model_name: str = "deepseek-ai/deepseek-coder-6.7b-base",
        n_clusters: int = 10,
        sample_rate: float = 0.4,
        device: str = "cuda"
    ):
        self.n_clusters = n_clusters
        self.sample_rate = sample_rate
        
        # Embedding model for KMeans (fast, lightweight)
        print("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load base model for IFD calculation
        print(f"Loading base model for IFD: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Quantized model for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.model.eval()
        
        print("Models loaded")
    
    def calculate_perplexity(self, text: str, context: str = None) -> float:
        """
        Calculate perplexity of generating text.
        Lower = more predictable (model finds it easy).
        Higher = less predictable (model finds it hard).
        """
        # Format: instruction + code if context provided
        if context:
            full_text = f"{context}\n\n{text}"
        else:
            full_text = text
        
        # Tokenize
        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Calculate loss
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["input_ids"]
            )
            loss = outputs.loss
        
        # Perplexity = exp(loss)
        perplexity = torch.exp(loss).item()
        return perplexity
    
    def calculate_ifd(self, example: Dict) -> float:
        """
        Calculate IFD score.
        
        IFD = PPL(code | instruction) / PPL(code)
        
        Higher IFD = instruction helps less = more complex/valuable
        """
        instruction = example.get('instruction', example.get('code', '')[:100])
        code = example.get('code', example.get('review', ''))
        
        # Conditional: PPL(code | instruction)
        ppl_conditional = self.calculate_perplexity(code, instruction)
        
        # Unconditional: PPL(code alone)
        ppl_unconditional = self.calculate_perplexity(code, None)
        
        # IFD ratio
        ifd_score = ppl_conditional / ppl_unconditional
        
        return ifd_score
    
    def cluster_examples(self, examples: List[Dict]) -> List[ScoredExample]:
        """
        Cluster examples by instruction similarity.
        Ensures diversity across problem types.
        """
        print(f"\nClustering {len(examples)} examples into {self.n_clusters} groups...")
        
        # Extract instructions (or code if no instruction)
        instructions = [
            ex.get('instruction', ex.get('code', '')[:200]) 
            for ex in examples
        ]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedder.encode(instructions, show_progress_bar=True)
        
        # KMeans clustering
        print(f"Running KMeans...")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Create scored examples
        scored = []
        for i, example in enumerate(examples):
            scored.append(ScoredExample(
                instruction=example.get('instruction', ''),
                code=example.get('code', example.get('review', '')),
                cluster=int(cluster_labels[i])
            ))
        
        # Print distribution
        from collections import Counter
        dist = Counter(cluster_labels)
        print(f"Cluster distribution: {dict(dist)}")
        
        return scored
    
    def calculate_all_ifd(self, examples: List[ScoredExample]) -> List[ScoredExample]:
        """
        Calculate IFD scores for all examples.
        This is SLOW (requires model inference).
        """
        print(f"\nCalculating IFD for {len(examples)} examples...")
        print("This takes ~1 minute per 10 examples...")
        
        for i, example in enumerate(examples):
            ex_dict = {
                'instruction': example.instruction,
                'code': example.code
            }
            example.ifd_score = self.calculate_ifd(ex_dict)
            
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(examples)} done (last IFD: {example.ifd_score:.3f})")
        
        return examples
    
    def select_top_examples(self, examples: List[ScoredExample]) -> List[Dict]:
        """
        Select top sample_rate% from each cluster.
        """
        print(f"\nSelecting top {self.sample_rate*100:.0f}% from each cluster...")
        
        selected = []
        
        for cluster_id in range(self.n_clusters):
            # Get examples in this cluster
            cluster_examples = [e for e in examples if e.cluster == cluster_id]
            
            if not cluster_examples:
                continue
            
            # Sort by IFD (descending - most complex first)
            cluster_examples.sort(key=lambda x: x.ifd_score, reverse=True)
            
            # Select top sample_rate%
            n_select = max(1, int(len(cluster_examples) * self.sample_rate))
            selected.extend(cluster_examples[:n_select])
            
            print(f"  Cluster {cluster_id}: {len(cluster_examples)} total, "
                  f"selected {n_select} (IFD: {cluster_examples[0].ifd_score:.3f} - "
                  f"{cluster_examples[n_select-1].ifd_score:.3f})")
        
        # Convert back to dict
        result = []
        for ex in selected:
            result.append({
                'instruction': ex.instruction,
                'code': ex.code,
                'cluster': ex.cluster,
                'ifd_score': ex.ifd_score
            })
        
        return result
    
    def select(self, input_file: str, output_file: str) -> Dict:
        """
        Main entry point: run full IFD+KMeans selection.
        """
        # Load data
        print(f"Loading: {input_file}")
        with open(input_file) as f:
            raw_examples = [json.loads(line) for line in f]
        
        print(f"Total: {len(raw_examples)} examples")
        
        # Step 1: Cluster
        scored = self.cluster_examples(raw_examples)
        
        # Step 2: Calculate IFD (SLOW)
        scored = self.calculate_all_ifd(scored)
        
        # Step 3: Select
        selected = self.select_top_examples(scored)
        
        # Save
        print(f"\nSaving {len(selected)} selected examples to {output_file}")
        with open(output_file, 'w') as f:
            for ex in selected:
                f.write(json.dumps(ex) + '\n')
        
        # Stats
        stats = {
            'input': len(raw_examples),
            'output': len(selected),
            'rate': len(selected) / len(raw_examples),
            'ifd_mean': np.mean([e['ifd_score'] for e in selected])
        }
        
        print(f"\n{'='*60}")
        print("IFD+KMEANS COMPLETE")
        print(f"{'='*60}")
        print(f"Input:  {stats['input']}")
        print(f"Output: {stats['output']}")
        print(f"Rate:   {stats['rate']*100:.1f}%")
        print(f"Avg IFD: {stats['ifd_mean']:.3f}")
        
        return stats


class FastSelector:
    """
    Fast selection without IFD (random stratified sampling).
    Use when IFD is too slow.
    """
    
    def __init__(self, n_clusters: int = 10, sample_rate: float = 0.4):
        self.n_clusters = n_clusters
        self.sample_rate = sample_rate
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def select(self, input_file: str, output_file: str) -> Dict:
        """Fast random selection within clusters."""
        import random
        
        with open(input_file) as f:
            examples = [json.loads(line) for line in f]
        
        # Cluster
        instructions = [ex.get('instruction', ex.get('code', '')[:200]) for ex in examples]
        embeddings = self.embedder.encode(instructions)
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        for i, ex in enumerate(examples):
            ex['cluster'] = int(clusters[i])
        
        # Random sample from each cluster
        selected = []
        for c in range(self.n_clusters):
            cluster_ex = [e for e in examples if e['cluster'] == c]
            n_select = max(1, int(len(cluster_ex) * self.sample_rate))
            selected.extend(random.sample(cluster_ex, n_select))
        
        # Save
        with open(output_file, 'w') as f:
            for ex in selected:
                f.write(json.dumps(ex) + '\n')
        
        return {
            'input': len(examples),
            'output': len(selected),
            'method': 'fast_random'
        }


def run_selection(agent_type: str, use_ifd: bool = True):
    """
    Run selection for specific agent.
    """
    input_file = f"data/final/{agent_type}_train.jsonl"
    output_file = f"data/selected/{agent_type}_selected.jsonl"
    
    # Create output dir
    Path("data/selected").mkdir(exist_ok=True)
    
    if use_ifd:
        print(f"Running IFD+KMeans for {agent_type}...")
        selector = IFDKMeansSelector()
    else:
        print(f"Running fast selection for {agent_type}...")
        selector = FastSelector()
    
    stats = selector.select(input_file, output_file)
    return output_file


if __name__ == "__main__":
    import sys
    
    use_ifd = "--fast" not in sys.argv
    
    # Run for both agents
    run_selection("coder", use_ifd=use_ifd)
    run_selection("reviewer", use_ifd=use_ifd)