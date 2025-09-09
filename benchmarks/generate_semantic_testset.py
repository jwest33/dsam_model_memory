#!/usr/bin/env python3
"""
Generate semantic test dataset using LLM paraphrasing.
Creates a persistent test set for reproducible benchmarking.
"""

import sys
import json
import sqlite3
import random
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_memory.config import cfg


@dataclass
class SemanticTestCase:
    """A semantic similarity test case."""
    test_id: str
    original_memory_id: str
    original_text: str
    paraphrased_query: str
    query_type: str  # 'paraphrase', 'summary', 'question', 'expansion', 'abstraction'
    expected_relevant: List[str]  # Memory IDs that should be retrieved
    metadata: Dict


class SemanticTestGenerator:
    """Generate semantic test cases using LLM."""
    
    def __init__(self):
        """Initialize generator."""
        # Database connection
        self.db_conn = sqlite3.connect(cfg.db_path)
        self.db_conn.row_factory = sqlite3.Row
        
        # LLM configuration
        self.llm_url = cfg.get('llm_base_url', 'http://localhost:8001/v1')
        self.llm_model = cfg.get('llm_model', 'local-model')
        
        # Output directory
        self.output_dir = Path("test_data")
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_paraphrase(self, text: str, style: str = "paraphrase") -> Optional[str]:
        """Generate semantic variation of text using LLM."""
        
        prompts = {
            "paraphrase": f"""Rewrite the following text using completely different words while preserving the exact same meaning. 
Rules:
- Do NOT use any of the key nouns or verbs from the original
- Express the same idea in a different way
- Maintain the same factual content

Original text: {text}

Paraphrase:""",
            
            "summary": f"""Create a brief summary that captures the core meaning of this text:

Text: {text}

Summary:""",
            
            "question": f"""What question would this text answer? Create a natural question that someone might ask to get this information:

Text: {text}

Question:""",
            
            "expansion": f"""Elaborate on this concept using different terminology and adding relevant context:

Text: {text}

Expanded version:""",
            
            "abstraction": f"""Express this same concept at a higher level of abstraction, focusing on the general principle rather than specifics:

Text: {text}

Abstract version:"""
        }
        
        prompt = prompts.get(style, prompts["paraphrase"])
        
        try:
            response = requests.post(
                f"{self.llm_url}/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": [
                        {"role": "system", "content": "You are an expert at creating semantic variations of text while preserving meaning."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 200
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                print(f"LLM request failed with status {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return None
    
    def find_semantically_similar_memories(self, memory_id: str, text: str, session_id: str) -> List[str]:
        """Find memories that are semantically similar to use as expected results."""
        
        similar_memories = []
        cursor = self.db_conn.cursor()
        
        # 1. Other memories from the same conversation (high semantic relevance)
        cursor.execute("""
            SELECT memory_id 
            FROM memories 
            WHERE session_id = ? AND memory_id != ?
            ORDER BY RANDOM()
            LIMIT 3
        """, (session_id, memory_id))
        
        same_session = [row['memory_id'] for row in cursor.fetchall()]
        similar_memories.extend(same_session)
        
        # 2. Extract key topics using simple heuristics
        # Look for capitalized words (likely proper nouns/topics)
        words = text.split()
        key_terms = [w for w in words if w[0].isupper() and len(w) > 3][:3]
        
        # Find memories with similar key terms
        for term in key_terms:
            cursor.execute("""
                SELECT memory_id
                FROM memories
                WHERE raw_text LIKE ? AND memory_id != ? AND session_id != ?
                LIMIT 2
            """, (f'%{term}%', memory_id, session_id))
            
            for row in cursor.fetchall():
                if row['memory_id'] not in similar_memories:
                    similar_memories.append(row['memory_id'])
        
        return similar_memories[:5]  # Return up to 5 similar memories
    
    def generate_test_cases(self, num_cases: int = 50) -> List[SemanticTestCase]:
        """Generate diverse semantic test cases."""
        
        print(f"Generating {num_cases} semantic test cases...")
        test_cases = []
        cursor = self.db_conn.cursor()
        
        # Query types to generate
        query_types = ["paraphrase", "summary", "question", "expansion", "abstraction"]
        
        # Select diverse source memories
        cursor.execute("""
            SELECT memory_id, raw_text, session_id, who_id
            FROM memories
            WHERE LENGTH(raw_text) BETWEEN 50 AND 500
            ORDER BY RANDOM()
            LIMIT ?
        """, (num_cases * 2,))  # Get extra in case some fail
        
        memories = cursor.fetchall()
        
        for i, mem in enumerate(memories[:num_cases]):
            # Choose query type (rotate through types)
            query_type = query_types[i % len(query_types)]
            
            print(f"  [{i+1}/{num_cases}] Generating {query_type} for memory {mem['memory_id'][:20]}...")
            
            # Generate the semantic variation
            variation = self.generate_paraphrase(mem['raw_text'], query_type)
            
            if variation:
                # Find semantically similar memories for ground truth
                similar = self.find_semantically_similar_memories(
                    mem['memory_id'],
                    mem['raw_text'],
                    mem['session_id']
                )
                
                # Primary expected result is always the original memory
                expected = [mem['memory_id']] + similar
                
                test_case = SemanticTestCase(
                    test_id=f"semantic_test_{i:04d}",
                    original_memory_id=mem['memory_id'],
                    original_text=mem['raw_text'],
                    paraphrased_query=variation,
                    query_type=query_type,
                    expected_relevant=expected,
                    metadata={
                        "session_id": mem['session_id'],
                        "who_id": mem['who_id'],
                        "original_length": len(mem['raw_text']),
                        "query_length": len(variation),
                        "generated_at": datetime.now().isoformat()
                    }
                )
                
                test_cases.append(test_case)
                
                # Show example
                if i < 3:
                    print(f"    Original: '{mem['raw_text'][:80]}...'")
                    print(f"    {query_type.capitalize()}: '{variation[:80]}...'")
        
        print(f"Successfully generated {len(test_cases)} test cases")
        return test_cases
    
    def save_test_set(self, test_cases: List[SemanticTestCase], filename: str = None):
        """Save test cases to JSON file."""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"semantic_testset_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert to serializable format
        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_cases": len(test_cases),
                "query_types": list(set(tc.query_type for tc in test_cases)),
                "database": cfg.db_path
            },
            "test_cases": [asdict(tc) for tc in test_cases]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Test set saved to: {filepath}")
        
        # Also save a human-readable summary
        summary_path = self.output_dir / f"{filename.replace('.json', '_summary.txt')}"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("SEMANTIC TEST SET SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {data['metadata']['generated_at']}\n")
            f.write(f"Total cases: {len(test_cases)}\n\n")
            
            # Count by type
            type_counts = {}
            for tc in test_cases:
                type_counts[tc.query_type] = type_counts.get(tc.query_type, 0) + 1
            
            f.write("Query type distribution:\n")
            for qtype, count in type_counts.items():
                f.write(f"  {qtype}: {count}\n")
            
            f.write("\n" + "-"*60 + "\n")
            f.write("SAMPLE TEST CASES\n")
            f.write("-"*60 + "\n\n")
            
            for tc in test_cases[:5]:
                f.write(f"Test ID: {tc.test_id}\n")
                f.write(f"Type: {tc.query_type}\n")
                f.write(f"Original ({len(tc.original_text)} chars): {tc.original_text[:150]}...\n")
                f.write(f"Query ({len(tc.paraphrased_query)} chars): {tc.paraphrased_query[:150]}...\n")
                f.write(f"Expected results: {len(tc.expected_relevant)} memories\n")
                f.write("\n")
        
        print(f"✓ Summary saved to: {summary_path}")
        
        return filepath
    
    def load_test_set(self, filename: str) -> List[SemanticTestCase]:
        """Load test cases from JSON file."""
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_cases = []
        for tc_dict in data['test_cases']:
            test_cases.append(SemanticTestCase(**tc_dict))
        
        print(f"Loaded {len(test_cases)} test cases from {filepath}")
        return test_cases


def main():
    """Generate semantic test set."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate semantic similarity test dataset')
    parser.add_argument('--cases', type=int, default=50,
                       help='Number of test cases to generate')
    parser.add_argument('--output', '-o', 
                       help='Output filename (default: timestamped)')
    
    args = parser.parse_args()
    
    generator = SemanticTestGenerator()
    
    # Generate test cases
    test_cases = generator.generate_test_cases(num_cases=args.cases)
    
    # Save to file
    if test_cases:
        filepath = generator.save_test_set(test_cases, args.output)
        
        print("\n" + "="*60)
        print("TEST SET GENERATION COMPLETE")
        print("="*60)
        print(f"Generated {len(test_cases)} test cases")
        print(f"Saved to: {filepath}")
        print("\nUse this file with semantic_similarity_eval.py to run benchmarks")


if __name__ == "__main__":
    main()
