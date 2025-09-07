#!/usr/bin/env python3
"""
Analyze LMSYS data imported into JAM memory system.
Provides comprehensive metrics on memory quality, retrieval performance, and system behavior.
"""

import sys
import time
import json
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import pandas as pd

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_memory.router import MemoryRouter
from agentic_memory.config import cfg
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.retrieval import HybridRetriever


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    total_memories: int
    unique_sessions: int
    memory_distribution: Dict[str, int]
    retrieval_metrics: Dict[str, float]
    quality_scores: Dict[str, float]
    temporal_distribution: Dict[str, int]
    actor_distribution: Dict[str, int]
    performance_metrics: Dict[str, float]


class LMSYSAnalyzer:
    """Analyze LMSYS data in JAM memory system."""
    
    def __init__(self):
        """Initialize analyzer with memory components."""
        self.store = MemoryStore(cfg.db_path)
        self.index = FaissIndex(dim=384, index_path=cfg.index_path)
        self.router = MemoryRouter(self.store, self.index)
        self.retriever = HybridRetriever(self.store, self.index)
        
        # Connect directly to database for analysis queries
        self.db_conn = sqlite3.connect(cfg.db_path)
        self.db_conn.row_factory = sqlite3.Row
        
    def analyze_memory_distribution(self) -> Dict[str, Any]:
        """Analyze the distribution of memories in the system."""
        cursor = self.db_conn.cursor()
        
        # Total memories
        cursor.execute("SELECT COUNT(*) as count FROM memories")
        total_memories = cursor.fetchone()['count']
        
        # Unique sessions
        cursor.execute("SELECT COUNT(DISTINCT session_id) as count FROM memories")
        unique_sessions = cursor.fetchone()['count']
        
        # Memory types distribution
        cursor.execute("""
            SELECT event_type, COUNT(*) as count 
            FROM memories 
            GROUP BY event_type
            ORDER BY count DESC
        """)
        event_types = {row['event_type']: row['count'] for row in cursor.fetchall()}
        
        # Actor distribution
        cursor.execute("""
            SELECT actor, COUNT(*) as count 
            FROM memories 
            GROUP BY actor
            ORDER BY count DESC
            LIMIT 20
        """)
        actors = {row['actor']: row['count'] for row in cursor.fetchall()}
        
        # Session size distribution
        cursor.execute("""
            SELECT session_id, COUNT(*) as memory_count
            FROM memories
            GROUP BY session_id
            ORDER BY memory_count DESC
        """)
        session_sizes = [row['memory_count'] for row in cursor.fetchall()]
        
        return {
            'total_memories': total_memories,
            'unique_sessions': unique_sessions,
            'event_types': event_types,
            'actors': actors,
            'session_sizes': {
                'mean': np.mean(session_sizes) if session_sizes else 0,
                'median': np.median(session_sizes) if session_sizes else 0,
                'max': max(session_sizes) if session_sizes else 0,
                'min': min(session_sizes) if session_sizes else 0
            }
        }
    
    def analyze_5w1h_quality(self) -> Dict[str, float]:
        """Analyze the quality of 5W1H extraction."""
        cursor = self.db_conn.cursor()
        
        # Sample memories for analysis
        cursor.execute("""
            SELECT who, what, when_occurred, where_occurred, why, how, content
            FROM memories
            ORDER BY RANDOM()
            LIMIT 1000
        """)
        
        samples = cursor.fetchall()
        if not samples:
            return {}
        
        quality_metrics = {
            'who_filled': 0,
            'what_filled': 0,
            'when_filled': 0,
            'where_filled': 0,
            'why_filled': 0,
            'how_filled': 0,
            'avg_who_length': [],
            'avg_what_length': [],
            'avg_why_length': [],
            'avg_how_length': [],
            'complete_5w1h': 0
        }
        
        for row in samples:
            # Check if fields are filled (not null/empty)
            if row['who'] and row['who'].strip():
                quality_metrics['who_filled'] += 1
                quality_metrics['avg_who_length'].append(len(row['who']))
            
            if row['what'] and row['what'].strip():
                quality_metrics['what_filled'] += 1
                quality_metrics['avg_what_length'].append(len(row['what']))
            
            if row['when_occurred'] and row['when_occurred'].strip():
                quality_metrics['when_filled'] += 1
            
            if row['where_occurred'] and row['where_occurred'].strip():
                quality_metrics['where_filled'] += 1
            
            if row['why'] and row['why'].strip():
                quality_metrics['why_filled'] += 1
                quality_metrics['avg_why_length'].append(len(row['why']))
            
            if row['how'] and row['how'].strip():
                quality_metrics['how_filled'] += 1
                quality_metrics['avg_how_length'].append(len(row['how']))
            
            # Check if all 5W1H fields are complete
            if all([
                row['who'] and row['who'].strip(),
                row['what'] and row['what'].strip(),
                row['when_occurred'] and row['when_occurred'].strip(),
                row['where_occurred'] and row['where_occurred'].strip(),
                row['why'] and row['why'].strip(),
                row['how'] and row['how'].strip()
            ]):
                quality_metrics['complete_5w1h'] += 1
        
        # Convert to percentages and averages
        total = len(samples)
        return {
            'who_coverage': 100 * quality_metrics['who_filled'] / total,
            'what_coverage': 100 * quality_metrics['what_filled'] / total,
            'when_coverage': 100 * quality_metrics['when_filled'] / total,
            'where_coverage': 100 * quality_metrics['where_filled'] / total,
            'why_coverage': 100 * quality_metrics['why_filled'] / total,
            'how_coverage': 100 * quality_metrics['how_filled'] / total,
            'complete_5w1h_pct': 100 * quality_metrics['complete_5w1h'] / total,
            'avg_who_length': np.mean(quality_metrics['avg_who_length']) if quality_metrics['avg_who_length'] else 0,
            'avg_what_length': np.mean(quality_metrics['avg_what_length']) if quality_metrics['avg_what_length'] else 0,
            'avg_why_length': np.mean(quality_metrics['avg_why_length']) if quality_metrics['avg_why_length'] else 0,
            'avg_how_length': np.mean(quality_metrics['avg_how_length']) if quality_metrics['avg_how_length'] else 0
        }
    
    def test_retrieval_performance(self, num_queries: int = 100) -> Dict[str, float]:
        """Test retrieval performance with random queries."""
        cursor = self.db_conn.cursor()
        
        # Get random memory contents to use as queries
        cursor.execute("""
            SELECT content, session_id, actor
            FROM memories
            ORDER BY RANDOM()
            LIMIT ?
        """, (num_queries,))
        
        test_queries = cursor.fetchall()
        if not test_queries:
            return {}
        
        retrieval_times = []
        retrieval_counts = []
        relevance_scores = []
        
        for query_row in test_queries:
            # Extract key terms from content for query
            content = query_row['content']
            query_terms = content[:100] if len(content) > 100 else content
            
            # Time the retrieval
            start_time = time.time()
            results = self.retriever.retrieve(
                query=query_terms,
                limit=10,
                session_id=query_row['session_id']
            )
            retrieval_time = time.time() - start_time
            
            retrieval_times.append(retrieval_time)
            retrieval_counts.append(len(results))
            
            # Calculate relevance (simple check if same session/actor appears in results)
            relevant = sum(1 for r in results if r.session_id == query_row['session_id'])
            relevance_scores.append(relevant / len(results) if results else 0)
        
        return {
            'avg_retrieval_time_ms': 1000 * np.mean(retrieval_times),
            'median_retrieval_time_ms': 1000 * np.median(retrieval_times),
            'max_retrieval_time_ms': 1000 * np.max(retrieval_times),
            'min_retrieval_time_ms': 1000 * np.min(retrieval_times),
            'avg_results_returned': np.mean(retrieval_counts),
            'avg_relevance_score': np.mean(relevance_scores),
            'queries_per_second': len(retrieval_times) / sum(retrieval_times) if sum(retrieval_times) > 0 else 0
        }
    
    def analyze_content_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in memory content."""
        cursor = self.db_conn.cursor()
        
        # Content length distribution
        cursor.execute("SELECT LENGTH(content) as len FROM memories")
        lengths = [row['len'] for row in cursor.fetchall()]
        
        # Language distribution (from metadata)
        cursor.execute("""
            SELECT 
                json_extract(metadata, '$.language') as language,
                COUNT(*) as count
            FROM memories
            WHERE json_extract(metadata, '$.language') IS NOT NULL
            GROUP BY language
            ORDER BY count DESC
            LIMIT 10
        """)
        languages = {row['language']: row['count'] for row in cursor.fetchall()}
        
        # Model distribution
        cursor.execute("""
            SELECT 
                json_extract(metadata, '$.model') as model,
                COUNT(*) as count
            FROM memories
            WHERE json_extract(metadata, '$.model') IS NOT NULL
            GROUP BY model
            ORDER BY count DESC
            LIMIT 20
        """)
        models = {row['model']: row['count'] for row in cursor.fetchall()}
        
        # Turn distribution
        cursor.execute("""
            SELECT 
                CAST(json_extract(metadata, '$.turn_index') as INTEGER) as turn,
                COUNT(*) as count
            FROM memories
            WHERE json_extract(metadata, '$.turn_index') IS NOT NULL
            GROUP BY turn
            ORDER BY turn
            LIMIT 50
        """)
        turn_distribution = {row['turn']: row['count'] for row in cursor.fetchall()}
        
        return {
            'content_length': {
                'mean': np.mean(lengths) if lengths else 0,
                'median': np.median(lengths) if lengths else 0,
                'max': max(lengths) if lengths else 0,
                'min': min(lengths) if lengths else 0,
                'std': np.std(lengths) if lengths else 0
            },
            'languages': languages,
            'models': models,
            'turn_distribution': turn_distribution
        }
    
    def analyze_index_quality(self) -> Dict[str, float]:
        """Analyze FAISS index quality and coverage."""
        cursor = self.db_conn.cursor()
        
        # Count memories in database
        cursor.execute("SELECT COUNT(*) as count FROM memories")
        db_count = cursor.fetchone()['count']
        
        # Get index stats
        index_count = len(self.index.memory_id_to_index)
        
        # Sample embeddings for similarity analysis
        if index_count > 0:
            sample_size = min(100, index_count)
            sample_ids = list(self.index.memory_id_to_index.keys())[:sample_size]
            
            similarities = []
            for memory_id in sample_ids:
                # Get k nearest neighbors
                similar = self.index.search_similar(
                    memory_id=memory_id,
                    k=10
                )
                if similar:
                    # Calculate average similarity score
                    avg_sim = np.mean([score for _, score in similar])
                    similarities.append(avg_sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0
            similarity_std = np.std(similarities) if similarities else 0
        else:
            avg_similarity = 0
            similarity_std = 0
        
        return {
            'memories_in_db': db_count,
            'memories_in_index': index_count,
            'index_coverage_pct': 100 * index_count / db_count if db_count > 0 else 0,
            'avg_similarity_score': avg_similarity,
            'similarity_std': similarity_std,
            'index_dimension': self.index.dim
        }
    
    def generate_report(self, output_file: Optional[str] = None) -> AnalysisResult:
        """Generate comprehensive analysis report."""
        print("\n" + "="*60)
        print("JAM MEMORY SYSTEM ANALYSIS - LMSYS DATA")
        print("="*60)
        
        # Memory distribution
        print("\n[1/5] Analyzing memory distribution...")
        distribution = self.analyze_memory_distribution()
        
        # 5W1H quality
        print("[2/5] Analyzing 5W1H extraction quality...")
        quality = self.analyze_5w1h_quality()
        
        # Retrieval performance
        print("[3/5] Testing retrieval performance...")
        retrieval = self.test_retrieval_performance()
        
        # Content patterns
        print("[4/5] Analyzing content patterns...")
        patterns = self.analyze_content_patterns()
        
        # Index quality
        print("[5/5] Analyzing index quality...")
        index_quality = self.analyze_index_quality()
        
        # Compile results
        result = AnalysisResult(
            total_memories=distribution['total_memories'],
            unique_sessions=distribution['unique_sessions'],
            memory_distribution=distribution,
            retrieval_metrics=retrieval,
            quality_scores=quality,
            temporal_distribution={},  # Not implemented for LMSYS
            actor_distribution=distribution['actors'],
            performance_metrics={
                **retrieval,
                **index_quality
            }
        )
        
        # Print report
        self._print_report(result, patterns)
        
        # Save to file if requested
        if output_file:
            self._save_report(result, patterns, output_file)
        
        return result
    
    def _print_report(self, result: AnalysisResult, patterns: Dict[str, Any]):
        """Print formatted analysis report."""
        print("\n" + "-"*60)
        print("MEMORY DISTRIBUTION")
        print("-"*60)
        print(f"Total Memories: {result.total_memories:,}")
        print(f"Unique Sessions: {result.unique_sessions:,}")
        print(f"Avg Memories/Session: {result.total_memories/result.unique_sessions:.1f}")
        
        print("\nEvent Type Distribution:")
        for event_type, count in result.memory_distribution.get('event_types', {}).items():
            pct = 100 * count / result.total_memories if result.total_memories > 0 else 0
            print(f"  {event_type}: {count:,} ({pct:.1f}%)")
        
        print("\nTop Actors:")
        for i, (actor, count) in enumerate(list(result.actor_distribution.items())[:5]):
            print(f"  {i+1}. {actor}: {count:,} memories")
        
        print("\n" + "-"*60)
        print("5W1H EXTRACTION QUALITY")
        print("-"*60)
        for field in ['who', 'what', 'when', 'where', 'why', 'how']:
            coverage = result.quality_scores.get(f'{field}_coverage', 0)
            print(f"{field.capitalize():8} Coverage: {coverage:.1f}%")
        print(f"\nComplete 5W1H: {result.quality_scores.get('complete_5w1h_pct', 0):.1f}%")
        
        print("\nAverage Field Lengths:")
        for field in ['who', 'what', 'why', 'how']:
            avg_len = result.quality_scores.get(f'avg_{field}_length', 0)
            if avg_len > 0:
                print(f"  {field.capitalize()}: {avg_len:.0f} chars")
        
        print("\n" + "-"*60)
        print("RETRIEVAL PERFORMANCE")
        print("-"*60)
        print(f"Avg Retrieval Time: {result.retrieval_metrics.get('avg_retrieval_time_ms', 0):.1f} ms")
        print(f"Median Retrieval Time: {result.retrieval_metrics.get('median_retrieval_time_ms', 0):.1f} ms")
        print(f"Queries per Second: {result.retrieval_metrics.get('queries_per_second', 0):.1f}")
        print(f"Avg Results Returned: {result.retrieval_metrics.get('avg_results_returned', 0):.1f}")
        print(f"Avg Relevance Score: {result.retrieval_metrics.get('avg_relevance_score', 0):.2f}")
        
        print("\n" + "-"*60)
        print("CONTENT ANALYSIS")
        print("-"*60)
        content_len = patterns.get('content_length', {})
        print(f"Content Length - Mean: {content_len.get('mean', 0):.0f}, Median: {content_len.get('median', 0):.0f}")
        print(f"Content Length - Range: [{content_len.get('min', 0)}, {content_len.get('max', 0)}]")
        
        print("\nTop Languages:")
        for lang, count in list(patterns.get('languages', {}).items())[:5]:
            print(f"  {lang}: {count:,}")
        
        print("\nTop Models:")
        for i, (model, count) in enumerate(list(patterns.get('models', {}).items())[:5]):
            print(f"  {i+1}. {model}: {count:,}")
        
        print("\n" + "-"*60)
        print("INDEX QUALITY")
        print("-"*60)
        print(f"Index Coverage: {result.performance_metrics.get('index_coverage_pct', 0):.1f}%")
        print(f"Memories in DB: {result.performance_metrics.get('memories_in_db', 0):,}")
        print(f"Memories in Index: {result.performance_metrics.get('memories_in_index', 0):,}")
        print(f"Avg Similarity Score: {result.performance_metrics.get('avg_similarity_score', 0):.3f}")
    
    def _save_report(self, result: AnalysisResult, patterns: Dict[str, Any], output_file: str):
        """Save analysis report to JSON file."""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'total_memories': result.total_memories,
            'unique_sessions': result.unique_sessions,
            'memory_distribution': result.memory_distribution,
            'quality_scores': result.quality_scores,
            'retrieval_metrics': result.retrieval_metrics,
            'performance_metrics': result.performance_metrics,
            'content_patterns': patterns
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n✓ Report saved to {output_file}")
    
    def plot_analysis(self):
        """Generate visualization plots for the analysis."""
        import matplotlib.pyplot as plt
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('JAM Memory System Analysis - LMSYS Data', fontsize=16)
        
        cursor = self.db_conn.cursor()
        
        # 1. Event type distribution
        cursor.execute("""
            SELECT event_type, COUNT(*) as count 
            FROM memories 
            GROUP BY event_type
        """)
        event_data = cursor.fetchall()
        if event_data:
            events = [row['event_type'] for row in event_data]
            counts = [row['count'] for row in event_data]
            axes[0, 0].pie(counts, labels=events, autopct='%1.1f%%')
            axes[0, 0].set_title('Event Type Distribution')
        
        # 2. Session size histogram
        cursor.execute("""
            SELECT COUNT(*) as memory_count
            FROM memories
            GROUP BY session_id
        """)
        session_sizes = [row['memory_count'] for row in cursor.fetchall()]
        if session_sizes:
            axes[0, 1].hist(session_sizes, bins=30, edgecolor='black')
            axes[0, 1].set_xlabel('Memories per Session')
            axes[0, 1].set_ylabel('Number of Sessions')
            axes[0, 1].set_title('Session Size Distribution')
        
        # 3. Content length distribution
        cursor.execute("SELECT LENGTH(content) as len FROM memories LIMIT 10000")
        lengths = [row['len'] for row in cursor.fetchall()]
        if lengths:
            axes[0, 2].hist(lengths, bins=50, edgecolor='black')
            axes[0, 2].set_xlabel('Content Length (chars)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Content Length Distribution')
        
        # 4. 5W1H coverage
        quality = self.analyze_5w1h_quality()
        fields = ['who', 'what', 'when', 'where', 'why', 'how']
        coverages = [quality.get(f'{field}_coverage', 0) for field in fields]
        axes[1, 0].bar(fields, coverages)
        axes[1, 0].set_ylabel('Coverage (%)')
        axes[1, 0].set_title('5W1H Field Coverage')
        axes[1, 0].set_ylim(0, 100)
        
        # 5. Turn distribution
        cursor.execute("""
            SELECT 
                CAST(json_extract(metadata, '$.turn_index') as INTEGER) as turn,
                COUNT(*) as count
            FROM memories
            WHERE json_extract(metadata, '$.turn_index') IS NOT NULL
            GROUP BY turn
            ORDER BY turn
            LIMIT 20
        """)
        turn_data = cursor.fetchall()
        if turn_data:
            turns = [row['turn'] for row in turn_data]
            counts = [row['count'] for row in turn_data]
            axes[1, 1].plot(turns, counts, marker='o')
            axes[1, 1].set_xlabel('Turn Index')
            axes[1, 1].set_ylabel('Number of Memories')
            axes[1, 1].set_title('Conversation Turn Distribution')
        
        # 6. Model distribution (top 10)
        cursor.execute("""
            SELECT 
                json_extract(metadata, '$.model') as model,
                COUNT(*) as count
            FROM memories
            WHERE json_extract(metadata, '$.model') IS NOT NULL
            GROUP BY model
            ORDER BY count DESC
            LIMIT 10
        """)
        model_data = cursor.fetchall()
        if model_data:
            models = [row['model'][:15] for row in model_data]  # Truncate long names
            counts = [row['count'] for row in model_data]
            axes[1, 2].barh(models, counts)
            axes[1, 2].set_xlabel('Number of Memories')
            axes[1, 2].set_title('Top 10 Models')
        
        plt.tight_layout()
        plt.savefig('lmsys_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\n✓ Plots saved to lmsys_analysis.png")


def main():
    """Main entry point for LMSYS data analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze LMSYS data in JAM memory system')
    parser.add_argument('--output', '-o', help='Output file for JSON report')
    parser.add_argument('--plot', action='store_true', help='Generate visualization plots')
    parser.add_argument('--queries', type=int, default=100, 
                       help='Number of queries for retrieval testing')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = LMSYSAnalyzer()
    
    # Generate report
    result = analyzer.generate_report(output_file=args.output)
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating visualization plots...")
        analyzer.plot_analysis()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()