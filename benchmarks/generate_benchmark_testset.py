#!/usr/bin/env python3
"""
Generate comprehensive benchmark test dataset for JAM memory system.
Creates diverse test cases using database content and LLM generation.
"""

import sys
import json
import sqlite3
import random
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_memory.config import cfg


@dataclass
class BenchmarkTestCase:
    """A comprehensive benchmark test case."""
    test_id: str
    test_type: str  # 'exact', 'semantic', 'temporal', 'actor', 'session', 'spatial', 'composite'
    query_text: str
    query_metadata: Dict  # Contains temporal_hint, actor_hint, spatial_hint, session_id
    expected_relevant: List[str]  # Memory IDs that should be retrieved
    ground_truth_metadata: Dict  # Metadata about why these are relevant
    difficulty: str  # 'easy', 'medium', 'hard'
    created_at: str


class BenchmarkTestGenerator:
    """Generate comprehensive benchmark test cases."""
    
    def __init__(self):
        """Initialize generator."""
        # Database connection
        self.db_conn = sqlite3.connect(cfg.db_path)
        self.db_conn.row_factory = sqlite3.Row
        
        # LLM configuration
        self.llm_url = cfg.get('llm_base_url', 'http://localhost:8001/v1')
        self.llm_model = cfg.get('llm_model', 'local-model')
        
        # Output directory
        self.output_dir = Path("benchmarks/test_data")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Cache database statistics
        self._cache_db_stats()
    
    def _cache_db_stats(self):
        """Cache database statistics for test generation."""
        cursor = self.db_conn.cursor()
        
        # Get date range
        cursor.execute("""
            SELECT MIN(when_ts) as min_date, MAX(when_ts) as max_date
            FROM memories
        """)
        row = cursor.fetchone()
        self.min_date = row['min_date']
        self.max_date = row['max_date']
        
        # Get active sessions
        cursor.execute("""
            SELECT session_id, COUNT(*) as count
            FROM memories
            GROUP BY session_id
            HAVING count >= 3
            ORDER BY count DESC
            LIMIT 100
        """)
        self.active_sessions = [(row['session_id'], row['count']) for row in cursor.fetchall()]
        
        # Get active actors
        cursor.execute("""
            SELECT who_id, who_type, COUNT(*) as count
            FROM memories
            GROUP BY who_id
            HAVING count >= 3
            ORDER BY count DESC
            LIMIT 50
        """)
        self.active_actors = [(row['who_id'], row['who_type'], row['count']) for row in cursor.fetchall()]
        
        # Get common where types and values
        cursor.execute("""
            SELECT where_type, where_value, COUNT(*) as count
            FROM memories
            GROUP BY where_type, where_value
            HAVING count >= 2
            ORDER BY count DESC
            LIMIT 50
        """)
        self.common_locations = [(row['where_type'], row['where_value'], row['count']) for row in cursor.fetchall()]
        
        print(f"Database stats cached:")
        print(f"  Date range: {self.min_date} to {self.max_date}")
        print(f"  Active sessions: {len(self.active_sessions)}")
        print(f"  Active actors: {len(self.active_actors)}")
        print(f"  Common locations: {len(self.common_locations)}")
    
    def _call_llm(self, prompt: str, system_prompt: str = None) -> Optional[str]:
        """Call LLM for text generation."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = requests.post(
                f"{self.llm_url}/chat/completions",
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 300
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
    
    def generate_exact_match_cases(self, num_cases: int = 20) -> List[BenchmarkTestCase]:
        """Generate exact match test cases."""
        print("Generating exact match test cases...")
        test_cases = []
        cursor = self.db_conn.cursor()
        
        # Find memories with distinctive phrases
        cursor.execute("""
            SELECT memory_id, raw_text, session_id, who_id, when_ts, where_type, where_value
            FROM memories
            WHERE LENGTH(raw_text) BETWEEN 100 AND 500
            ORDER BY RANDOM()
            LIMIT ?
        """, (num_cases * 3,))
        
        memories = cursor.fetchall()
        
        for i, mem in enumerate(memories[:num_cases]):
            # Extract a distinctive phrase from the text
            text = mem['raw_text']
            words = text.split()
            
            if len(words) > 15:
                # Find a meaningful phrase (avoid common words)
                start_idx = random.randint(5, min(len(words) - 10, 20))
                phrase_length = random.randint(4, 8)
                query_phrase = ' '.join(words[start_idx:start_idx + phrase_length])
                
                # Find all memories containing this exact phrase
                cursor.execute("""
                    SELECT memory_id FROM memories
                    WHERE raw_text LIKE ?
                    LIMIT 30
                """, (f'%{query_phrase}%',))
                
                relevant_ids = [row['memory_id'] for row in cursor.fetchall()]
                
                if relevant_ids:
                    test_cases.append(BenchmarkTestCase(
                        test_id=f"exact_{i:04d}",
                        test_type='exact',
                        query_text=query_phrase,
                        query_metadata={
                            'session_id': mem['session_id'],
                            'original_memory_id': mem['memory_id']
                        },
                        expected_relevant=relevant_ids,
                        ground_truth_metadata={
                            'match_type': 'exact_phrase',
                            'phrase_location': 'middle',
                            'original_text_length': len(text)
                        },
                        difficulty='easy',
                        created_at=datetime.now().isoformat()
                    ))
        
        return test_cases
    
    def generate_semantic_cases(self, num_cases: int = 30) -> List[BenchmarkTestCase]:
        """Generate semantic similarity test cases using LLM."""
        print("Generating semantic similarity test cases...")
        test_cases = []
        cursor = self.db_conn.cursor()
        
        # Query types for variety
        query_types = [
            ("paraphrase", "Rewrite this text using completely different words while preserving the exact same meaning"),
            ("question", "What question would someone ask to get this information as an answer"),
            ("summary", "Summarize the key points of this text in a single sentence"),
            ("expansion", "Elaborate on this concept with more detail and context"),
            ("abstraction", "Express this concept at a higher level of abstraction"),
            ("contextualization", "Add realistic context about when, where, and why this might occur")
        ]
        
        # Get diverse source memories
        cursor.execute("""
            SELECT m.*, 
                   GROUP_CONCAT(m2.memory_id) as related_memories
            FROM memories m
            LEFT JOIN memories m2 ON m.session_id = m2.session_id 
                AND m.memory_id != m2.memory_id
            WHERE LENGTH(m.raw_text) BETWEEN 80 AND 400
            GROUP BY m.memory_id
            HAVING COUNT(m2.memory_id) >= 2
            ORDER BY RANDOM()
            LIMIT ?
        """, (num_cases * 2,))
        
        memories = cursor.fetchall()
        
        for i, mem in enumerate(memories[:num_cases]):
            query_type, instruction = query_types[i % len(query_types)]
            
            # Generate semantic variation using LLM
            prompt = f"""{instruction}:

Original text: {mem['raw_text']}

Additional context:
- Actor: {mem['who_id']} ({mem['who_type']})
- When: {mem['when_ts']}
- Where: {mem['where_type']} - {mem['where_value']}

Your response:"""
            
            variation = self._call_llm(prompt)
            
            if variation:
                # Get related memories as ground truth
                related = mem['related_memories'].split(',') if mem['related_memories'] else []
                expected = [mem['memory_id']] + related[:5]
                
                # Also find topically similar memories
                key_words = [w for w in mem['raw_text'].split() if len(w) > 5][:3]
                for word in key_words:
                    cursor.execute("""
                        SELECT memory_id FROM memories
                        WHERE raw_text LIKE ? AND memory_id NOT IN ({})
                        LIMIT 3
                    """.format(','.join(['?'] * len(expected))), 
                    (f'%{word}%', *expected))
                    
                    for row in cursor.fetchall():
                        if row['memory_id'] not in expected:
                            expected.append(row['memory_id'])
                
                test_cases.append(BenchmarkTestCase(
                    test_id=f"semantic_{i:04d}",
                    test_type='semantic',
                    query_text=variation,
                    query_metadata={
                        'query_subtype': query_type,
                        'session_id': mem['session_id'],
                        'original_memory_id': mem['memory_id'],
                        'actor_hint': mem['who_id'],
                        'temporal_hint': mem['when_ts'],
                        'spatial_hint': f"{mem['where_type']}:{mem['where_value']}"
                    },
                    expected_relevant=expected[:10],
                    ground_truth_metadata={
                        'original_text': mem['raw_text'][:200],
                        'variation_type': query_type,
                        'semantic_similarity': 'high'
                    },
                    difficulty='medium' if query_type in ['paraphrase', 'summary'] else 'hard',
                    created_at=datetime.now().isoformat()
                ))
        
        return test_cases
    
    def generate_temporal_cases(self, num_cases: int = 20) -> List[BenchmarkTestCase]:
        """Generate temporal-based test cases."""
        print("Generating temporal test cases...")
        test_cases = []
        cursor = self.db_conn.cursor()
        
        # Find dates with multiple memories
        cursor.execute("""
            SELECT DATE(when_ts) as date, COUNT(*) as count,
                   GROUP_CONCAT(memory_id) as memory_ids
            FROM memories
            GROUP BY DATE(when_ts)
            HAVING count >= 5
            ORDER BY RANDOM()
            LIMIT ?
        """, (num_cases,))
        
        date_groups = cursor.fetchall()
        
        for i, group in enumerate(date_groups):
            date = group['date']
            memory_ids = group['memory_ids'].split(',')
            
            # Pick a sample memory from that date for context
            cursor.execute("""
                SELECT raw_text, who_id, where_value 
                FROM memories 
                WHERE memory_id = ?
            """, (memory_ids[0],))
            sample = cursor.fetchone()
            
            # Generate different temporal queries
            temporal_queries = [
                f"What happened on {date}?",
                f"Show me everything from {date}",
                f"What did we discuss on {date}?",
                f"Retrieve memories from {date}",
                f"What was going on during {date}?"
            ]
            
            # Use LLM to create a more natural temporal query
            prompt = f"""Create a natural query asking about events from {date}.
Context: On this date, there were {group['count']} memories including: "{sample['raw_text'][:100]}..."
The query should feel like something a user would naturally ask to recall that day's events.

Query:"""
            
            llm_query = self._call_llm(prompt) or random.choice(temporal_queries)
            
            test_cases.append(BenchmarkTestCase(
                test_id=f"temporal_{i:04d}",
                test_type='temporal',
                query_text=llm_query,
                query_metadata={
                    'temporal_hint': date,
                    'temporal_type': 'exact_date',
                    'memory_count': group['count']
                },
                expected_relevant=memory_ids[:20],  # Limit to 20 for practical testing
                ground_truth_metadata={
                    'date': date,
                    'total_memories_on_date': group['count'],
                    'sample_content': sample['raw_text'][:200]
                },
                difficulty='easy' if group['count'] < 10 else 'medium',
                created_at=datetime.now().isoformat()
            ))
        
        return test_cases
    
    def generate_actor_cases(self, num_cases: int = 15) -> List[BenchmarkTestCase]:
        """Generate actor-based test cases."""
        print("Generating actor-based test cases...")
        test_cases = []
        
        for i, (actor_id, actor_type, count) in enumerate(self.active_actors[:num_cases]):
            cursor = self.db_conn.cursor()
            
            # Get memories from this actor
            cursor.execute("""
                SELECT memory_id, raw_text, when_ts, where_value
                FROM memories
                WHERE who_id = ?
                ORDER BY when_ts DESC
                LIMIT 30
            """, (actor_id,))
            
            actor_memories = cursor.fetchall()
            memory_ids = [m['memory_id'] for m in actor_memories]
            
            # Sample content for query generation
            sample_content = actor_memories[0]['raw_text'][:150] if actor_memories else ""
            
            # Generate natural queries about this actor
            prompt = f"""Create a natural query asking about interactions with or content from {actor_id} (type: {actor_type}).
Sample content from this actor: "{sample_content}"
The query should sound like someone trying to recall what this person/entity said or did.

Query:"""
            
            query = self._call_llm(prompt) or f"What has {actor_id} been saying?"
            
            test_cases.append(BenchmarkTestCase(
                test_id=f"actor_{i:04d}",
                test_type='actor',
                query_text=query,
                query_metadata={
                    'actor_hint': actor_id,
                    'actor_type': actor_type,
                    'memory_count': len(memory_ids)
                },
                expected_relevant=memory_ids[:15],
                ground_truth_metadata={
                    'actor_id': actor_id,
                    'actor_type': actor_type,
                    'total_memories': count,
                    'sample_content': sample_content
                },
                difficulty='easy' if count > 20 else 'medium',
                created_at=datetime.now().isoformat()
            ))
        
        return test_cases
    
    def generate_session_cases(self, num_cases: int = 20) -> List[BenchmarkTestCase]:
        """Generate session-based test cases."""
        print("Generating session-based test cases...")
        test_cases = []
        
        for i, (session_id, count) in enumerate(self.active_sessions[:num_cases]):
            cursor = self.db_conn.cursor()
            
            # Get session details
            cursor.execute("""
                SELECT memory_id, raw_text, who_id, when_ts,
                       MIN(when_ts) as session_start,
                       MAX(when_ts) as session_end
                FROM memories
                WHERE session_id = ?
                GROUP BY memory_id
                ORDER BY when_ts
                LIMIT 30
            """, (session_id,))
            
            session_memories = cursor.fetchall()
            
            if session_memories:
                memory_ids = [m['memory_id'] for m in session_memories]
                
                # Get session theme/topic
                sample_texts = ' '.join([m['raw_text'][:100] for m in session_memories[:3]])
                
                # Generate contextual query about this session
                prompt = f"""Based on this conversation session, create a natural query someone might ask to recall this discussion.
Session sample: "{sample_texts[:300]}..."
The query should ask about the conversation's topic or what was discussed.

Query:"""
                
                query = self._call_llm(prompt) or "What did we talk about in that conversation?"
                
                test_cases.append(BenchmarkTestCase(
                    test_id=f"session_{i:04d}",
                    test_type='session',
                    query_text=query,
                    query_metadata={
                        'session_id': session_id,
                        'session_length': count
                    },
                    expected_relevant=memory_ids,
                    ground_truth_metadata={
                        'session_id': session_id,
                        'total_memories': count,
                        'session_sample': sample_texts[:200]
                    },
                    difficulty='easy' if count < 10 else 'medium',
                    created_at=datetime.now().isoformat()
                ))
        
        return test_cases
    
    def generate_spatial_cases(self, num_cases: int = 15) -> List[BenchmarkTestCase]:
        """Generate spatial/location-based test cases."""
        print("Generating spatial/location test cases...")
        test_cases = []
        
        for i, (where_type, where_value, count) in enumerate(self.common_locations[:num_cases]):
            cursor = self.db_conn.cursor()
            
            # Get memories from this location
            cursor.execute("""
                SELECT memory_id, raw_text, who_id, when_ts
                FROM memories
                WHERE where_type = ? AND where_value = ?
                ORDER BY when_ts DESC
                LIMIT 25
            """, (where_type, where_value))
            
            location_memories = cursor.fetchall()
            
            if location_memories:
                memory_ids = [m['memory_id'] for m in location_memories]
                sample_content = location_memories[0]['raw_text'][:150]
                
                # Generate natural spatial query
                prompt = f"""Create a natural query asking about activities or events in a {where_type} context: {where_value}.
Sample content from this location: "{sample_content}"
The query should ask about what happened in/at this location or context.

Query:"""
                
                query = self._call_llm(prompt) or f"What happened in {where_value}?"
                
                test_cases.append(BenchmarkTestCase(
                    test_id=f"spatial_{i:04d}",
                    test_type='spatial',
                    query_text=query,
                    query_metadata={
                        'spatial_hint': f"{where_type}:{where_value}",
                        'where_type': where_type,
                        'where_value': where_value
                    },
                    expected_relevant=memory_ids[:15],
                    ground_truth_metadata={
                        'location_type': where_type,
                        'location_value': where_value,
                        'total_memories': count,
                        'sample_content': sample_content
                    },
                    difficulty='medium',
                    created_at=datetime.now().isoformat()
                ))
        
        return test_cases
    
    def generate_composite_cases(self, num_cases: int = 20) -> List[BenchmarkTestCase]:
        """Generate complex composite test cases combining multiple retrieval signals."""
        print("Generating composite test cases...")
        test_cases = []
        cursor = self.db_conn.cursor()
        
        for i in range(num_cases):
            # Pick random combination of constraints
            constraints = []
            params = []
            query_metadata = {}
            
            # Maybe add temporal constraint
            if random.random() > 0.3:
                date_offset = random.randint(0, 30)
                target_date = (datetime.now() - timedelta(days=date_offset)).date()
                constraints.append("DATE(when_ts) = ?")
                params.append(str(target_date))
                query_metadata['temporal_hint'] = str(target_date)
            
            # Maybe add actor constraint
            if random.random() > 0.3 and self.active_actors:
                actor = random.choice(self.active_actors[:10])
                constraints.append("who_id = ?")
                params.append(actor[0])
                query_metadata['actor_hint'] = actor[0]
            
            # Maybe add location constraint
            if random.random() > 0.3 and self.common_locations:
                location = random.choice(self.common_locations[:10])
                constraints.append("where_type = ? AND where_value = ?")
                params.extend([location[0], location[1]])
                query_metadata['spatial_hint'] = f"{location[0]}:{location[1]}"
            
            # Build query
            if constraints:
                where_clause = " AND ".join(constraints)
                query = f"""
                    SELECT memory_id, raw_text, who_id, when_ts, where_value
                    FROM memories
                    WHERE {where_clause}
                    ORDER BY RANDOM()
                    LIMIT 30
                """
                
                cursor.execute(query, params)
                memories = cursor.fetchall()
                
                if len(memories) >= 3:
                    memory_ids = [m['memory_id'] for m in memories]
                    sample_texts = ' '.join([m['raw_text'][:100] for m in memories[:3]])
                    
                    # Generate complex query using LLM
                    context_parts = []
                    if 'temporal_hint' in query_metadata:
                        context_parts.append(f"Date: {query_metadata['temporal_hint']}")
                    if 'actor_hint' in query_metadata:
                        context_parts.append(f"Person: {query_metadata['actor_hint']}")
                    if 'spatial_hint' in query_metadata:
                        context_parts.append(f"Location: {query_metadata['spatial_hint']}")
                    
                    prompt = f"""Create a natural, complex query that asks about specific memories with these constraints:
{' | '.join(context_parts)}
Sample content: "{sample_texts[:200]}..."
The query should naturally incorporate multiple aspects (who, when, where) without being too mechanical.

Query:"""
                    
                    query_text = self._call_llm(prompt) or f"What was discussed {' '.join(context_parts)}?"
                    
                    test_cases.append(BenchmarkTestCase(
                        test_id=f"composite_{i:04d}",
                        test_type='composite',
                        query_text=query_text,
                        query_metadata=query_metadata,
                        expected_relevant=memory_ids[:15],
                        ground_truth_metadata={
                            'constraint_count': len(query_metadata),
                            'constraints': list(query_metadata.keys()),
                            'total_matches': len(memories),
                            'sample_content': sample_texts[:200]
                        },
                        difficulty='hard',
                        created_at=datetime.now().isoformat()
                    ))
        
        return test_cases
    
    def generate_complete_testset(self, 
                                 exact_cases: int = 20,
                                 semantic_cases: int = 30,
                                 temporal_cases: int = 20,
                                 actor_cases: int = 15,
                                 session_cases: int = 20,
                                 spatial_cases: int = 15,
                                 composite_cases: int = 20) -> List[BenchmarkTestCase]:
        """Generate a complete benchmark test set."""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE BENCHMARK TEST SET")
        print("="*60)
        
        all_cases = []
        
        # Generate each type
        if exact_cases > 0:
            all_cases.extend(self.generate_exact_match_cases(exact_cases))
        
        if semantic_cases > 0:
            all_cases.extend(self.generate_semantic_cases(semantic_cases))
        
        if temporal_cases > 0:
            all_cases.extend(self.generate_temporal_cases(temporal_cases))
        
        if actor_cases > 0:
            all_cases.extend(self.generate_actor_cases(actor_cases))
        
        if session_cases > 0:
            all_cases.extend(self.generate_session_cases(session_cases))
        
        if spatial_cases > 0:
            all_cases.extend(self.generate_spatial_cases(spatial_cases))
        
        if composite_cases > 0:
            all_cases.extend(self.generate_composite_cases(composite_cases))
        
        print(f"\nTotal test cases generated: {len(all_cases)}")
        
        # Show distribution
        type_counts = defaultdict(int)
        difficulty_counts = defaultdict(int)
        for case in all_cases:
            type_counts[case.test_type] += 1
            difficulty_counts[case.difficulty] += 1
        
        print("\nTest type distribution:")
        for test_type, count in sorted(type_counts.items()):
            print(f"  {test_type}: {count}")
        
        print("\nDifficulty distribution:")
        for difficulty, count in sorted(difficulty_counts.items()):
            print(f"  {difficulty}: {count}")
        
        return all_cases
    
    def save_testset(self, test_cases: List[BenchmarkTestCase], filename: str = None):
        """Save test cases to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_testset_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Convert to serializable format
        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_cases": len(test_cases),
                "test_types": list(set(tc.test_type for tc in test_cases)),
                "difficulty_levels": list(set(tc.difficulty for tc in test_cases)),
                "database": str(cfg.db_path),
                "database_stats": {
                    "date_range": f"{self.min_date} to {self.max_date}",
                    "active_sessions": len(self.active_sessions),
                    "active_actors": len(self.active_actors),
                    "common_locations": len(self.common_locations)
                }
            },
            "test_cases": [asdict(tc) for tc in test_cases]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Test set saved to: {filepath}")
        
        # Generate human-readable summary
        self._save_summary(test_cases, filepath)
        
        return filepath
    
    def _save_summary(self, test_cases: List[BenchmarkTestCase], json_path: Path):
        """Save human-readable summary of test set."""
        summary_path = str(json_path).replace('.json', '_summary.txt')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("BENCHMARK TEST SET SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total cases: {len(test_cases)}\n\n")
            
            # Type distribution
            type_counts = defaultdict(int)
            for tc in test_cases:
                type_counts[tc.test_type] += 1
            
            f.write("Test type distribution:\n")
            for test_type, count in sorted(type_counts.items()):
                f.write(f"  {test_type}: {count}\n")
            
            # Difficulty distribution
            diff_counts = defaultdict(int)
            for tc in test_cases:
                diff_counts[tc.difficulty] += 1
            
            f.write("\nDifficulty distribution:\n")
            for difficulty, count in sorted(diff_counts.items()):
                f.write(f"  {difficulty}: {count}\n")
            
            # Sample cases
            f.write("\n" + "-"*60 + "\n")
            f.write("SAMPLE TEST CASES\n")
            f.write("-"*60 + "\n\n")
            
            # Show 2 examples of each type
            for test_type in sorted(type_counts.keys()):
                type_cases = [tc for tc in test_cases if tc.test_type == test_type][:2]
                
                for tc in type_cases:
                    f.write(f"Test ID: {tc.test_id}\n")
                    f.write(f"Type: {tc.test_type} | Difficulty: {tc.difficulty}\n")
                    f.write(f"Query: {tc.query_text[:150]}{'...' if len(tc.query_text) > 150 else ''}\n")
                    f.write(f"Expected results: {len(tc.expected_relevant)} memories\n")
                    
                    if tc.query_metadata:
                        f.write(f"Metadata: {', '.join(f'{k}={v}' for k, v in tc.query_metadata.items() if not k.endswith('_id'))}\n")
                    
                    f.write("\n")
        
        print(f"✓ Summary saved to: {summary_path}")


def main():
    """Generate comprehensive benchmark test set."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comprehensive benchmark test dataset')
    parser.add_argument('--exact', type=int, default=20,
                       help='Number of exact match cases')
    parser.add_argument('--semantic', type=int, default=30,
                       help='Number of semantic similarity cases')
    parser.add_argument('--temporal', type=int, default=20,
                       help='Number of temporal cases')
    parser.add_argument('--actor', type=int, default=15,
                       help='Number of actor-based cases')
    parser.add_argument('--session', type=int, default=20,
                       help='Number of session-based cases')
    parser.add_argument('--spatial', type=int, default=15,
                       help='Number of spatial/location cases')
    parser.add_argument('--composite', type=int, default=20,
                       help='Number of composite cases')
    parser.add_argument('--output', '-o',
                       help='Output filename (default: timestamped)')
    
    args = parser.parse_args()
    
    generator = BenchmarkTestGenerator()
    
    # Generate complete test set
    test_cases = generator.generate_complete_testset(
        exact_cases=args.exact,
        semantic_cases=args.semantic,
        temporal_cases=args.temporal,
        actor_cases=args.actor,
        session_cases=args.session,
        spatial_cases=args.spatial,
        composite_cases=args.composite
    )
    
    # Save to file
    if test_cases:
        filepath = generator.save_testset(test_cases, args.output)
        
        print("\n" + "="*60)
        print("BENCHMARK TEST SET GENERATION COMPLETE")
        print("="*60)
        print(f"Generated {len(test_cases)} test cases")
        print(f"Saved to: {filepath}")
        print("\nUse this file with recall_benchmark.py to run benchmarks")


if __name__ == "__main__":
    main()
