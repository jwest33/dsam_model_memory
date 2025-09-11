#!/usr/bin/env python3
"""
Test module that uses Flask app's chat interface to test memory retrieval.
Gets raw memories from DB, uses LLM to alter them, then tests retrieval.
"""

import sys
import json
import sqlite3
import random
import requests
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_memory.config import cfg


class FlaskRetrievalTester:
    """Test memory retrieval using Flask app's actual chat interface."""
    
    def __init__(self, flask_url: str = "http://localhost:5001", llm_url: str = "http://localhost:8000", 
                 seed: Optional[int] = None, memory_ids: Optional[List[str]] = None):
        """Initialize tester with Flask app URL and LLM URL.
        
        Args:
            flask_url: URL of the Flask app
            llm_url: URL of the LLM server
            seed: Random seed for reproducible memory selection
            memory_ids: Specific memory IDs to test (overrides random selection)
        """
        self.flask_url = flask_url
        self.llm_url = llm_url
        self.db_path = cfg.db_path  # Store the db_path
        self.db_conn = sqlite3.connect(self.db_path)
        self.db_conn.row_factory = sqlite3.Row
        self.session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.seed = seed
        self.specific_memory_ids = memory_ids
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            print(f"Using random seed: {seed}")
        
        # Test if Flask is running
        try:
            response = requests.get(f"{self.flask_url}/")
            if response.status_code != 200:
                print(f"Warning: Flask app may not be running properly at {self.flask_url}")
        except requests.exceptions.ConnectionError:
            print(f"Error: Cannot connect to Flask app at {self.flask_url}")
            print("Please start the Flask app with: python -m agentic_memory.server.flask_app")
            sys.exit(1)
            
        # Test if LLM is running
        try:
            response = requests.get(f"{self.llm_url}/health")
            if response.status_code != 200:
                print(f"Warning: LLM server may not be running properly at {self.llm_url}")
        except requests.exceptions.ConnectionError:
            print(f"Warning: Cannot connect to LLM server at {self.llm_url}")
            print("LLM paraphrasing will use fallback methods")
    
    def get_raw_memories(self, num_memories: int = 3) -> List[Dict]:
        """Get raw memories from the database.
        
        If specific_memory_ids was set, uses those IDs.
        Otherwise selects randomly (using seed if provided).
        """
        cursor = self.db_conn.cursor()
        
        memories = []
        
        if self.specific_memory_ids:
            # Use specific memory IDs
            print(f"Using specific memory IDs: {self.specific_memory_ids[:num_memories]}")
            memory_ids_to_use = self.specific_memory_ids[:num_memories]
            
            for memory_id in memory_ids_to_use:
                cursor.execute("""
                    SELECT memory_id, raw_text, who_id, what, when_ts, where_value, why, how
                    FROM memories
                    WHERE memory_id = ?
                """, (memory_id,))
                
                row = cursor.fetchone()
                if row:
                    memories.append({
                        'memory_id': row['memory_id'],
                        'raw_text': row['raw_text'],
                        'who': row['who_id'],
                        'what': row['what'],
                        'when': row['when_ts'],
                        'where': row['where_value'],
                        'why': row['why'],
                        'how': row['how']
                    })
                else:
                    print(f"  Warning: Memory ID {memory_id} not found in database")
        else:
            # Random selection (deterministic if seed was set)
            # Note: SQLite's RANDOM() is not affected by Python's random.seed()
            # So we need to fetch all candidates and select in Python
            cursor.execute("""
                SELECT memory_id, raw_text, who_id, what, when_ts, where_value, why, how
                FROM memories
                WHERE LENGTH(raw_text) > 50 AND LENGTH(raw_text) < 1000
                    AND raw_text IS NOT NULL
            """)
            
            all_candidates = []
            for row in cursor.fetchall():
                all_candidates.append({
                    'memory_id': row['memory_id'],
                    'raw_text': row['raw_text'],
                    'who': row['who_id'],
                    'what': row['what'],
                    'when': row['when_ts'],
                    'where': row['where_value'],
                    'why': row['why'],
                    'how': row['how']
                })
            
            # Use Python's random.sample for deterministic selection with seed
            if len(all_candidates) >= num_memories:
                memories = random.sample(all_candidates, num_memories)
            else:
                memories = all_candidates
                print(f"  Warning: Only found {len(memories)} suitable memories")
        
        print(f"Selected {len(memories)} raw memories from database")
        for mem in memories:
            print(f"  - {mem['memory_id']}: {mem['raw_text'][:100]}...")
        
        return memories
    
    def paraphrase_with_llm(self, text: str) -> Optional[str]:
        """Use LLM to paraphrase text, keeping similar meaning but different words."""
        try:
            # Use synonym replacement instruction
            prompt = f"""Replace some words with synonyms in this sentence but keep the exact same meaning. Do not change what the sentence is asking for or describing.

Original: {text}
With synonyms: """
            
            # Try the llama.cpp completion endpoint
            print(f"  Calling LLM at {self.llm_url}/completion...")
            response = requests.post(
                f"{self.llm_url}/completion",
                json={
                    "prompt": prompt,
                    "temperature": 0.7,
                    "n_predict": 500,
                    "stop": ["\\n", "Original:", "With synonyms:"]
                },
                timeout=30  # Reduced timeout
            )
            
            print(f"  LLM response status: {response.status_code}")
            
            if response.status_code == 200:
                response_json = response.json()
                # llama.cpp returns 'content' field
                if 'content' in response_json:
                    raw_response = response_json['content']
                    print(f"  Raw LLM response: '{raw_response}'")
                    
                    paraphrased = raw_response.strip()
                    # Clean up any trailing quotes or extra formatting
                    paraphrased = paraphrased.strip('"').strip()
                    # Take only the first line if multiple lines are returned
                    if '\n' in paraphrased:
                        paraphrased = paraphrased.split('\n')[0].strip()
                    
                    print(f"  Cleaned paraphrase: '{paraphrased}'")
                    
                    # Reject obviously wrong responses
                    if paraphrased and paraphrased not in ['1', '2', '3', 'Yes', 'No', 'OK']:
                        print(f"  LLM paraphrase successful")
                        return paraphrased
                    else:
                        print(f"  LLM returned invalid paraphrase: '{paraphrased}'")
                        return None
                else:
                    print(f"  Unexpected response format: {list(response_json.keys())}")
                    return None
            else:
                print(f"  LLM request failed with status {response.status_code}")
                try:
                    error_detail = response.text[:200]
                    print(f"  Error detail: {error_detail}")
                except:
                    pass
                return None
                
        except requests.exceptions.ConnectionError:
            print(f"  Could not connect to LLM at {self.llm_url}")
            return None
        except requests.exceptions.Timeout:
            print(f"  LLM request timed out")
            return None
        except Exception as e:
            print(f"  Unexpected error calling LLM: {e}")
            return None
    
    def create_altered_memories(self, raw_memories: List[Dict]) -> List[Dict]:
        """Create altered versions of memories using LLM paraphrasing.
        
        Returns:
            List of dicts with 'original' memory and 'altered' text
        """
        altered_memories = []
        
        # List of query prefixes to make it more like a retrieval query
        query_prefixes = [
            "Have we discussed ",
            "Do you remember when we talked about ",
            "What do you remember about ",
            "Can you remember our conversation about "
        ]
        
        for i, memory in enumerate(raw_memories):
            print(f"\nProcessing memory {i+1}/{len(raw_memories)}: {memory['memory_id']}")
            original_text = memory['raw_text']
            
            # Skip LLM paraphrasing - use simple word variations
            # This is more predictable for testing retrieval
            print("  Using simple paraphrasing method")
            
            # Apply simple word substitutions that preserve meaning
            paraphrased = original_text
            
            # Common word replacements
            replacements = [
                ("write", "create"),
                ("short", "brief"),
                ("about", "regarding"),
                ("new", "fresh"),
                ("help", "assist"),
                ("create", "make"),
                ("need", "require"),
                ("want", "would like"),
                ("can you", "could you"),
                ("please", "kindly"),
                ("show", "display"),
                ("tell", "inform"),
                ("find", "locate"),
                ("get", "obtain"),
                ("use", "utilize"),
                ("good", "nice"),
                ("big", "large"),
                ("small", "little")
            ]
            
            # Apply some replacements randomly (not all, to keep it close to original)
            for old_word, new_word in replacements:
                if old_word in paraphrased.lower() and random.random() > 0.5:
                    # Case-insensitive replacement
                    import re
                    paraphrased = re.sub(r'\b' + re.escape(old_word) + r'\b', new_word, paraphrased, flags=re.IGNORECASE)
            
            # Add a query prefix to make it more like a retrieval query
            prefix = random.choice(query_prefixes)
            altered_text_with_prefix = prefix + paraphrased.lower()
            
            altered_memories.append({
                'original': memory,
                'altered_text': altered_text_with_prefix,
                'paraphrase_method': 'simple'
            })
            
            print(f"  Original: {original_text[:100]}...")
            print(f"  Altered:  {altered_text_with_prefix[:100]}...")
        
        return altered_memories
    
    def test_retrieval(self, query_text: str, expected_memory_id: Optional[str] = None) -> Dict:
        """Test retrieval using Flask's chat API.
        
        Args:
            query_text: The query to send (altered/paraphrased text)
            expected_memory_id: The original memory ID we expect to retrieve
            
        Returns:
            Dict with retrieval results and analysis
        """
        # Prepare the request
        payload = {
            'session_id': self.session_id,
            'text': query_text,
            'messages': []  # Empty conversation history
        }
        
        try:
            # Send request to Flask chat API
            response = requests.post(
                f"{self.flask_url}/api/chat",
                json=payload,
                timeout=600
            )
            
            if response.status_code != 200:
                return {
                    'success': False,
                    'error': f"API returned status {response.status_code}",
                    'query': query_text,
                    'expected': expected_memory_id
                }
            
            # Parse response
            result = response.json()
            
            # Store the LLM reply for analysis
            if 'reply' in result:
                result['llm_reply'] = result['reply']
            
            # Add success flag and ensure expected fields exist
            result['success'] = True
            result['query'] = query_text
            result['expected'] = expected_memory_id
            
            
            # Extract retrieval information from the response
            # The Flask /api/chat returns: {"reply": str, "block": dict, "tool_calls": int}
            retrieved_mems = []
            retrieved_ids = []
            
            # The 'block' contains the memories used for context
            if 'block' in result and result['block']:
                block = result['block']
                
                # Debug: Show more details about the block
                print(f"  Block debug:")
                print(f"    Block type: {block.get('block', 'N/A')}")
                print(f"    Block has {len(block.get('members', []))} members")
                if 'members' in block and block['members']:
                    print(f"    First 5 member IDs: {block['members'][:5]}")
                    print(f"    Last 5 member IDs: {block['members'][-5:]}")
                
                # Block contains memory IDs in 'members' field
                if 'members' in block and block['members']:
                    retrieved_ids = block['members']
                    
                    # Fetch full memory details from database for these IDs
                    if retrieved_ids:
                        con = sqlite3.connect(self.db_path)
                        con.row_factory = sqlite3.Row
                        placeholders = ','.join(['?'] * len(retrieved_ids))
                        query = f"""
                        SELECT memory_id as id, raw_text, what, why, how, who_id, where_value, when_ts
                        FROM memories 
                        WHERE memory_id IN ({placeholders})
                        """
                        rows = con.execute(query, retrieved_ids).fetchall()
                        retrieved_mems = [dict(row) for row in rows]
                        con.close()
                        
                        # Try to get scoring information by making a direct retrieval call
                        # This helps us understand the scoring mechanism
                        try:
                            print(f"    Attempting to get scoring weights...")
                            # Make a direct call to the retrieval endpoint to get scores
                            scoring_response = requests.post(
                                f"{self.flask_url}/api/debug_retrieval",
                                json={'query': query_text, 'session_id': self.session_id},
                                timeout=10
                            )
                            if scoring_response.status_code == 200:
                                scoring_data = scoring_response.json()
                                # Extract weights if available
                                if 'scores' in scoring_data:
                                    result['memory_scores'] = scoring_data['scores']
                                    print(f"    Retrieved scoring data for {len(scoring_data['scores'])} memories")
                                if 'weights' in scoring_data:
                                    result['scoring_weights'] = scoring_data['weights']
                                    print(f"    Current scoring weights: {scoring_data['weights']}")
                        except:
                            # Debug endpoint might not exist yet
                            pass
                
                # Store for analysis
                result['retrieved_memories'] = retrieved_mems
                result['block_info'] = block
            
            if retrieved_ids or retrieved_mems:
                # Use retrieved_ids if we have them, otherwise extract from retrieved_mems
                if retrieved_ids:
                    result['retrieved_ids'] = retrieved_ids
                else:
                    result['retrieved_ids'] = [m.get('id') for m in retrieved_mems]
                
                result['retrieved_count'] = len(result['retrieved_ids'])
                
                # Check if expected memory was found
                if expected_memory_id:
                    found_position = None
                    for i, mem_id in enumerate(result['retrieved_ids']):
                        if mem_id == expected_memory_id:
                            found_position = i + 1
                            break
                    
                    result['found_expected'] = found_position is not None
                    result['expected_position'] = found_position
            else:
                result['retrieved_ids'] = []
                result['retrieved_count'] = 0
                result['found_expected'] = False
        
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'query': query_text,
                'expected': expected_memory_id
            }
    
    def run_test_suite(self, num_memories: int = 3) -> Dict:
        """Run a complete test suite and return results."""
        print("\n" + "="*60)
        print("FLASK RETRIEVAL TEST SUITE")
        print("="*60)
        print(f"Using Flask app at: {self.flask_url}")
        print(f"Using LLM at: {self.llm_url}")
        print(f"Session ID: {self.session_id}")
        if self.seed is not None:
            print(f"Random seed: {self.seed}")
        if self.specific_memory_ids:
            print(f"Using specific memories: {self.specific_memory_ids[:num_memories]}")
        print()
        
        # Step 1: Get raw memories from database
        print("Step 1: Getting raw memories from database...")
        raw_memories = self.get_raw_memories(num_memories)
        
        if not raw_memories:
            print("Error: No suitable memories found in database")
            return {'error': 'No memories found'}
        
        # Step 2: Create altered versions using LLM
        print("\nStep 2: Creating altered versions using LLM...")
        altered_memories = self.create_altered_memories(raw_memories)
        
        # Step 3: Test retrieval with altered text
        print(f"\nStep 3: Testing retrieval with {len(altered_memories)} altered queries...")
        print("-" * 60)
        
        results = []
        successful_retrievals = 0
        
        for i, altered_mem in enumerate(altered_memories, 1):
            original = altered_mem['original']
            altered_text = altered_mem['altered_text']
            expected_id = original['memory_id']
            
            print(f"\nTest {i}/{len(altered_memories)}:")
            print(f"  Original ID: {expected_id}")
            # Fix the print statement formatting
            if len(altered_text) > 100:
                print(f"  Altered query: '{altered_text[:100]}...'")
            else:
                print(f"  Altered query: '{altered_text}'")
            
            # Run the test - send altered text, expect original memory
            result = self.test_retrieval(altered_text, expected_id)
            
            # Add original memory details to result
            result['original_memory'] = original
            result['altered_text'] = altered_text
            result['paraphrase_method'] = altered_mem['paraphrase_method']
            
            results.append(result)
            
            # Print immediate feedback
            if result['success']:
                # Show the LLM's reply
                if 'llm_reply' in result:
                    reply = result['llm_reply']
                    print(f"\n  LLM Reply:")
                    # Show first 500 chars of reply, formatted nicely
                    if len(reply) > 500:
                        print(f"  {reply[:500]}...")
                    else:
                        print(f"  {reply}")
                
                # Show retrieval status
                print(f"\n  Retrieval Status:")
                if result['found_expected']:
                    print(f"  [SUCCESS] Found original memory at position {result.get('expected_position', '?')}")
                    successful_retrievals += 1
                elif result['found_expected'] is False:
                    print(f"  [MISS] Original memory not retrieved (had {result['retrieved_count']} memories in context)")
                else:
                    print(f"  [INFO] Retrieved {result['retrieved_count']} memories for context")
                
                # Show which memory IDs were used for context (brief)
                if result.get('retrieved_ids'):
                    print(f"  Memory IDs used: {result['retrieved_ids'][:3]}{'...' if len(result['retrieved_ids']) > 3 else ''}")
            else:
                print(f"  [ERROR] {result['error']}")
            
            # Small delay to not overwhelm the server
            time.sleep(0.5)
        
        # Calculate statistics
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['success'])
        tests_with_expected = sum(1 for r in results if r.get('expected'))
        found_expected = sum(1 for r in results if r.get('found_expected'))
        llm_paraphrases = sum(1 for r in results if r.get('paraphrase_method') == 'llm')
        
        stats = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': total_tests - successful_tests,
            'tests_with_ground_truth': tests_with_expected,
            'correct_retrievals': found_expected,
            'recall': found_expected / tests_with_expected if tests_with_expected > 0 else 0,
            'avg_retrieved': sum(r.get('retrieved_count', 0) for r in results) / len(results) if results else 0,
            'llm_paraphrases': llm_paraphrases,
            'fallback_paraphrases': total_tests - llm_paraphrases
        }
        
        print(f"Total tests run: {stats['total_tests']}")
        print(f"Successful API calls: {stats['successful_tests']}")
        print(f"Failed API calls: {stats['failed_tests']}")
        print(f"LLM paraphrases: {stats['llm_paraphrases']}")
        print(f"Fallback paraphrases: {stats['fallback_paraphrases']}")
        print(f"Correct retrievals: {stats['correct_retrievals']}/{stats['tests_with_ground_truth']}")
        print(f"Recall rate: {stats['recall']:.1%}")
        print(f"Average memories retrieved: {stats['avg_retrieved']:.1f}")
        
        # Save detailed results with full memory details
        output_file = Path('benchmarks/results') / f'flask_test_{self.session_id}.json'
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        # Prepare detailed output
        detailed_output = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'flask_url': self.flask_url,
            'llm_url': self.llm_url,
            'seed': self.seed,
            'specific_memory_ids': self.specific_memory_ids,
            'statistics': stats,
            'test_cases': [],
            'reproduction_command': self._get_reproduction_command(raw_memories)
        }
        
        for result in results:
            test_case = {
                'original_memory': result.get('original_memory'),
                'altered_text': result.get('altered_text'),
                'paraphrase_method': result.get('paraphrase_method'),
                'expected_id': result.get('expected'),
                'found': result.get('found_expected'),
                'position': result.get('expected_position'),
                'retrieved_ids': result.get('retrieved_ids', []),
                'retrieved_count': result.get('retrieved_count', 0),
                'retrieved_memories': result.get('retrieved_memories', []),  # Add full memory details
                'memory_scores': result.get('memory_scores', {}),  # Add individual memory scores
                'scoring_weights': result.get('scoring_weights', {}),  # Add scoring weight configuration
                'block_info': result.get('block_info'),
                'llm_reply_preview': result.get('llm_reply', '')[:200]
            }
            detailed_output['test_cases'].append(test_case)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_output, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {output_file}")
        
        # Print reproduction command
        print(f"\nTo reproduce this exact test, run:")
        print(f"  {detailed_output['reproduction_command']}")
        
        return stats
    
    def _get_reproduction_command(self, raw_memories: List[Dict]) -> str:
        """Generate command to reproduce this exact test."""
        memory_ids = [m['memory_id'] for m in raw_memories]
        cmd = f"python benchmarks/test_flask_retrieval.py"
        
        if self.flask_url != "http://localhost:5001":
            cmd += f" --flask-url {self.flask_url}"
        if self.llm_url != "http://localhost:8000":
            cmd += f" --llm-url {self.llm_url}"
        
        # Use memory IDs for exact reproduction
        if memory_ids:
            cmd += f" --memory-ids {' '.join(memory_ids)}"
        elif self.seed is not None:
            cmd += f" --seed {self.seed}"
        
        cmd += f" --num-memories {len(raw_memories)}"
        
        return cmd
    
    def cleanup(self):
        """Clean up resources."""
        if self.db_conn:
            self.db_conn.close()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test JAM retrieval using Flask app')
    parser.add_argument('--flask-url', default='http://localhost:5001',
                       help='Flask app URL (default: http://localhost:5001)')
    parser.add_argument('--llm-url', default='http://localhost:8000',
                       help='LLM server URL (default: http://localhost:8000)')
    parser.add_argument('--num-memories', type=int, default=3,
                       help='Number of memories to test (default: 3)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible memory selection')
    parser.add_argument('--memory-ids', nargs='+', default=None,
                       help='Specific memory IDs to test (e.g., --memory-ids mem_123 mem_456)')
    parser.add_argument('--list-memories', action='store_true',
                       help='List available memory IDs and exit')
    
    args = parser.parse_args()
    
    # Handle --list-memories option
    if args.list_memories:
        import sqlite3
        con = sqlite3.connect(cfg.db_path)
        con.row_factory = sqlite3.Row
        cursor = con.cursor()
        
        print("\nAvailable memories with good raw_text:")
        print("-" * 80)
        
        cursor.execute("""
            SELECT memory_id, raw_text, who_id, when_ts
            FROM memories
            WHERE LENGTH(raw_text) > 50 AND LENGTH(raw_text) < 1000
                AND raw_text IS NOT NULL
            ORDER BY when_ts DESC
            LIMIT 50
        """)
        
        for row in cursor.fetchall():
            preview = row['raw_text'][:100].replace('\n', ' ')
            print(f"{row['memory_id']}: [{row['who_id']}] {preview}...")
        
        con.close()
        sys.exit(0)
    
    # Create tester with seed and memory_ids
    tester = FlaskRetrievalTester(args.flask_url, args.llm_url, 
                                  seed=args.seed, memory_ids=args.memory_ids)
    
    try:
        # Run tests
        stats = tester.run_test_suite(args.num_memories)
        
        # Return appropriate exit code
        if stats.get('recall', 0) < 0.5:
            print("\n[WARNING] Recall rate is below 50%")
            sys.exit(1)
        else:
            print("\n[OK] Tests completed successfully")
            sys.exit(0)
            
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
