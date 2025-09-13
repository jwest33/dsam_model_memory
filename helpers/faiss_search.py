#!/usr/bin/env python
"""
FAISS Search Helper Script
Adhoc tool for searching memories directly in the FAISS index
"""

import sys
import os
import json
import sqlite3
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.embedding import get_llama_embedder
from agentic_memory.config import cfg
from agentic_memory.retrieval import HybridRetriever, RetrievalQuery
from agentic_memory.types import Candidate

def get_embedding(text: str) -> np.ndarray:
    """Get embedding for a text query"""
    embedder = get_llama_embedder()
    embedding = embedder.encode(text)
    if len(embedding.shape) > 1:
        embedding = embedding[0]  # Get first embedding if batched
    # L2 normalize for cosine similarity
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding

def format_memory(memory_data: dict) -> str:
    """Format a memory for display"""
    output = []
    output.append(f"\n{'='*80}")
    output.append(f"Memory ID: {memory_data['memory_id']}")
    output.append(f"Created: {memory_data.get('created_at', 'Unknown')}")

    # 5W1H fields - handle both 'value' and direct field names
    if memory_data.get('who_list'):
        who = json.loads(memory_data['who_list']) if isinstance(memory_data['who_list'], str) else memory_data['who_list']
        output.append(f"WHO: {', '.join(who)}")

    what_field = memory_data.get('what') or memory_data.get('what_value')
    if what_field:
        # Try to parse as JSON array for better display
        try:
            what_items = json.loads(what_field)
            if isinstance(what_items, list):
                output.append(f"WHAT: {', '.join(what_items)}")
            else:
                output.append(f"WHAT: {what_field}")
        except (json.JSONDecodeError, TypeError):
            output.append(f"WHAT: {what_field}")

    if memory_data.get('when_list'):
        when = json.loads(memory_data['when_list']) if isinstance(memory_data['when_list'], str) else memory_data['when_list']
        output.append(f"WHEN: {', '.join(when)}")

    if memory_data.get('where_value'):
        output.append(f"WHERE: {memory_data['where_value']}")

    why_field = memory_data.get('why') or memory_data.get('why_value')
    if why_field:
        output.append(f"WHY: {why_field}")

    how_field = memory_data.get('how') or memory_data.get('how_value')
    if how_field:
        output.append(f"HOW: {how_field}")

    # Original text - check both 'text' and 'raw_text' fields
    text_field = memory_data.get('raw_text') or memory_data.get('text')
    if text_field:
        output.append(f"\nTEXT:\n{text_field[:500]}{'...' if len(text_field) > 500 else ''}")

    # Score if available
    if 'score' in memory_data:
        output.append(f"\nScore: {memory_data['score']:.4f}")

    return '\n'.join(output)

def search_memories(query: str, top_k: int = 10, show_vectors: bool = False):
    """Search for memories using FAISS index"""

    print(f"\nSearching for: '{query}'")
    print(f"Top K results: {top_k}")
    print("="*80)

    # Initialize components
    print("Loading FAISS index...")
    index = FaissIndex(
        dim=cfg.get('embedding_dim', 2048),
        index_path=cfg.get('index_path', './data/faiss.index')
    )
    print(f"Index loaded with {index.index.ntotal} vectors")

    print("Connecting to database...")
    store = MemoryStore(db_path=cfg.get('db_path', './data/amemory.sqlite3'))

    # Get embedding for query
    print("\nGenerating query embedding...")
    query_vec = get_embedding(query)

    if show_vectors:
        print(f"Query vector shape: {query_vec.shape}")
        print(f"Query vector (first 10 dims): {query_vec[:10]}")

    # Search FAISS index
    print(f"\nSearching FAISS index for top {top_k} results...")
    results = index.search(query_vec, top_k)

    if not results:
        print("No results found!")
        return

    print(f"Found {len(results)} results")

    # Fetch full memory data from database
    for i, (memory_id, score) in enumerate(results, 1):
        print(f"\n--- Result {i}/{len(results)} ---")

        # Get memory from database using fetch_memories
        memories = store.fetch_memories([memory_id])

        if memories and len(memories) > 0:
            memory_data = memories[0]
            # Add score to memory data for display
            memory_data['score'] = score

            # Check if vector exists in FAISS
            stored_vec = index.get_vector(memory_id)
            if stored_vec is None:
                print(f"WARNING: Vector not found in FAISS index for memory {memory_id}")

            print(format_memory(memory_data))

            if show_vectors and stored_vec is not None:
                # Show the stored vector
                print(f"\nStored vector (first 10 dims): {stored_vec[:10]}")
                # Calculate cosine similarity manually for verification
                manual_sim = np.dot(query_vec, stored_vec)
                print(f"Manual cosine similarity: {manual_sim:.4f}")
        else:
            print(f"Memory ID: {memory_id}")
            print(f"Score: {score:.4f}")
            print("(Memory not found in database)")

def list_recent_memories(limit: int = 10):
    """List most recent memories"""
    print(f"\nListing {limit} most recent memories")
    print("="*80)

    store = MemoryStore(db_path=cfg.get('db_path', './data/amemory.sqlite3'))

    # Get recent memories
    conn = sqlite3.connect(store.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM memories
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    for i, row in enumerate(rows, 1):
        memory_data = dict(row)
        print(format_memory(memory_data))

def lookup_memory(memory_id: str, show_vector: bool = False):
    """Look up a specific memory by ID"""
    print(f"\nLooking up memory: {memory_id}")
    print("="*80)

    # Connect to database
    print("Connecting to database...")
    store = MemoryStore(db_path=cfg.get('db_path', './data/amemory.sqlite3'))

    # Fetch the memory
    memories = store.fetch_memories([memory_id])

    if not memories:
        print(f"\nMemory ID '{memory_id}' not found in database!")
        return

    memory_data = memories[0]
    print("\nMemory found:")
    print(format_memory(memory_data))

    if show_vector:
        # Load FAISS index to get the vector
        print("\nLoading FAISS index for vector information...")
        index = FaissIndex(
            dim=cfg.get('embedding_dim', 2048),
            index_path=cfg.get('index_path', './data/faiss.index')
        )

        # Try to get vector from FAISS
        stored_vec = index.get_vector(memory_id)
        if stored_vec is not None:
            print(f"\nVector information:")
            print(f"  Dimension: {len(stored_vec)}")
            print(f"  First 10 dims: {stored_vec[:10]}")
            print(f"  L2 norm: {np.linalg.norm(stored_vec):.4f}")
        else:
            print("\nVector not found in FAISS index")

def compare_memory_to_query(memory_id: str, query: str):
    """Compare a specific memory against a query and show all scoring components"""
    print(f"\nComparing memory '{memory_id}' to query: '{query}'")
    print("="*80)

    # Connect to database and index
    store = MemoryStore(db_path=cfg.get('db_path', './data/amemory.sqlite3'))
    index = FaissIndex(
        dim=cfg.get('embedding_dim', 2048),
        index_path=cfg.get('index_path', './data/faiss.index')
    )

    # Initialize retriever with store and index
    retriever = HybridRetriever(store=store, index=index)

    # Fetch the memory
    memories = store.fetch_memories([memory_id])
    if not memories:
        print(f"\nMemory ID '{memory_id}' not found in database!")
        return

    memory_data = memories[0]

    # Display the memory
    print("\nMemory Details:")
    print(format_memory(memory_data))

    # Decompose the query
    print("\nQuery Decomposition:")
    decomposition = retriever.decompose_query(query)
    for key, value in decomposition.items():
        if value:
            print(f"  {key.upper()}: {value}")

    # Create retrieval query
    rq = RetrievalQuery(
        session_id='compare',
        text=decomposition.get('what', query),
        actor_hint=decomposition['who'].get('id') if decomposition.get('who') else None,
        temporal_hint=decomposition.get('when')
    )

    # Generate query embedding (matching the format used in flask_app.py)
    what_text = decomposition.get('what', query)
    why_text = decomposition.get('why', 'search')
    how_text = decomposition.get('how', 'query')
    embed_text = f"WHAT: {what_text}\nWHY: {why_text}\nHOW: {how_text}\nRAW: {query}"
    query_vec = get_embedding(embed_text)

    # Get the memory's vector
    memory_vec = index.get_vector(memory_id)

    # Calculate semantic similarity
    semantic_score = 0.0
    if memory_vec is not None:
        semantic_score = float(np.dot(query_vec, memory_vec))

    # Create a candidate object for scoring
    candidate = Candidate(
        memory_id=memory_id,
        score=0.0,
        semantic_score=semantic_score,
        recency_score=0.0,
        actor_score=0.0,
        temporal_score=0.0,
        usage_score=0.0,
        token_count=memory_data.get('token_count', 0)
    )

    # Calculate recency score
    created_at = memory_data.get('created_at', '')
    if created_at:
        try:
            created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            now = datetime.now(created_dt.tzinfo)
            age_hours = (now - created_dt).total_seconds() / 3600
            candidate.recency_score = retriever._calculate_recency_score(age_hours)
        except:
            pass

    # Calculate actor score
    if rq.actor_hint:
        who_list = memory_data.get('who_list', '[]')
        try:
            actors = json.loads(who_list) if isinstance(who_list, str) else who_list
            if rq.actor_hint in actors:
                candidate.actor_score = 1.0
        except:
            pass

    # Calculate temporal score
    if rq.temporal_hint:
        when_list = memory_data.get('when_list', '[]')
        try:
            when_items = json.loads(when_list) if isinstance(when_list, str) else when_list
            for item in when_items:
                if rq.temporal_hint.lower() in item.lower():
                    candidate.temporal_score = 1.0
                    break
        except:
            pass

    # Get usage score
    usage_stats = store.get_usage_stats([memory_id])
    if memory_id in usage_stats:
        stats = usage_stats[memory_id]
        accesses = stats.get('accesses', 0)
        candidate.usage_score = retriever._calculate_usage_score(accesses)

    # Get current weights from config
    weights = {
        'semantic': cfg.get('weight_semantic', 0.35),
        'recency': cfg.get('weight_recency', 0.20),
        'actor': cfg.get('weight_actor', 0.15),
        'temporal': cfg.get('weight_temporal', 0.10),
        'spatial': cfg.get('weight_spatial', 0.05),
        'usage': cfg.get('weight_usage', 0.15)
    }

    # Calculate total weighted score
    total_score = (
        weights['semantic'] * candidate.semantic_score +
        weights['recency'] * candidate.recency_score +
        weights['actor'] * candidate.actor_score +
        weights['temporal'] * candidate.temporal_score +
        weights['usage'] * candidate.usage_score
    )

    # Display scores
    print("\n" + "="*80)
    print("SCORING BREAKDOWN")
    print("="*80)

    print("\nIndividual Scores:")
    print(f"  Semantic:  {candidate.semantic_score:.4f} (weight: {weights['semantic']:.2f}) = {weights['semantic'] * candidate.semantic_score:.4f}")
    print(f"  Recency:   {candidate.recency_score:.4f} (weight: {weights['recency']:.2f}) = {weights['recency'] * candidate.recency_score:.4f}")
    print(f"  Actor:     {candidate.actor_score:.4f} (weight: {weights['actor']:.2f}) = {weights['actor'] * candidate.actor_score:.4f}")
    print(f"  Temporal:  {candidate.temporal_score:.4f} (weight: {weights['temporal']:.2f}) = {weights['temporal'] * candidate.temporal_score:.4f}")
    print(f"  Spatial:   0.0000 (weight: {weights['spatial']:.2f}) = 0.0000")
    print(f"  Usage:     {candidate.usage_score:.4f} (weight: {weights['usage']:.2f}) = {weights['usage'] * candidate.usage_score:.4f}")

    print(f"\nTotal Weighted Score: {total_score:.4f}")
    print(f"Token Count: {candidate.token_count}")

    # Show how this would rank
    print("\nContext:")
    if semantic_score > 0.7:
        print("  - High semantic similarity (>0.7) - Very relevant to query")
    elif semantic_score > 0.4:
        print("  - Moderate semantic similarity (0.4-0.7) - Somewhat relevant")
    else:
        print("  - Low semantic similarity (<0.4) - Weak relevance")

    if candidate.recency_score > 0.5:
        print("  - Recent memory - Higher priority in results")

    if candidate.actor_score > 0:
        print("  - Actor match - Query mentions same actor as memory")

    if candidate.temporal_score > 0:
        print("  - Temporal match - Query time reference matches memory")

    if candidate.usage_score > 0.5:
        print("  - Frequently accessed memory")

def check_zero_embeddings(limit: Optional[int] = None, fix: bool = False, delete: bool = False):
    """Find all memories with zero or missing embeddings"""
    print("\nChecking for memories with zero or missing embeddings")
    print("="*80)

    # Initialize components
    store = MemoryStore(db_path=cfg.get('db_path', './data/amemory.sqlite3'))
    index = FaissIndex(
        dim=cfg.get('embedding_dim', 2048),
        index_path=cfg.get('index_path', './data/faiss.index')
    )

    # Get all memory IDs from database
    conn = sqlite3.connect(store.db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT memory_id, created_at, raw_text FROM memories ORDER BY created_at DESC"
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    print(f"Checking {len(rows)} memories...")

    zero_embeddings = []
    missing_embeddings = []
    valid_embeddings = []

    for row in rows:
        memory_id = row['memory_id']
        created_at = row['created_at']
        raw_text = row['raw_text'][:100] if row['raw_text'] else 'N/A'

        # Try to get vector from FAISS
        vec = index.get_vector(memory_id)

        if vec is None:
            missing_embeddings.append({
                'memory_id': memory_id,
                'created_at': created_at,
                'text_preview': raw_text
            })
        elif np.allclose(vec, 0):
            # Check if all values are zero
            zero_embeddings.append({
                'memory_id': memory_id,
                'created_at': created_at,
                'text_preview': raw_text,
                'norm': np.linalg.norm(vec)
            })
        else:
            valid_embeddings.append(memory_id)

    # Report findings
    print(f"\nResults:")
    print(f"  Valid embeddings: {len(valid_embeddings)}")
    print(f"  Zero embeddings: {len(zero_embeddings)}")
    print(f"  Missing from index: {len(missing_embeddings)}")

    if zero_embeddings:
        print(f"\n{'='*80}")
        print(f"MEMORIES WITH ZERO EMBEDDINGS ({len(zero_embeddings)})")
        print(f"{'='*80}")
        for mem in zero_embeddings[:20]:  # Show first 20
            print(f"\nID: {mem['memory_id']}")
            print(f"Created: {mem['created_at']}")
            print(f"Text: {mem['text_preview']}...")
            print(f"Vector norm: {mem['norm']:.6f}")

        if len(zero_embeddings) > 20:
            print(f"\n... and {len(zero_embeddings) - 20} more")

        # Save full list to file
        if len(zero_embeddings) > 0:
            output_file = 'zero_embeddings.txt'
            with open(output_file, 'w') as f:
                f.write("Memories with zero embeddings:\n")
                f.write("="*80 + "\n\n")
                for mem in zero_embeddings:
                    f.write(f"ID: {mem['memory_id']}\n")
                    f.write(f"Created: {mem['created_at']}\n")
                    f.write(f"Text: {mem['text_preview']}...\n")
                    f.write("-"*40 + "\n")
            print(f"\nFull list saved to: {output_file}")

    if missing_embeddings:
        print(f"\n{'='*80}")
        print(f"MEMORIES MISSING FROM INDEX ({len(missing_embeddings)})")
        print(f"{'='*80}")
        for mem in missing_embeddings[:20]:  # Show first 20
            print(f"\nID: {mem['memory_id']}")
            print(f"Created: {mem['created_at']}")
            print(f"Text: {mem['text_preview']}...")

        if len(missing_embeddings) > 20:
            print(f"\n... and {len(missing_embeddings) - 20} more")

        # Save full list to file
        if len(missing_embeddings) > 0:
            output_file = 'missing_embeddings.txt'
            with open(output_file, 'w') as f:
                f.write("Memories missing from FAISS index:\n")
                f.write("="*80 + "\n\n")
                for mem in missing_embeddings:
                    f.write(f"ID: {mem['memory_id']}\n")
                    f.write(f"Created: {mem['created_at']}\n")
                    f.write(f"Text: {mem['text_preview']}...\n")
                    f.write("-"*40 + "\n")
            print(f"\nFull list saved to: {output_file}")

    if delete and (zero_embeddings or missing_embeddings):
        print(f"\n{'='*80}")
        print("DELETE MODE: Removing memories with zero/missing embeddings")
        print(f"{'='*80}")

        all_to_delete = []
        for mem in zero_embeddings:
            all_to_delete.append(mem['memory_id'])
        for mem in missing_embeddings:
            all_to_delete.append(mem['memory_id'])

        print(f"\nWARNING: This will permanently delete {len(all_to_delete)} memories!")
        print("Press Ctrl+C to cancel, or Enter to continue...")
        try:
            input()
        except KeyboardInterrupt:
            print("\nDeletion cancelled.")
            return zero_embeddings, missing_embeddings

        deleted_count = 0
        failed_count = 0

        print(f"\nDeleting {len(all_to_delete)} memories...")

        # Delete from SQLite database
        conn = sqlite3.connect(store.db_path)
        cursor = conn.cursor()

        for i, memory_id in enumerate(all_to_delete, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(all_to_delete)}")

            try:
                # Delete from database
                cursor.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))

                # Try to remove from FAISS index (it might not exist there)
                try:
                    index.remove(memory_id)
                except:
                    pass  # Memory might not be in index

                deleted_count += 1
            except Exception as e:
                print(f"  Failed to delete {memory_id}: {e}")
                failed_count += 1

        # Commit database changes
        conn.commit()
        conn.close()

        # Save the FAISS index
        index.save()

        print(f"\nDeletion complete:")
        print(f"  Deleted: {deleted_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Database and index updated.")

    elif fix and (zero_embeddings or missing_embeddings):
        print(f"\n{'='*80}")
        print("FIX MODE: Regenerating embeddings for affected memories")
        print(f"{'='*80}")

        embedder = get_llama_embedder()
        fixed_count = 0
        failed_count = 0

        all_to_fix = []
        for mem in zero_embeddings:
            all_to_fix.append(mem['memory_id'])
        for mem in missing_embeddings:
            all_to_fix.append(mem['memory_id'])

        print(f"\nRegenerating embeddings for {len(all_to_fix)} memories...")

        for i, memory_id in enumerate(all_to_fix, 1):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(all_to_fix)}")

            # Fetch full memory data
            memories = store.fetch_memories([memory_id])
            if not memories:
                failed_count += 1
                continue

            memory = memories[0]

            # Generate embedding text (matching format from storage)
            what_text = memory.get('what_value', memory.get('what', ''))
            why_text = memory.get('why_value', memory.get('why', ''))
            how_text = memory.get('how_value', memory.get('how', ''))
            raw_text = memory.get('raw_text', '')

            embed_text = f"WHAT: {what_text}\nWHY: {why_text}\nHOW: {how_text}\nRAW: {raw_text}"

            try:
                # Generate embedding
                embedding = embedder.encode(embed_text)
                if len(embedding.shape) > 1:
                    embedding = embedding[0]

                # L2 normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                # First remove if exists, then add
                try:
                    index.remove(memory_id)
                except:
                    pass  # Memory might not be in index

                # Now add the new embedding
                index.add(memory_id, embedding)
                fixed_count += 1
            except Exception as e:
                print(f"  Failed to fix {memory_id}: {e}")
                failed_count += 1

        # Save the index
        index.save()

        print(f"\nFix complete:")
        print(f"  Fixed: {fixed_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Index saved to: {index.index_path}")

    return zero_embeddings, missing_embeddings

def show_stats():
    """Show database and index statistics"""
    print("\nMemory System Statistics")
    print("="*80)

    # Index stats
    index = FaissIndex(
        dim=cfg.get('embedding_dim', 2048),
        index_path=cfg.get('index_path', './data/faiss.index')
    )
    print(f"FAISS Index: {index.index.ntotal} vectors")
    print(f"Index dimension: {index.dim}")
    print(f"Index type: {type(index.index).__name__}")

    # Database stats
    store = MemoryStore(db_path=cfg.get('db_path', './data/amemory.sqlite3'))
    conn = sqlite3.connect(store.db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM memories")
    total_memories = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM memories")
    min_ts, max_ts = cursor.fetchone()

    # Count by actor
    cursor.execute("""
        SELECT who_list, COUNT(*) as count
        FROM memories
        WHERE who_list IS NOT NULL
        GROUP BY who_list
        ORDER BY count DESC
        LIMIT 10
    """)
    top_actors = cursor.fetchall()

    conn.close()

    print(f"\nDatabase: {total_memories} memories")
    print(f"Date range: {min_ts} to {max_ts}")

    if top_actors:
        print("\nTop actors:")
        for who_list, count in top_actors:
            try:
                actors = json.loads(who_list) if who_list else []
                if actors:
                    print(f"  {', '.join(actors[:2])}: {count} memories")
            except:
                pass

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='FAISS Memory Search Helper')
    parser.add_argument('command', choices=['search', 'lookup', 'compare', 'recent', 'stats', 'check-embeddings'],
                       help='Command to run (search: query memories, lookup: find by ID, compare: compare memory to query, recent: show recent, stats: show statistics, check-embeddings: find zero/missing embeddings)')
    parser.add_argument('query', nargs='?', default='',
                       help='Search query for search command, memory ID for lookup command, or memory ID for compare command')
    parser.add_argument('--compare-query', type=str,
                       help='Query text for compare command')
    parser.add_argument('-k', '--top-k', type=int, default=10,
                       help='Number of results to return (default: 10)')
    parser.add_argument('-v', '--vectors', action='store_true',
                       help='Show vector information')
    parser.add_argument('--limit', type=int, default=10,
                       help='Limit for recent command (default: 10)')
    parser.add_argument('--fix', action='store_true',
                       help='Fix zero/missing embeddings by regenerating them')
    parser.add_argument('--delete', action='store_true',
                       help='Delete memories with zero/missing embeddings from database and index')

    args = parser.parse_args()

    try:
        if args.command == 'search':
            if not args.query:
                print("Error: Search query required")
                sys.exit(1)
            search_memories(args.query, args.top_k, args.vectors)
        elif args.command == 'lookup':
            if not args.query:
                print("Error: Memory ID required for lookup")
                sys.exit(1)
            lookup_memory(args.query, args.vectors)
        elif args.command == 'compare':
            if not args.query:
                print("Error: Memory ID required for compare")
                sys.exit(1)
            if not args.compare_query:
                print("Error: Query text required for compare (use --compare-query)")
                sys.exit(1)
            compare_memory_to_query(args.query, args.compare_query)
        elif args.command == 'recent':
            list_recent_memories(args.limit)
        elif args.command == 'stats':
            show_stats()
        elif args.command == 'check-embeddings':
            check_zero_embeddings(limit=args.limit if args.limit != 10 else None, fix=args.fix, delete=args.delete)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
