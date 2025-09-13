#!/usr/bin/env python3
"""
Rebuild FAISS index and embeddings table with proper format.
This ensures all memories have embeddings and are searchable.
"""

import sqlite3
import numpy as np
import faiss
import os
import shutil
from datetime import datetime
from tqdm import tqdm
from agentic_memory.embedding import get_llama_embedder
from agentic_memory.config import Config

def backup_files():
    """Backup existing index and database"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Backup FAISS index
    if os.path.exists('faiss.index'):
        backup_name = f'faiss.index.backup_{timestamp}'
        shutil.copy2('faiss.index', backup_name)
        print(f"[OK] Backed up FAISS index to {backup_name}")

    # Backup index map
    if os.path.exists('faiss.index.map'):
        backup_name = f'faiss.index.map.backup_{timestamp}'
        shutil.copy2('faiss.index.map', backup_name)
        print(f"[OK] Backed up index map to {backup_name}")

    print()

def rebuild_index():
    """Rebuild FAISS index and embeddings table"""

    print("=== FAISS Index and Embeddings Rebuilder ===\n")

    # Backup existing files
    print("Step 1: Backing up existing files...")
    backup_files()

    # Initialize components
    print("Step 2: Initializing components...")
    cfg = Config()
    embedder = get_llama_embedder()

    # Get embedding dimension
    test_vec = embedder.encode(["test"], normalize_embeddings=True)[0]
    embed_dim = test_vec.shape[0]
    print(f"[OK] Embedding dimension: {embed_dim}")

    # Connect to database
    con = sqlite3.connect('amemory.sqlite3')
    con.row_factory = sqlite3.Row

    # Get all memories
    print("\nStep 3: Loading memories from database...")
    rows = con.execute("""
        SELECT memory_id, what, why, how, raw_text
        FROM memories
        ORDER BY created_at
    """).fetchall()

    total_memories = len(rows)
    print(f"[OK] Found {total_memories} memories to process")

    if total_memories == 0:
        print("No memories found. Exiting.")
        return

    # Create new FAISS index
    print("\nStep 4: Creating new FAISS index...")
    index = faiss.IndexHNSWFlat(embed_dim, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = 64
    index.hnsw.efConstruction = 80
    print(f"[OK] Created HNSW index with dimension {embed_dim}")

    # Process memories in batches
    print("\nStep 5: Generating embeddings and rebuilding index...")
    batch_size = 32
    id_map = []

    # Clear existing embeddings table
    print("  Clearing old embeddings table...")
    con.execute("DELETE FROM embeddings")
    con.commit()

    # Process all memories
    for i in tqdm(range(0, total_memories, batch_size), desc="Processing batches"):
        batch_rows = rows[i:i+batch_size]
        batch_texts = []
        batch_ids = []

        # Prepare embedding texts in the correct format
        for row in batch_rows:
            memory_id = row['memory_id']
            what = row['what'] or ''
            why = row['why'] or ''
            how = row['how'] or ''
            raw_text = row['raw_text'] or ''

            # Format embedding text to match storage format
            embed_text = f"WHAT: {what}\nWHY: {why}\nHOW: {how}\nRAW: {raw_text}"
            batch_texts.append(embed_text)
            batch_ids.append(memory_id)

        # Generate embeddings for batch
        if batch_texts:
            embeddings = embedder.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)

            # Add to FAISS index
            embeddings_float32 = embeddings.astype('float32')
            index.add(embeddings_float32)

            # Add to ID map
            id_map.extend(batch_ids)

            # Store in database
            for memory_id, embedding in zip(batch_ids, embeddings):
                embedding_bytes = embedding.astype('float32').tobytes()
                con.execute(
                    "INSERT INTO embeddings (memory_id, dim, vector) VALUES (?, ?, ?)",
                    (memory_id, embed_dim, embedding_bytes)
                )

            # Commit batch
            con.commit()

    print(f"\n[OK] Processed {total_memories} memories")
    print(f"[OK] FAISS index contains {index.ntotal} vectors")

    # Save FAISS index
    print("\nStep 6: Saving FAISS index...")
    faiss.write_index(index, 'faiss.index')
    print("[OK] Saved FAISS index to faiss.index")

    # Save ID map
    with open('faiss.index.map', 'w', encoding='utf-8') as f:
        for memory_id in id_map:
            f.write(memory_id + '\n')
    print(f"[OK] Saved ID map with {len(id_map)} entries")

    # Verify embeddings count
    embed_count = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    print(f"\nStep 7: Verification")
    print(f"[OK] Embeddings table contains {embed_count} entries")
    print(f"[OK] FAISS index contains {index.ntotal} vectors")
    print(f"[OK] ID map contains {len(id_map)} entries")

    if embed_count == index.ntotal == len(id_map) == total_memories:
        print("\n[SUCCESS] All counts match! Index rebuilt successfully.")
    else:
        print("\n[WARNING] Counts don't match:")
        print(f"   Memories: {total_memories}")
        print(f"   Embeddings: {embed_count}")
        print(f"   FAISS vectors: {index.ntotal}")
        print(f"   ID map entries: {len(id_map)}")

    con.close()

    print("\n=== Rebuild Complete ===")
    print("\nNote: The embeddings are now in the format:")
    print("  WHAT: {what}\\nWHY: {why}\\nHOW: {how}\\nRAW: {raw_text}")
    print("\nSearch queries will need to use the same format for best results.")

if __name__ == "__main__":
    try:
        rebuild_index()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Rebuild interrupted")
    except Exception as e:
        print(f"\n\n[ERROR] Error during rebuild: {e}")
        import traceback
        traceback.print_exc()
