"""
Analyze conceptual similarity between memory groups
Shows pairwise similarities and compares to merge threshold
"""

import numpy as np
from typing import List, Dict, Tuple
import json
from memory.memory_store import MemoryStore
from memory.chromadb_store import ChromaDBStore
from memory.multi_dimensional_merger import MultiDimensionalMerger
from memory.dual_space_encoder import DualSpaceEncoder
from models.merge_types import MergeType
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import sys

def get_conceptual_groups():
    """Get all conceptual merge groups from ChromaDB"""
    chroma_store = ChromaDBStore()
    
    try:
        # Access the conceptual_merges collection directly
        conceptual_collection = chroma_store.client.get_collection('conceptual_merges')
        result = conceptual_collection.get()
        
        if not result['ids']:
            print("No conceptual groups found in memory store")
            print("\nTo create some sample data, run:")
            print("  python load_llm_dataset.py")
            return []
        
        # Convert ChromaDB results to group format
        groups = []
        for i, group_id in enumerate(result['ids']):
            metadata = result['metadatas'][i] if i < len(result['metadatas']) else {}
            
            group = {
                'id': group_id,
                'primary_field': metadata.get('primary_field', ''),
                'secondary_field': metadata.get('secondary_field', ''),
                'group_what': metadata.get('group_what', ''),
                'group_why': metadata.get('group_why', ''),
                'size': int(metadata.get('component_count', 1)),
                'embeddings': result['embeddings'][i] if result['embeddings'] and i < len(result['embeddings']) else None
            }
            groups.append(group)
        
        print(f"Found {len(groups)} conceptual groups")
        return groups
        
    except Exception as e:
        print(f"Error accessing conceptual groups: {e}")
        
        # Try alternative approach through MemoryStore
        try:
            store = MemoryStore()
            merger = MultiDimensionalMerger(store)
            all_groups = merger.get_all_merge_groups()
            
            conceptual_groups = []
            for group_id, group_data in all_groups.items():
                if group_data.get('type') == 'CONCEPTUAL':
                    group = {
                        'id': group_id,
                        'primary_field': group_data.get('key', ''),
                        'secondary_field': '',
                        'group_what': group_data.get('key', ''),
                        'group_why': '',
                        'size': len(group_data.get('raw_event_ids', [])),
                        'embeddings': None
                    }
                    conceptual_groups.append(group)
            
            if conceptual_groups:
                print(f"Found {len(conceptual_groups)} conceptual groups (via merger)")
                return conceptual_groups
                
        except Exception as e2:
            print(f"Alternative approach also failed: {e2}")
        
        return []

def compute_group_embeddings(groups: List[Dict]) -> Dict[str, np.ndarray]:
    """Compute average embeddings for each group"""
    encoder = DualSpaceEncoder()
    group_embeddings = {}
    
    for group in groups:
        group_id = group['id']
        
        # Check if we already have embeddings from ChromaDB
        if group.get('embeddings') is not None:
            group_embeddings[group_id] = np.array(group['embeddings'])
        else:
            # Get the fields for embedding
            primary_field = group.get('primary_field', '')
            secondary_field = group.get('secondary_field', '')
            group_what = group.get('group_what', '')
            group_why = group.get('group_why', '')
            
            # Create 5W1H dictionary for the encoder
            five_w1h = {
                'what': group_what or primary_field or 'Unknown',
                'why': group_why or secondary_field or 'Unknown purpose',
                'who': '',  # Not typically used for conceptual groups
                'where': '',  # Not typically used for conceptual groups
                'when': '',  # Not typically used for conceptual groups
                'how': ''  # Could be filled if available
            }
            
            # Generate embedding - returns dict with euclidean_anchor and hyperbolic_anchor
            embeddings_dict = encoder.encode(five_w1h)
            group_embeddings[group_id] = embeddings_dict['euclidean_anchor']
        
    return group_embeddings

def calculate_pairwise_similarities(groups: List[Dict], embeddings: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Calculate pairwise cosine similarities between all groups"""
    
    group_ids = list(embeddings.keys())
    n_groups = len(group_ids)
    
    # Create similarity matrix
    similarity_matrix = np.zeros((n_groups, n_groups))
    
    for i, id1 in enumerate(group_ids):
        for j, id2 in enumerate(group_ids):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                emb1 = embeddings[id1].reshape(1, -1)
                emb2 = embeddings[id2].reshape(1, -1)
                similarity = cosine_similarity(emb1, emb2)[0, 0]
                similarity_matrix[i, j] = similarity
    
    # Create DataFrame for better visualization
    df = pd.DataFrame(similarity_matrix, index=group_ids, columns=group_ids)
    return df

def get_group_descriptions(groups: List[Dict]) -> Dict[str, str]:
    """Extract meaningful descriptions for each group"""
    descriptions = {}
    
    for group in groups:
        group_id = group['id']
        
        # Try to get the most descriptive fields
        what = group.get('group_what', '')
        why = group.get('group_why', '')
        primary = group.get('primary_field', '')
        secondary = group.get('secondary_field', '')
        
        # Build description
        desc_parts = []
        if what:
            desc_parts.append(f"WHAT: {what}")
        if why:
            desc_parts.append(f"WHY: {why}")
        if not desc_parts:  # Fallback
            if primary:
                desc_parts.append(f"PRIMARY: {primary}")
            if secondary:
                desc_parts.append(f"SECONDARY: {secondary}")
        
        descriptions[group_id] = " | ".join(desc_parts) if desc_parts else "No description"
        
    return descriptions

def main():
    print("=" * 80)
    print("CONCEPTUAL GROUP SIMILARITY ANALYSIS")
    print("=" * 80)
    
    # Get the merge threshold from configuration
    CONCEPTUAL_THRESHOLD = 0.35  # From CLAUDE.md
    print(f"\nMerge Threshold: {CONCEPTUAL_THRESHOLD}")
    print("(Groups with similarity >= {:.1%} would be merged)\n".format(CONCEPTUAL_THRESHOLD))
    
    # Get groups
    groups = get_conceptual_groups()
    if not groups:
        return
    
    # Get descriptions
    descriptions = get_group_descriptions(groups)
    
    # Display groups
    print("\nConceptual Groups Found:")
    print("-" * 80)
    for i, group in enumerate(groups, 1):
        group_id = group['id']
        size = group.get('size', 0)
        desc = descriptions[group_id]
        print(f"{i}. Group {group_id[:8]}... (size: {size})")
        print(f"   {desc}")
    
    # Compute embeddings
    print("\nComputing group embeddings...")
    embeddings = compute_group_embeddings(groups)
    
    # Calculate similarities
    print("Calculating pairwise similarities...")
    similarity_df = calculate_pairwise_similarities(groups, embeddings)
    
    # Find interesting pairs (high similarity but not merged)
    print("\n" + "=" * 80)
    print("PAIRWISE SIMILARITY ANALYSIS")
    print("=" * 80)
    
    pairs_analyzed = set()
    interesting_pairs = []
    
    for i, id1 in enumerate(similarity_df.index):
        for j, id2 in enumerate(similarity_df.columns):
            if i >= j:  # Skip diagonal and duplicates
                continue
            
            similarity = similarity_df.loc[id1, id2]
            pair_key = tuple(sorted([id1, id2]))
            
            if pair_key not in pairs_analyzed:
                pairs_analyzed.add(pair_key)
                interesting_pairs.append((id1, id2, similarity))
    
    # Sort by similarity
    interesting_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Display results
    print("\nMost Similar Group Pairs:")
    print("-" * 80)
    
    for id1, id2, similarity in interesting_pairs[:10]:  # Top 10 pairs
        desc1 = descriptions[id1]
        desc2 = descriptions[id2]
        
        # Determine if they would merge
        would_merge = similarity >= CONCEPTUAL_THRESHOLD
        status = "🔴 WOULD MERGE" if would_merge else "🟢 DISTINCT"
        
        print(f"\nSimilarity: {similarity:.3f} ({similarity*100:.1f}%) - {status}")
        print(f"  Group 1: {id1[:8]}...")
        print(f"    {desc1}")
        print(f"  Group 2: {id2[:8]}...")
        print(f"    {desc2}")
    
    # Statistical summary
    print("\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)
    
    all_similarities = []
    for id1, id2, sim in interesting_pairs:
        all_similarities.append(sim)
    
    if all_similarities:
        print(f"Number of group pairs: {len(all_similarities)}")
        print(f"Average similarity: {np.mean(all_similarities):.3f}")
        print(f"Median similarity: {np.median(all_similarities):.3f}")
        print(f"Min similarity: {np.min(all_similarities):.3f}")
        print(f"Max similarity: {np.max(all_similarities):.3f}")
        
        # Count how many would merge
        would_merge_count = sum(1 for sim in all_similarities if sim >= CONCEPTUAL_THRESHOLD)
        print(f"\nPairs above merge threshold ({CONCEPTUAL_THRESHOLD}): {would_merge_count}/{len(all_similarities)}")
        
        # Distribution
        print("\nSimilarity Distribution:")
        bins = [0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(all_similarities, bins=bins)
        
        for i in range(len(bins)-1):
            bar_length = int(hist[i] * 50 / max(hist)) if max(hist) > 0 else 0
            bar = '█' * bar_length
            label = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
            if bins[i] <= CONCEPTUAL_THRESHOLD < bins[i+1]:
                label += " ← THRESHOLD"
            print(f"  {label:12} [{hist[i]:3}] {bar}")
    
    # Show full similarity matrix for small numbers of groups
    if len(groups) <= 8:
        print("\n" + "=" * 80)
        print("FULL SIMILARITY MATRIX")
        print("=" * 80)
        print("\nRows/Columns are group IDs (first 8 chars)")
        print("Values show similarity (0.0 to 1.0)")
        print(f"Values >= {CONCEPTUAL_THRESHOLD} would trigger merging\n")
        
        # Shorten IDs for display
        short_ids = [id[:8] for id in similarity_df.index]
        display_df = similarity_df.copy()
        display_df.index = short_ids
        display_df.columns = short_ids
        
        # Format for display
        pd.set_option('display.float_format', '{:.3f}'.format)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(display_df)

if __name__ == "__main__":
    main()
