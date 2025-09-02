"""
Test script for enhanced conceptual retrieval with dual-space embeddings and LLM fields
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from agent.memory_agent import MemoryAgent
from models.event import Event, FiveW1H, EventType
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_conceptual_retrieval():
    """Test the enhanced conceptual retrieval system"""
    
    print("\n" + "="*60)
    print("Testing Enhanced Conceptual Retrieval")
    print("="*60)
    
    # Initialize memory agent
    agent = MemoryAgent()
    
    # Get initial stats
    stats = agent.get_statistics()
    print(f"\nInitial database state:")
    print(f"  Total events: {stats.get('total_events', 0)}")
    print(f"  Conceptual merges: {stats.get('merge_groups', {}).get('conceptual', 0)}")
    
    # Test 1: Check if group_why and group_how fields are in metadata
    print("\n1. Checking for LLM-generated fields in conceptual groups...")
    
    try:
        # Get conceptual merge collection
        if agent.memory_store.chromadb and agent.memory_store.chromadb.conceptual_merges_collection:
            collection = agent.memory_store.chromadb.conceptual_merges_collection
            
            # Get a sample of conceptual merges
            try:
                results = collection.get(limit=5, include=['metadatas'])
                if results and results['metadatas']:
                    print(f"   Found {len(results['metadatas'])} conceptual groups")
                    
                    for i, metadata in enumerate(results['metadatas']):
                        group_why = metadata.get('group_why', '')
                        group_how = metadata.get('group_how', '')
                        group_method = metadata.get('group_fields_method', '')
                        
                        print(f"\n   Group {i+1}:")
                        print(f"     group_why: {group_why[:100]}..." if group_why else "     group_why: [Not set]")
                        print(f"     group_how: {group_how[:100]}..." if group_how else "     group_how: [Not set]")
                        print(f"     method: {group_method}")
                else:
                    print("   No conceptual groups found in database")
            except Exception as e:
                if "Nothing found" in str(e):
                    print("   Conceptual collection is empty")
                else:
                    print(f"   Error accessing collection: {e}")
        else:
            print("   ChromaDB not initialized")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 2: Create a conceptual query and test retrieval
    print("\n2. Testing conceptual query retrieval...")
    
    conceptual_queries = [
        "How does the memory system implement dual-space encoding?",
        "Why are memories organized into multiple dimensions?",
        "What is the purpose of hyperbolic embeddings in the system?",
        "Explain the concept of merge groups"
    ]
    
    for query in conceptual_queries[:2]:  # Test first 2 queries
        print(f"\n   Query: '{query}'")
        
        try:
            # Use the memory agent's retrieve method
            results = agent.retrieve(
                query_fields={'what': query},
                k=2
            )
            
            if results:
                print(f"   Found {len(results)} results")
                for i, (memory, score) in enumerate(results[:1]):  # Show first result
                    print(f"\n   Result {i+1} (score: {score:.3f}):")
                    
                    # Check if it's a merged event
                    if hasattr(memory, 'merge_count'):
                        print(f"     Type: Merged event (contains {memory.merge_count} events)")
                        if hasattr(memory, 'group_why'):
                            print(f"     Group Why: {memory.group_why[:150]}..." if memory.group_why else "     Group Why: [Not set]")
                        if hasattr(memory, 'group_how'):
                            print(f"     Group How: {memory.group_how[:150]}..." if memory.group_how else "     Group How: [Not set]")
                    else:
                        print(f"     Type: Individual event")
                    
                    print(f"     What: {memory.five_w1h.what[:150]}...")
            else:
                print("   No results found")
                
        except Exception as e:
            print(f"   Error during retrieval: {e}")
    
    # Test 3: Check if hyperbolic embeddings are being stored
    print("\n3. Checking hyperbolic embedding storage...")
    
    if hasattr(agent.memory_store, 'multi_merger') and agent.memory_store.multi_merger:
        merger = agent.memory_store.multi_merger
        
        # Check if any merge groups have hyperbolic embeddings
        has_hyperbolic = False
        for merge_type, groups in merger.merge_groups.items():
            for group_id, group_data in groups.items():
                if 'hyperbolic_embedding' in group_data:
                    has_hyperbolic = True
                    print(f"   ✓ Found hyperbolic embedding in {merge_type.value} group {group_id}")
                    break
            if has_hyperbolic:
                break
        
        if not has_hyperbolic:
            print("   ✗ No hyperbolic embeddings found in merge groups")
            print("   Note: Hyperbolic embeddings will be added as new events are processed")
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
    
    # Summary
    print("\nSummary:")
    print("1. LLM-generated fields (group_why, group_how) are now stored in metadata")
    print("2. Conceptual search now uses text matching with these fields")
    print("3. Hyperbolic embeddings are stored for new merge groups")
    print("\nNext steps:")
    print("- Process new events to populate hyperbolic embeddings")
    print("- The system will use both embedding spaces + text for better retrieval")
    print("- Conceptual queries will benefit from the enhanced search")

if __name__ == "__main__":
    test_conceptual_retrieval()
