"""
Dual-storage memory system with raw and processed stores

Combines ChromaDB vector storage with Modern Hopfield Network.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

import numpy as np

from config import get_config
from models.event import Event, EventType, FiveW1H
from models.memory_block import MemoryBlock
from embedding.embedder import FiveW1HEmbedder, TextEmbedder
from memory.hopfield import ModernHopfieldNetwork
from memory.block_manager import MemoryBlockManager

logger = logging.getLogger(__name__)

class MemoryStore:
    """
    Dual-storage memory system:
    1. Raw store: All events with timestamps (ChromaDB/JSON)
    2. Processed store: Deduplicated, salience-filtered (MHN + ChromaDB)
    """
    
    def __init__(self, config=None):
        """Initialize memory stores"""
        self.config = config or get_config()
        
        # Embedding system
        self.text_embedder = TextEmbedder(self.config.embedding)
        self.embedder = FiveW1HEmbedder(self.text_embedder)
        
        # Modern Hopfield Network for processed memories
        self.hopfield = ModernHopfieldNetwork(self.config.memory)
        
        # Memory Block Manager for content-addressable linking
        self.block_manager = MemoryBlockManager(self.embedder)
        
        # Storage backends
        self.use_chromadb = self.config.storage.use_chromadb
        self.raw_collection = None
        self.processed_collection = None
        
        # Memory tracking
        self.raw_memories: List[Event] = []
        self.processed_memories: List[Event] = []
        self.processed_blocks: List[MemoryBlock] = []  # Memory blocks
        self.episode_map: Dict[str, List[str]] = {}  # episode_id -> event_ids
        
        # Statistics
        self.operation_count = 0
        self.last_save = datetime.utcnow()
        
        # Initialize storage
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize storage backends"""
        # Create state directory
        self.config.storage.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB if available
        if self.use_chromadb:
            try:
                import chromadb
                from chromadb.config import Settings
                
                # Create persistent client
                self.chroma_client = chromadb.PersistentClient(
                    path=str(self.config.storage.chromadb_path),
                    settings=Settings(anonymized_telemetry=False)
                )
                
                # Get or create collections
                self.raw_collection = self.chroma_client.get_or_create_collection(
                    name=self.config.storage.raw_collection_name,
                    metadata={"description": "Raw event memories"}
                )
                
                self.processed_collection = self.chroma_client.get_or_create_collection(
                    name=self.config.storage.processed_collection_name,
                    metadata={"description": "Processed and deduplicated memories"}
                )
                
                logger.info("Initialized ChromaDB collections")
                
            except ImportError:
                logger.warning("ChromaDB not available, falling back to JSON storage")
                self.use_chromadb = False
            except Exception as e:
                logger.warning(f"Failed to initialize ChromaDB: {e}, falling back to JSON")
                self.use_chromadb = False
        
        # Load existing memories from disk
        self._load_from_disk()
    
    def store_event(
        self,
        event: Event,
        skip_salience_check: bool = False
    ) -> Tuple[bool, str]:
        """
        Store an event in memory with content-addressable block linking
        
        Args:
            event: Event to store
            skip_salience_check: Store regardless of salience threshold
        
        Returns:
            (success, message)
        """
        try:
            # Always store in raw memory
            self._store_raw(event)
            
            # Process event into memory blocks
            # Get recent events for context (last 10 processed events)
            context_events = self.processed_memories[-10:] if self.processed_memories else []
            
            # Add to memory block (handles linking automatically)
            memory_block = self.block_manager.process_event(event, context_events)
            
            # Check if we should store in processed memory
            # Use block-level salience from the salience matrix
            if not skip_salience_check and memory_block.block_salience < self.config.memory.salience_threshold:
                return True, f"Stored in raw memory only (block salience {memory_block.block_salience:.2f} below threshold)"
            
            # Embed event
            key, value = self.embedder.embed_event(event)
            
            # Get event-specific salience from block's salience matrix
            event_salience = memory_block.get_event_salience(event.id)
            
            # Store in Hopfield network
            metadata = {
                'event_id': event.id,
                'episode_id': event.episode_id,
                'event_salience': event_salience,  # From salience matrix
                'block_id': memory_block.id,
                'block_salience': memory_block.block_salience,
                'priority_score': event.priority_score,
                'timestamp': event.timestamp
            }
            
            hopfield_idx = self.hopfield.store(
                key=key,
                value=value,
                metadata=metadata,
                salience=memory_block.block_salience  # Use block salience for Hopfield storage
            )
            
            # Store in processed collection
            self._store_processed(event, key)
            
            # Track the memory block if new
            if memory_block not in self.processed_blocks:
                self.processed_blocks.append(memory_block)
            
            # Update episode map
            if event.episode_id not in self.episode_map:
                self.episode_map[event.episode_id] = []
            self.episode_map[event.episode_id].append(event.id)
            
            # Auto-save if configured
            self.operation_count += 1
            if self.config.storage.auto_save and self.operation_count % self.config.storage.save_interval == 0:
                self.save()
            
            return True, f"Stored event in block {memory_block.id[:8]} (block salience: {memory_block.block_salience:.2f}, event salience: {event_salience:.2f})"
            
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            return False, str(e)
    
    def retrieve(
        self,
        query: Dict[str, str],
        k: int = 5,
        include_raw: bool = False,
        return_blocks: bool = False
    ) -> List[Tuple[Event, float]]:
        """
        Retrieve memories matching a partial 5W1H query using memory blocks
        
        Args:
            query: Partial 5W1H query
            k: Number of results
            include_raw: Include raw memories in search
            return_blocks: Return entire memory blocks instead of individual events
        
        Returns:
            List of (event, similarity_score) tuples, or (block, score) if return_blocks=True
        """
        # First, find relevant memory blocks
        relevant_blocks = self.block_manager.retrieve_relevant_blocks(query, k=k*2)
        
        if return_blocks:
            # Return the blocks themselves
            for block, score in relevant_blocks[:k]:
                block.update_access()
            return relevant_blocks[:k]
        
        # Extract events from relevant blocks with block context
        results = []
        seen_events = set()
        
        for block, block_score in relevant_blocks:
            # Update block access
            block.update_access()
            
            # Score each event in the block based on query
            for event in block.events:
                if event.id in seen_events:
                    continue
                
                # Compute event-specific score
                event_score = 0.0
                
                # Check how well event matches query
                for key, value in query.items():
                    if hasattr(event.five_w1h, key) and value:
                        event_value = getattr(event.five_w1h, key)
                        if event_value:
                            # Simple text matching
                            if value.lower() in event_value.lower():
                                event_score += 0.2
                            elif any(word in event_value.lower() for word in value.lower().split()):
                                event_score += 0.1
                
                # Combine block score with event score
                # Events in highly relevant blocks get a boost
                combined_score = block_score * 0.6 + event_score * 0.4
                
                # Additional boost based on event's salience within its block
                event_salience = block.get_event_salience(event.id)
                combined_score = combined_score * 0.8 + event_salience * 0.2
                
                results.append((event, combined_score))
                seen_events.add(event.id)
        
        # Also search in Hopfield network for individual high-salience memories
        query_embedding = self.embedder.embed_partial_query(query)
        hopfield_results = self.hopfield.retrieve(query_embedding, k=k)
        
        for value, metadata, score in hopfield_results:
            if metadata and 'event_id' in metadata:
                event_id = metadata['event_id']
                if event_id not in seen_events:
                    event = self._find_event_by_id(event_id)
                    if event:
                        event.update_access()
                        # Check if event is in a block
                        blocks = self.block_manager.get_blocks_for_event(event_id)
                        if blocks:
                            # Boost score if in a relevant block
                            score = score * 1.2
                        results.append((event, score))
                        seen_events.add(event_id)
        
        # Search in ChromaDB if available and requested
        if include_raw and self.use_chromadb:
            collection = self.raw_collection if include_raw else self.processed_collection
            
            if collection:
                try:
                    query_text = " ".join(f"{k}:{v}" for k, v in query.items() if v)
                    
                    chroma_results = collection.query(
                        query_texts=[query_text],
                        n_results=k
                    )
                    
                    if chroma_results['ids'] and chroma_results['ids'][0]:
                        for idx, event_id in enumerate(chroma_results['ids'][0]):
                            if event_id not in seen_events:
                                event = self._find_event_by_id(event_id)
                                if event:
                                    distance = chroma_results['distances'][0][idx] if chroma_results['distances'] else 0
                                    similarity = 1.0 / (1.0 + distance)
                                    event.update_access()
                                    results.append((event, similarity))
                                    seen_events.add(event_id)
                
                except Exception as e:
                    logger.warning(f"ChromaDB search failed: {e}")
        
        # Sort by score and limit to k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def retrieve_episode(self, episode_id: str) -> List[Event]:
        """Retrieve all events in an episode"""
        if episode_id not in self.episode_map:
            return []
        
        events = []
        for event_id in self.episode_map[episode_id]:
            event = self._find_event_by_id(event_id)
            if event:
                events.append(event)
        
        # Sort by timestamp
        events.sort(key=lambda e: e.created_at)
        return events
    
    def get_causal_chain(
        self,
        event_id: str,
        max_depth: int = 5
    ) -> List[Tuple[Event, Event]]:
        """
        Get causal chain (action -> observation pairs) for an event
        
        Returns:
            List of (action, observation) tuples
        """
        event = self._find_event_by_id(event_id)
        if not event:
            return []
        
        episode_events = self.retrieve_episode(event.episode_id)
        
        # Pair actions with observations
        pairs = []
        for i in range(len(episode_events) - 1):
            if (episode_events[i].event_type == EventType.ACTION and
                episode_events[i+1].event_type == EventType.OBSERVATION):
                pairs.append((episode_events[i], episode_events[i+1]))
                if len(pairs) >= max_depth:
                    break
        
        return pairs
    
    def _store_raw(self, event: Event):
        """Store event in raw memory"""
        self.raw_memories.append(event)
        
        if self.use_chromadb and self.raw_collection:
            try:
                # Prepare document for ChromaDB
                document = json.dumps(event.five_w1h.to_dict())
                metadata = {
                    "event_type": event.event_type.value,
                    "episode_id": event.episode_id,
                    "salience": 0.5,  # Now in block salience matrix
                    "timestamp": event.timestamp
                }
                
                # Add tags as metadata
                for i, tag in enumerate(event.tags[:5]):  # Limit tags
                    metadata[f"tag_{i}"] = tag
                
                self.raw_collection.add(
                    ids=[event.id],
                    documents=[document],
                    metadatas=[metadata]
                )
            except Exception as e:
                logger.warning(f"Failed to store in ChromaDB raw collection: {e}")
    
    def _store_processed(self, event: Event, embedding: np.ndarray):
        """Store event in processed memory"""
        self.processed_memories.append(event)
        
        if self.use_chromadb and self.processed_collection:
            try:
                document = json.dumps(event.five_w1h.to_dict())
                metadata = {
                    "event_type": event.event_type.value,
                    "episode_id": event.episode_id,
                    "salience": 0.5,  # Now in block salience matrix
                    "timestamp": event.timestamp
                }
                
                self.processed_collection.add(
                    ids=[event.id],
                    documents=[document],
                    embeddings=[embedding.tolist()],
                    metadatas=[metadata]
                )
            except Exception as e:
                logger.warning(f"Failed to store in ChromaDB processed collection: {e}")
    
    def _find_event_by_id(self, event_id: str) -> Optional[Event]:
        """Find event by ID in memory"""
        # Search processed first (more likely to be accessed)
        for event in self.processed_memories:
            if event.id == event_id:
                return event
        
        # Then search raw
        for event in self.raw_memories:
            if event.id == event_id:
                return event
        
        return None
    
    def save(self):
        """Save memory state to disk"""
        try:
            # Save raw memories
            raw_data = [event.to_dict() for event in self.raw_memories]
            with open(self.config.storage.raw_memory_path, 'w') as f:
                json.dump(raw_data, f, indent=2, default=str)
            
            # Save processed memories
            processed_data = [event.to_dict() for event in self.processed_memories]
            with open(self.config.storage.processed_memory_path, 'w') as f:
                json.dump(processed_data, f, indent=2, default=str)
            
            # Save Hopfield state
            self.hopfield.save_state(self.config.storage.hopfield_state_path)
            
            # Save memory blocks
            blocks_path = self.config.storage.state_dir / "memory_blocks.json"
            self.block_manager.save_state(str(blocks_path))
            
            # Save episode map
            episode_map_path = self.config.storage.state_dir / "episode_map.json"
            with open(episode_map_path, 'w') as f:
                json.dump(self.episode_map, f, indent=2)
            
            # Save embedding cache
            cache_path = self.config.storage.state_dir / "embedding_cache.json"
            self.text_embedder.save_cache(cache_path)
            
            self.last_save = datetime.utcnow()
            logger.info(f"Saved memory state: {len(self.raw_memories)} raw, {len(self.processed_memories)} processed, {len(self.block_manager.blocks)} blocks")
            
        except Exception as e:
            logger.error(f"Failed to save memory state: {e}")
    
    def _load_from_disk(self):
        """Load memory state from disk"""
        try:
            # Load raw memories
            if self.config.storage.raw_memory_path.exists():
                with open(self.config.storage.raw_memory_path, 'r') as f:
                    raw_data = json.load(f)
                self.raw_memories = [Event.from_dict(d) for d in raw_data]
                logger.info(f"Loaded {len(self.raw_memories)} raw memories")
            
            # Load processed memories
            if self.config.storage.processed_memory_path.exists():
                with open(self.config.storage.processed_memory_path, 'r') as f:
                    processed_data = json.load(f)
                self.processed_memories = [Event.from_dict(d) for d in processed_data]
                logger.info(f"Loaded {len(self.processed_memories)} processed memories")
            
            # Load Hopfield state
            self.hopfield.load_state(self.config.storage.hopfield_state_path)
            
            # Load memory blocks
            blocks_path = self.config.storage.state_dir / "memory_blocks.json"
            all_events = self.raw_memories + self.processed_memories
            self.block_manager.load_state(str(blocks_path), all_events)
            
            # Load episode map
            episode_map_path = self.config.storage.state_dir / "episode_map.json"
            if episode_map_path.exists():
                with open(episode_map_path, 'r') as f:
                    self.episode_map = json.load(f)
            
            # Load embedding cache
            cache_path = self.config.storage.state_dir / "embedding_cache.json"
            self.text_embedder.load_cache(cache_path)
            
        except Exception as e:
            logger.warning(f"Failed to load memory state: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = {
            'raw_count': len(self.raw_memories),
            'processed_count': len(self.processed_memories),
            'episode_count': len(self.episode_map),
            'total_events': len(self.raw_memories) + len(self.processed_memories),
            'hopfield': self.hopfield.get_statistics(),
            'blocks': self.block_manager.get_statistics(),  # Add block statistics
            'last_save': self.last_save.isoformat(),
            'operations_since_save': self.operation_count % self.config.storage.save_interval
        }
        
        # Calculate average salience
        if self.processed_memories:
            avg_salience = np.mean([b.block_salience for b in self.processed_blocks]) if self.processed_blocks else 0.5
            stats['avg_salience'] = float(avg_salience)
        
        # Calculate average block salience
        if self.block_manager.blocks:
            block_saliences = [b.salience for b in self.block_manager.blocks.values()]
            stats['avg_block_salience'] = float(np.mean(block_saliences))
            
            # Get block coherence distribution
            coherences = [b.coherence_score for b in self.block_manager.blocks.values()]
            stats['avg_block_coherence'] = float(np.mean(coherences)) if coherences else 0.0
        
        # Memory usage by type
        type_counts = {}
        for event in self.raw_memories:
            type_counts[event.event_type.value] = type_counts.get(event.event_type.value, 0) + 1
        stats['event_types'] = type_counts
        
        return stats
    
    def clear(self):
        """Clear all memories"""
        self.raw_memories.clear()
        self.processed_memories.clear()
        self.episode_map.clear()
        self.hopfield.clear()
        
        if self.use_chromadb:
            try:
                # Clear ChromaDB collections
                if self.raw_collection:
                    self.chroma_client.delete_collection(self.config.storage.raw_collection_name)
                if self.processed_collection:
                    self.chroma_client.delete_collection(self.config.storage.processed_collection_name)
                
                # Recreate collections
                self._initialize_storage()
            except Exception as e:
                logger.warning(f"Failed to clear ChromaDB collections: {e}")
        
        logger.info("Cleared all memories")
    
    def __len__(self) -> int:
        """Total number of memories"""
        return len(self.raw_memories) + len(self.processed_memories)
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"MemoryStore(raw={len(self.raw_memories)}, "
            f"processed={len(self.processed_memories)}, "
            f"episodes={len(self.episode_map)})"
        )
