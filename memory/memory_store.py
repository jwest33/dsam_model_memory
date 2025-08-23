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
from embedding.embedder import FiveW1HEmbedder, TextEmbedder
from memory.hopfield import ModernHopfieldNetwork

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
        
        # Storage backends
        self.use_chromadb = self.config.storage.use_chromadb
        self.raw_collection = None
        self.processed_collection = None
        
        # Memory tracking
        self.raw_memories: List[Event] = []
        self.processed_memories: List[Event] = []
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
        Store an event in memory
        
        Args:
            event: Event to store
            skip_salience_check: Store regardless of salience threshold
        
        Returns:
            (success, message)
        """
        try:
            # Always store in raw memory
            self._store_raw(event)
            
            # Check salience threshold for processed storage
            if not skip_salience_check and event.salience < self.config.memory.salience_threshold:
                return True, f"Stored in raw memory only (salience {event.salience:.2f} below threshold)"
            
            # Embed event
            key, value = self.embedder.embed_event(event)
            
            # Store in Hopfield network
            metadata = {
                'event_id': event.id,
                'episode_id': event.episode_id,
                'salience': event.salience,
                'priority_score': event.priority_score,
                'timestamp': event.timestamp
            }
            
            hopfield_idx = self.hopfield.store(
                key=key,
                value=value,
                metadata=metadata,
                salience=event.salience
            )
            
            # Store in processed collection
            self._store_processed(event, key)
            
            # Update episode map
            if event.episode_id not in self.episode_map:
                self.episode_map[event.episode_id] = []
            self.episode_map[event.episode_id].append(event.id)
            
            # Auto-save if configured
            self.operation_count += 1
            if self.config.storage.auto_save and self.operation_count % self.config.storage.save_interval == 0:
                self.save()
            
            return True, f"Stored event (salience: {event.salience:.2f}, hopfield_idx: {hopfield_idx})"
            
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            return False, str(e)
    
    def retrieve(
        self,
        query: Dict[str, str],
        k: int = 5,
        include_raw: bool = False
    ) -> List[Tuple[Event, float]]:
        """
        Retrieve memories matching a partial 5W1H query
        
        Args:
            query: Partial 5W1H query
            k: Number of results
            include_raw: Include raw memories in search
        
        Returns:
            List of (event, similarity_score) tuples
        """
        # Embed query
        query_embedding = self.embedder.embed_partial_query(query)
        
        results = []
        
        # Search in Hopfield network
        hopfield_results = self.hopfield.retrieve(query_embedding, k=k)
        
        for value, metadata, score in hopfield_results:
            if metadata and 'event_id' in metadata:
                # Find corresponding event
                event = self._find_event_by_id(metadata['event_id'])
                if event:
                    event.update_access()
                    results.append((event, score))
        
        # Search in ChromaDB if available
        if self.use_chromadb:
            collection = self.raw_collection if include_raw else self.processed_collection
            
            if collection:
                try:
                    # Convert query to text for ChromaDB
                    query_text = " ".join(f"{k}:{v}" for k, v in query.items() if v)
                    
                    chroma_results = collection.query(
                        query_texts=[query_text],
                        n_results=k
                    )
                    
                    if chroma_results['ids'] and chroma_results['ids'][0]:
                        for idx, event_id in enumerate(chroma_results['ids'][0]):
                            event = self._find_event_by_id(event_id)
                            if event:
                                # Calculate similarity from distance
                                distance = chroma_results['distances'][0][idx] if chroma_results['distances'] else 0
                                similarity = 1.0 / (1.0 + distance)
                                
                                # Avoid duplicates
                                if not any(e.id == event.id for e, _ in results):
                                    event.update_access()
                                    results.append((event, similarity))
                
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
                    "salience": event.salience,
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
                    "salience": event.salience,
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
            
            # Save episode map
            episode_map_path = self.config.storage.state_dir / "episode_map.json"
            with open(episode_map_path, 'w') as f:
                json.dump(self.episode_map, f, indent=2)
            
            # Save embedding cache
            cache_path = self.config.storage.state_dir / "embedding_cache.json"
            self.text_embedder.save_cache(cache_path)
            
            self.last_save = datetime.utcnow()
            logger.info(f"Saved memory state: {len(self.raw_memories)} raw, {len(self.processed_memories)} processed")
            
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
            'last_save': self.last_save.isoformat(),
            'operations_since_save': self.operation_count % self.config.storage.save_interval
        }
        
        # Calculate average salience
        if self.processed_memories:
            avg_salience = np.mean([e.salience for e in self.processed_memories])
            stats['avg_salience'] = float(avg_salience)
        
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