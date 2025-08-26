"""
ChromaDB-based memory storage system

This module implements efficient database storage for the memory system,
replacing JSON files with ChromaDB for better performance and scalability.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import pickle
import base64

from models.event import Event, FiveW1H, EventType

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    
from models.event import Event, FiveW1H
from models.memory_block import MemoryBlock
from config import get_config

logger = logging.getLogger(__name__)

class ChromaDBStore:
    """
    Efficient database storage using ChromaDB for memories and blocks.
    
    Collections:
    1. events: Individual events with embeddings
    2. blocks: Memory blocks with aggregate embeddings and salience matrices
    3. metadata: System metadata (episodes, statistics, etc.)
    """
    
    def __init__(self, config=None):
        """Initialize ChromaDB storage"""
        self.config = config or get_config()
        
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is required for database storage. Install with: pip install chromadb")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.config.storage.chromadb_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize collections
        self._init_collections()
        
        # Cache for frequently accessed data
        self.cache = {
            'blocks': {},  # block_id -> MemoryBlock
            'events': {},  # event_id -> Event
            'episode_map': {}  # episode_id -> [event_ids]
        }
        
        # Load cache
        self._refresh_cache()
    
    def _init_collections(self):
        """Initialize or get ChromaDB collections"""
        # Events collection
        self.events_collection = self.client.get_or_create_collection(
            name="events",
            metadata={"description": "Individual memory events"},
            embedding_function=None  # We'll provide embeddings
        )
        
        # Blocks collection
        self.blocks_collection = self.client.get_or_create_collection(
            name="memory_blocks",
            metadata={"description": "Memory blocks with salience matrices"},
            embedding_function=None
        )
        
        # Metadata collection (for system state)
        self.metadata_collection = self.client.get_or_create_collection(
            name="metadata",
            metadata={"description": "System metadata and indices"},
            embedding_function=None
        )
    
    def store_event(self, event: Event, embedding: np.ndarray, block_id: Optional[str] = None) -> bool:
        """
        Store an event in the database
        
        Args:
            event: Event to store
            embedding: Event embedding vector
            block_id: ID of the containing memory block
        
        Returns:
            Success status
        """
        try:
            # Prepare metadata - include ALL 5W1H fields
            metadata = {
                "event_type": event.event_type.value,
                "episode_id": event.episode_id,
                "timestamp": event.timestamp,
                "created_at": event.created_at.isoformat(),
                "who": event.five_w1h.who or "",
                "what": event.five_w1h.what or "",
                "when": event.five_w1h.when or "",
                "where": event.five_w1h.where or "",
                "why": event.five_w1h.why or "",
                "how": event.five_w1h.how or "",
                "confidence": float(event.confidence)
            }
            
            # Add block_id only if provided (for dynamic clustering)
            if block_id:
                metadata["block_id"] = block_id
            
            # Prepare document (5W1H as JSON)
            document = json.dumps({
                "who": event.five_w1h.who,
                "what": event.five_w1h.what,
                "when": event.five_w1h.when,
                "where": event.five_w1h.where,
                "why": event.five_w1h.why,
                "how": event.five_w1h.how,
                "full_content": event.full_content
            })
            
            # Store in ChromaDB
            self.events_collection.upsert(
                ids=[event.id],
                embeddings=[embedding.tolist()],
                documents=[document],
                metadatas=[metadata]
            )
            
            # Update cache
            self.cache['events'][event.id] = event
            
            # Update episode map
            if event.episode_id not in self.cache['episode_map']:
                self.cache['episode_map'][event.episode_id] = []
            if event.id not in self.cache['episode_map'][event.episode_id]:
                self.cache['episode_map'][event.episode_id].append(event.id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            return False
    
    def store_block(self, block: MemoryBlock) -> bool:
        """
        Store a memory block with its salience matrix
        
        Args:
            block: MemoryBlock to store
        
        Returns:
            Success status
        """
        try:
            # Serialize salience matrix and event embeddings
            matrix_data = {
                "salience_matrix": self._encode_array(block.salience_matrix) if block.salience_matrix is not None else None,
                "event_saliences": self._encode_array(block.event_saliences) if block.event_saliences is not None else None,
                "event_embeddings": [self._encode_array(emb) for emb in block.event_embeddings] if block.event_embeddings else []
            }
            
            # Prepare metadata
            metadata = {
                "block_type": block.block_type,
                "block_salience": float(block.block_salience),
                "coherence_score": float(block.coherence_score),
                "event_count": len(block.events),
                "link_count": len(block.links),
                "created_at": block.created_at.isoformat(),
                "updated_at": block.updated_at.isoformat(),
                "embedding_version": block.embedding_version
            }
            
            # Prepare document (includes matrix data and links)
            document = json.dumps({
                "event_ids": list(block.event_ids),
                "matrix_data": matrix_data,
                "links": [
                    {
                        "source_id": link.source_id,
                        "target_id": link.target_id,
                        "link_type": link.link_type.value,
                        "strength": link.strength
                    }
                    for link in block.links
                ],
                "aggregate_signature": block.aggregate_signature.to_dict() if block.aggregate_signature else None,
                "tags": block.tags
            })
            
            # Use block embedding for retrieval, or create a placeholder
            if block.block_embedding is not None:
                embedding = block.block_embedding.tolist()
            else:
                # Create a simple average embedding from events
                embedding = np.zeros(self.config.memory.embedding_dim).tolist()
            
            # Store in ChromaDB
            self.blocks_collection.upsert(
                ids=[block.id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata]
            )
            
            # Update cache
            self.cache['blocks'][block.id] = block
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store block: {e}")
            return False
    
    def retrieve_events_by_query(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_conditions: Optional[Dict] = None
    ) -> List[Tuple[Event, float]]:
        """
        Retrieve events by embedding similarity
        
        Args:
            query_embedding: Query vector
            k: Number of results
            filter_conditions: Optional metadata filters
        
        Returns:
            List of (event, similarity_score) tuples
        """
        try:
            # Build where clause
            where = filter_conditions if filter_conditions else None
            
            # Query ChromaDB
            results = self.events_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                where=where
            )
            
            # Convert results to events
            events_with_scores = []
            if results['ids'] and results['ids'][0]:
                for i, event_id in enumerate(results['ids'][0]):
                    # Try cache first
                    if event_id in self.cache['events']:
                        event = self.cache['events'][event_id]
                    else:
                        # Reconstruct from database
                        event = self._reconstruct_event(
                            event_id,
                            json.loads(results['documents'][0][i]),
                            results['metadatas'][0][i]
                        )
                        self.cache['events'][event_id] = event
                    
                    # Calculate similarity (ChromaDB returns distances)
                    distance = results['distances'][0][i] if results['distances'] else 0
                    similarity = 1.0 / (1.0 + distance)
                    
                    events_with_scores.append((event, similarity))
            
            return events_with_scores
            
        except Exception as e:
            logger.error(f"Failed to retrieve events: {e}")
            return []
    
    def retrieve_blocks_by_query(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[MemoryBlock, float]]:
        """
        Retrieve memory blocks by embedding similarity
        
        Args:
            query_embedding: Query vector
            k: Number of results
        
        Returns:
            List of (block, similarity_score) tuples
        """
        try:
            # Query ChromaDB
            results = self.blocks_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k
            )
            
            # Convert results to blocks
            blocks_with_scores = []
            if results['ids'] and results['ids'][0]:
                for i, block_id in enumerate(results['ids'][0]):
                    # Try cache first
                    if block_id in self.cache['blocks']:
                        block = self.cache['blocks'][block_id]
                    else:
                        # Reconstruct from database
                        block = self._reconstruct_block(
                            block_id,
                            json.loads(results['documents'][0][i]),
                            results['metadatas'][0][i]
                        )
                        self.cache['blocks'][block_id] = block
                    
                    # Calculate similarity
                    distance = results['distances'][0][i] if results['distances'] else 0
                    similarity = 1.0 / (1.0 + distance)
                    
                    blocks_with_scores.append((block, similarity))
            
            return blocks_with_scores
            
        except Exception as e:
            logger.error(f"Failed to retrieve blocks: {e}")
            return []
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """Get a specific event by ID"""
        # Check cache first
        if event_id in self.cache['events']:
            return self.cache['events'][event_id]
        
        try:
            result = self.events_collection.get(
                ids=[event_id],
                include=['documents', 'metadatas']
            )
            
            if result['ids']:
                event = self._reconstruct_event(
                    event_id,
                    json.loads(result['documents'][0]),
                    result['metadatas'][0]
                )
                self.cache['events'][event_id] = event
                return event
                
        except Exception as e:
            logger.error(f"Failed to get event {event_id}: {e}")
        
        return None
    
    def get_block(self, block_id: str) -> Optional[MemoryBlock]:
        """Get a specific block by ID"""
        # Check cache first
        if block_id in self.cache['blocks']:
            return self.cache['blocks'][block_id]
        
        try:
            result = self.blocks_collection.get(
                ids=[block_id],
                include=['documents', 'metadatas', 'embeddings']
            )
            
            if result['ids']:
                block = self._reconstruct_block(
                    block_id,
                    json.loads(result['documents'][0]),
                    result['metadatas'][0],
                    np.array(result['embeddings'][0]) if result['embeddings'] else None
                )
                self.cache['blocks'][block_id] = block
                return block
                
        except Exception as e:
            logger.error(f"Failed to get block {block_id}: {e}")
        
        return None
    
    def get_episode_events(self, episode_id: str) -> List[Event]:
        """Get all events in an episode"""
        try:
            # Query events by episode_id
            results = self.events_collection.get(
                where={"episode_id": episode_id},
                include=['ids', 'documents', 'metadatas']
            )
            
            events = []
            if results['ids']:
                for i, event_id in enumerate(results['ids']):
                    if event_id in self.cache['events']:
                        events.append(self.cache['events'][event_id])
                    else:
                        event = self._reconstruct_event(
                            event_id,
                            json.loads(results['documents'][i]),
                            results['metadatas'][i]
                        )
                        self.cache['events'][event_id] = event
                        events.append(event)
            
            # Sort by timestamp
            events.sort(key=lambda e: e.created_at)
            return events
            
        except Exception as e:
            logger.error(f"Failed to get episode events: {e}")
            return []
    
    def update_block_salience(self, block_id: str, salience_matrix: np.ndarray, event_saliences: np.ndarray):
        """Update a block's salience matrix efficiently"""
        try:
            # Get existing block
            block = self.get_block(block_id)
            if not block:
                return False
            
            # Update matrices
            block.salience_matrix = salience_matrix
            block.event_saliences = event_saliences
            block.block_salience = float(np.mean(event_saliences))
            
            # Store updated block
            return self.store_block(block)
            
        except Exception as e:
            logger.error(f"Failed to update block salience: {e}")
            return False
    
    def delete_event(self, event_id: str) -> bool:
        """Delete an event from the database"""
        try:
            self.events_collection.delete(ids=[event_id])
            if event_id in self.cache['events']:
                del self.cache['events'][event_id]
            return True
        except Exception as e:
            logger.error(f"Failed to delete event: {e}")
            return False
    
    def delete_block(self, block_id: str) -> bool:
        """Delete a block from the database"""
        try:
            self.blocks_collection.delete(ids=[block_id])
            if block_id in self.cache['blocks']:
                del self.cache['blocks'][block_id]
            return True
        except Exception as e:
            logger.error(f"Failed to delete block: {e}")
            return False
    
    def get_embedding(self, event_id: str) -> Optional[np.ndarray]:
        """
        Get embedding for a specific event
        
        Args:
            event_id: Event ID
            
        Returns:
            Embedding vector or None if not found
        """
        try:
            result = self.events_collection.get(
                ids=[event_id],
                include=['embeddings']
            )
            
            if result['ids'] and len(result['embeddings']) > 0 and len(result['embeddings'][0]) > 0:
                return np.array(result['embeddings'][0])
            return None
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None
    
    def update_embedding(self, event_id: str, embedding: np.ndarray) -> bool:
        """
        Update embedding for an event
        
        Args:
            event_id: Event ID
            embedding: New embedding vector
            
        Returns:
            Success status
        """
        try:
            self.events_collection.update(
                ids=[event_id],
                embeddings=[embedding.tolist()]
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to update embedding: {e}")
            return False
    
    def update_event(self, event: Event) -> bool:
        """
        Update an event in the database
        
        Args:
            event: Event to update
            
        Returns:
            Success status
        """
        try:
            # Update metadata
            metadata = {
                "event_type": event.event_type.value,
                "episode_id": event.episode_id,
                "timestamp": event.timestamp,
                "created_at": event.created_at.isoformat(),
                "who": event.five_w1h.who or "",
                "where": event.five_w1h.where or "",
                "confidence": float(event.confidence),
                "accessed_count": event.accessed_count,
                "last_accessed": event.last_accessed.isoformat() if event.last_accessed else None
            }
            
            # Update document
            document = json.dumps({
                "who": event.five_w1h.who,
                "what": event.five_w1h.what,
                "when": event.five_w1h.when,
                "where": event.five_w1h.where,
                "why": event.five_w1h.why,
                "how": event.five_w1h.how,
                "full_content": event.full_content,
                "tags": event.tags,
                "metadata": event.metadata
            })
            
            # Get existing embedding to include in update
            existing_embedding = self.get_embedding(event.id)
            if existing_embedding is not None:
                self.events_collection.update(
                    ids=[event.id],
                    documents=[document],
                    metadatas=[metadata],
                    embeddings=[existing_embedding.tolist()]
                )
            else:
                # If no existing embedding, just update metadata
                # Note: This shouldn't happen in normal flow
                logger.warning(f"No embedding found for event {event.id}, skipping update")
            
            # Update cache
            self.cache['events'][event.id] = event
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update event: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Alias for get_statistics for compatibility"""
        return self.get_statistics()
    
    def retrieve_all_events(self) -> List[Event]:
        """
        Retrieve all events from the database
        
        Returns:
            List of all Event objects
        """
        try:
            # Get all events from ChromaDB
            results = self.events_collection.get(
                include=['documents', 'metadatas']
            )
            
            events = []
            for i, doc in enumerate(results['documents']):
                # Parse the JSON document
                data = json.loads(doc)
                metadata = results['metadatas'][i]
                
                # Reconstruct Event object
                event = Event(
                    id=results['ids'][i],
                    five_w1h=FiveW1H(
                        who=data.get('who', ''),
                        what=data.get('what', ''),
                        when=data.get('when', ''),
                        where=data.get('where', ''),
                        why=data.get('why', ''),
                        how=data.get('how', '')
                    ),
                    event_type=EventType(metadata.get('event_type', 'observation')),
                    episode_id=metadata.get('episode_id', ''),
                    confidence=metadata.get('confidence', 1.0),
                    full_content=data.get('full_content', '')
                )
                
                # Timestamp is a computed property, don't try to set it
                
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to retrieve all events: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            # ChromaDB doesn't have a count() method, use get() with limit
            events_result = self.events_collection.get(limit=1)
            blocks_result = self.blocks_collection.get(limit=1)
            
            # Get actual counts by retrieving all IDs (lightweight operation)
            all_events = self.events_collection.get(include=[])  # Only get IDs
            all_blocks = self.blocks_collection.get(include=[])
            
            return {
                "total_events": len(all_events['ids']),
                "total_blocks": len(all_blocks['ids']),
                "episodes": len(self.cache['episode_map']),
                "cache_size": {
                    "events": len(self.cache['events']),
                    "blocks": len(self.cache['blocks'])
                }
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def export_to_json(self, filepath: str):
        """Export database to JSON for backup"""
        try:
            # Get all data
            events = self.events_collection.get(include=['documents', 'metadatas', 'embeddings'])
            blocks = self.blocks_collection.get(include=['documents', 'metadatas', 'embeddings'])
            
            # Prepare export data
            export_data = {
                "events": {
                    "ids": events['ids'],
                    "documents": events['documents'],
                    "metadatas": events['metadatas'],
                    "embeddings": events['embeddings']
                },
                "blocks": {
                    "ids": blocks['ids'],
                    "documents": blocks['documents'],
                    "metadatas": blocks['metadatas'],
                    "embeddings": blocks['embeddings']
                },
                "metadata": {
                    "exported_at": datetime.utcnow().isoformat(),
                    "statistics": self.get_statistics()
                }
            }
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported database to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export database: {e}")
            return False
    
    def import_from_json(self, filepath: str):
        """Import database from JSON backup"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Clear existing collections
            self.clear_all()
            
            # Import events
            if 'events' in data and data['events']['ids']:
                self.events_collection.add(
                    ids=data['events']['ids'],
                    documents=data['events']['documents'],
                    metadatas=data['events']['metadatas'],
                    embeddings=data['events']['embeddings']
                )
            
            # Import blocks
            if 'blocks' in data and data['blocks']['ids']:
                self.blocks_collection.add(
                    ids=data['blocks']['ids'],
                    documents=data['blocks']['documents'],
                    metadatas=data['blocks']['metadatas'],
                    embeddings=data['blocks']['embeddings']
                )
            
            # Refresh cache
            self._refresh_cache()
            
            logger.info(f"Imported database from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import database: {e}")
            return False
    
    def clear_all(self):
        """Clear all collections (use with caution!)"""
        try:
            self.client.delete_collection("events")
            self.client.delete_collection("memory_blocks")
            self.client.delete_collection("metadata")
            self._init_collections()
            self.cache = {'blocks': {}, 'events': {}, 'episode_map': {}}
            logger.info("Cleared all collections")
        except Exception as e:
            logger.error(f"Failed to clear collections: {e}")
    
    def _encode_array(self, arr: np.ndarray) -> str:
        """Encode numpy array as base64 string for storage"""
        if arr is None:
            return None
        return base64.b64encode(pickle.dumps(arr)).decode('utf-8')
    
    def _decode_array(self, encoded: str) -> np.ndarray:
        """Decode base64 string back to numpy array"""
        if encoded is None:
            return None
        return pickle.loads(base64.b64decode(encoded))
    
    def _reconstruct_event(self, event_id: str, document: Dict, metadata: Dict) -> Event:
        """Reconstruct an Event object from database data"""
        from models.event import EventType
        
        five_w1h = FiveW1H(
            who=document.get('who'),
            what=document.get('what'),
            when=document.get('when'),
            where=document.get('where'),
            why=document.get('why'),
            how=document.get('how')
        )
        
        event = Event(
            five_w1h=five_w1h,
            id=event_id,
            event_type=EventType(metadata['event_type']),
            episode_id=metadata['episode_id'],
            confidence=metadata.get('confidence', 1.0),
            full_content=document.get('full_content')
        )
        
        # Set timestamps
        if 'created_at' in metadata:
            event.created_at = datetime.fromisoformat(metadata['created_at'])
        
        return event
    
    def _reconstruct_block(self, block_id: str, document: Dict, metadata: Dict, embedding: Optional[np.ndarray] = None) -> MemoryBlock:
        """Reconstruct a MemoryBlock object from database data"""
        from models.memory_block import LinkType, MemoryLink
        
        block = MemoryBlock(id=block_id)
        
        # Set basic metadata
        block.block_type = metadata['block_type']
        block.block_salience = metadata['block_salience']
        block.coherence_score = metadata['coherence_score']
        block.embedding_version = metadata.get('embedding_version', 0)
        
        # Set timestamps
        if 'created_at' in metadata:
            block.created_at = datetime.fromisoformat(metadata['created_at'])
        if 'updated_at' in metadata:
            block.updated_at = datetime.fromisoformat(metadata['updated_at'])
        
        # Reconstruct event list
        event_ids = document.get('event_ids', [])
        block.event_ids = set(event_ids)
        
        # Get actual events
        block.events = []
        for event_id in event_ids:
            event = self.get_event(event_id)
            if event:
                block.events.append(event)
        
        # Reconstruct links
        block.links = []
        for link_data in document.get('links', []):
            link = MemoryLink(
                source_id=link_data['source_id'],
                target_id=link_data['target_id'],
                link_type=LinkType(link_data['link_type']),
                strength=link_data['strength']
            )
            block.links.append(link)
        
        # Reconstruct matrices
        matrix_data = document.get('matrix_data', {})
        if matrix_data:
            block.salience_matrix = self._decode_array(matrix_data.get('salience_matrix'))
            block.event_saliences = self._decode_array(matrix_data.get('event_saliences'))
            if matrix_data.get('event_embeddings'):
                block.event_embeddings = [self._decode_array(emb) for emb in matrix_data['event_embeddings']]
        
        # Set block embedding
        if embedding is not None:
            block.block_embedding = embedding
        
        # Reconstruct aggregate signature
        if document.get('aggregate_signature'):
            sig_data = document['aggregate_signature']
            block.aggregate_signature = FiveW1H(**sig_data)
        
        block.tags = document.get('tags', [])
        
        return block
    
    def _refresh_cache(self):
        """Refresh the cache from database"""
        try:
            # Get sample of recent blocks for cache
            blocks_result = self.blocks_collection.get(
                limit=100,
                include=['documents', 'metadatas']
            )
            
            if blocks_result['ids']:
                for i, block_id in enumerate(blocks_result['ids']):
                    block = self._reconstruct_block(
                        block_id,
                        json.loads(blocks_result['documents'][i]),
                        blocks_result['metadatas'][i]
                    )
                    self.cache['blocks'][block_id] = block
            
            logger.info(f"Cache refreshed with {len(self.cache['blocks'])} blocks")
            
        except Exception as e:
            logger.error(f"Failed to refresh cache: {e}")
    
    def __repr__(self) -> str:
        """String representation"""
        stats = self.get_statistics()
        return f"ChromaDBStore(events={stats.get('total_events', 0)}, blocks={stats.get('total_blocks', 0)})"
