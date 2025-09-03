"""
Qdrant-based memory storage system with multi-vector support

This module implements efficient database storage for the memory system using Qdrant,
which provides native multi-vector support for dual-space (Euclidean + Hyperbolic) embeddings.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
import uuid
from pathlib import Path

from models.event import Event, FiveW1H, EventType
from models.memory_block import MemoryBlock
from models.merged_event import MergedEvent
from config import get_config
from memory.similarity_cache import SimilarityCache

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, Filter, 
        FieldCondition, Range, MatchValue, SearchRequest,
        NamedVector, ScoredPoint, UpdateStatus,
        CollectionStatus, OptimizersConfig, WalConfig, PointIdsList
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    
logger = logging.getLogger(__name__)


class QdrantStore:
    """
    Efficient database storage using Qdrant for memories with dual-space embeddings.
    
    Collections:
    1. events: Individual events with dual embeddings
    2. merged_events: Merged events across different dimensions
    3. memory_blocks: Memory blocks with aggregate embeddings
    """
    
    def __init__(self, config=None):
        """Initialize Qdrant storage"""
        self.config = config or get_config()
        
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant is required for database storage. Install with: pip install qdrant-client")
        
        # Initialize Qdrant client (using local file storage)
        db_path = self.config.storage.qdrant_path
        db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = QdrantClient(path=str(db_path))
        logger.info(f"Initialized Qdrant client with path: {db_path}")
        
        # Initialize collections
        self._init_collections()
        
        # Cache for frequently accessed data
        self.cache = {
            'blocks': {},  # block_id -> MemoryBlock
            'events': {},  # event_id -> Event
            'episode_map': {}  # episode_id -> [event_ids]
        }
        
        # Initialize similarity cache
        self.similarity_cache = SimilarityCache(similarity_threshold=0.2)
        
        # Load cache
        self._refresh_cache()
        
        # Create collection aliases for compatibility
        self._setup_collection_aliases()
    
    def _init_collections(self):
        """Initialize or get Qdrant collections with multi-vector support"""
        
        # Events collection with dual-space vectors
        try:
            self.client.get_collection("events")
            logger.info("Events collection exists")
        except:
            self.client.create_collection(
                collection_name="events",
                vectors_config={
                    "euclidean": VectorParams(
                        size=self.config.memory.embedding_dim,  # 768
                        distance=Distance.COSINE
                    ),
                    "hyperbolic": VectorParams(
                        size=64,  # Hyperbolic dimension
                        distance=Distance.EUCLID  # We'll compute geodesic distance client-side
                    )
                },
                optimizers_config=OptimizersConfig(
                    default_segment_number=2,
                    indexing_threshold=20000,
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,
                    flush_interval_sec=5,
                ),
                wal_config=WalConfig(
                    wal_capacity_mb=32,
                    wal_segments_ahead=2
                )
            )
            logger.info("Created events collection with dual-space vectors")
        
        # Merged events collection (4D merge groups)
        try:
            self.client.get_collection("merged_events")
            logger.info("Merged events collection exists")
        except:
            self.client.create_collection(
                collection_name="merged_events",
                vectors_config={
                    "euclidean": VectorParams(
                        size=self.config.memory.embedding_dim,
                        distance=Distance.COSINE
                    ),
                    "hyperbolic": VectorParams(
                        size=64,
                        distance=Distance.EUCLID
                    )
                }
            )
            logger.info("Created merged events collection")
        
        # Memory blocks collection
        try:
            self.client.get_collection("memory_blocks")
            logger.info("Memory blocks collection exists")
        except:
            self.client.create_collection(
                collection_name="memory_blocks",
                vectors_config={
                    "euclidean": VectorParams(
                        size=self.config.memory.embedding_dim,
                        distance=Distance.COSINE
                    )
                }
            )
            logger.info("Created memory blocks collection")
    
    def store_event(self, event: Event, embedding: np.ndarray, block_id: Optional[str] = None,
                    euclidean_weight: float = 0.5, hyperbolic_weight: float = 0.5,
                    hyperbolic_embedding: Optional[np.ndarray] = None) -> bool:
        """
        Store an event in the database with dual-space embeddings
        
        Args:
            event: Event to store
            embedding: Euclidean embedding vector (768-dim)
            block_id: ID of the containing memory block
            euclidean_weight: Weight for Euclidean space
            hyperbolic_weight: Weight for Hyperbolic space
            hyperbolic_embedding: Optional hyperbolic embedding (64-dim)
        
        Returns:
            Success status
        """
        try:
            # Generate hyperbolic embedding if not provided
            if hyperbolic_embedding is None:
                # Simple projection for now - will be replaced by dual_space_encoder
                hyperbolic_embedding = np.random.randn(64).astype(np.float32)
                hyperbolic_embedding = hyperbolic_embedding / (np.linalg.norm(hyperbolic_embedding) + 1e-8)
                hyperbolic_embedding *= 0.9  # Keep within Poincaré ball
            
            # Prepare payload with all metadata
            payload = {
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
                "full_content": event.full_content or "",
                "confidence": float(event.confidence),
                "euclidean_weight": float(euclidean_weight),
                "hyperbolic_weight": float(hyperbolic_weight),
                "version": 1,
                "last_accessed": event.created_at.isoformat(),
                "access_count": 0,
            }
            
            if block_id:
                payload["block_id"] = block_id
            
            # Store in Qdrant with both vectors
            point = PointStruct(
                id=str(uuid.uuid4()) if not event.id else event.id,
                vector={
                    "euclidean": embedding.tolist(),
                    "hyperbolic": hyperbolic_embedding.tolist()
                },
                payload=payload
            )
            
            self.client.upsert(
                collection_name="events",
                points=[point]
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
    
    def store_merged_event(self, merged_event: MergedEvent, euclidean_embedding: np.ndarray,
                          hyperbolic_embedding: Optional[np.ndarray] = None,
                          merge_type: str = "conceptual", metadata: Optional[Dict] = None) -> bool:
        """
        Store a merged event with dual-space embeddings
        
        Args:
            merged_event: MergedEvent object to store
            euclidean_embedding: Euclidean centroid embedding
            hyperbolic_embedding: Hyperbolic centroid embedding
            merge_type: Type of merge (actor, temporal, conceptual, spatial)
        
        Returns:
            Success status
        """
        try:
            if hyperbolic_embedding is None:
                # Generate default hyperbolic embedding
                hyperbolic_embedding = np.random.randn(64).astype(np.float32)
                hyperbolic_embedding = hyperbolic_embedding / (np.linalg.norm(hyperbolic_embedding) + 1e-8)
                hyperbolic_embedding *= 0.9
            
            # Prepare payload
            payload = {
                'merge_type': merge_type,
                'base_event_id': merged_event.base_event_id,
                'merge_count': merged_event.merge_count,
                'created_at': merged_event.created_at.isoformat(),
                'last_updated': merged_event.last_updated.isoformat(),
                'raw_event_count': len(merged_event.raw_event_ids),
                'raw_event_ids': list(merged_event.raw_event_ids),  # Store as list
                'component_ids': list(merged_event.raw_event_ids),  # Store as list for Qdrant
                'confidence_score': float(getattr(merged_event, 'confidence_score', 1.0)),
                'coherence_score': float(getattr(merged_event, 'coherence_score', 1.0)),
            }
            
            # Add dominant pattern
            if merged_event.dominant_pattern:
                for field in ['who', 'what', 'when', 'where', 'why', 'how']:
                    payload[f'dominant_{field}'] = merged_event.dominant_pattern.get(field, '')
            
            # Add group-level fields if present
            if hasattr(merged_event, 'group_why'):
                payload['group_why'] = merged_event.group_why
            if hasattr(merged_event, 'group_how'):
                payload['group_how'] = merged_event.group_how
            
            # Add any additional metadata passed in (e.g., latest_who, latest_what, etc.)
            if metadata:
                payload.update(metadata)
            
            # Store in Qdrant
            # Ensure ID is a valid UUID
            import uuid
            try:
                # Try to parse as UUID
                uuid.UUID(merged_event.id)
                point_id = merged_event.id
            except ValueError:
                # Generate a deterministic UUID from the string ID
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, merged_event.id))
            
            point = PointStruct(
                id=point_id,
                vector={
                    "euclidean": euclidean_embedding.tolist(),
                    "hyperbolic": hyperbolic_embedding.tolist()
                },
                payload=payload
            )
            
            self.client.upsert(
                collection_name="merged_events",
                points=[point]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store merged event: {e}")
            return False
    
    def retrieve_events_by_query(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_conditions: Optional[Dict] = None,
        hyperbolic_query: Optional[np.ndarray] = None,
        lambda_e: float = 0.5,
        lambda_h: float = 0.5
    ) -> List[Tuple[Event, float]]:
        """
        Retrieve events using dual-space similarity search
        
        Args:
            query_embedding: Euclidean query vector
            k: Number of results
            filter_conditions: Optional metadata filters
            hyperbolic_query: Optional hyperbolic query vector
            lambda_e: Weight for Euclidean space
            lambda_h: Weight for Hyperbolic space
        
        Returns:
            List of (event, similarity_score) tuples
        """
        try:
            # Build filter from conditions
            filter_obj = None
            if filter_conditions:
                must_conditions = []
                for key, value in filter_conditions.items():
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                if must_conditions:
                    filter_obj = Filter(must=must_conditions)
            
            # Perform multi-vector search if hyperbolic query provided
            if hyperbolic_query is not None:
                # Search both spaces
                euclidean_results = self.client.search(
                    collection_name="events",
                    query_vector=("euclidean", query_embedding.tolist()),
                    limit=k * 2,  # Get more candidates
                    query_filter=filter_obj
                )
                
                hyperbolic_results = self.client.search(
                    collection_name="events",
                    query_vector=("hyperbolic", hyperbolic_query.tolist()),
                    limit=k * 2,
                    query_filter=filter_obj
                )
                
                # Combine and re-rank results
                combined_scores = {}
                for point in euclidean_results:
                    combined_scores[point.id] = lambda_e * point.score
                
                for point in hyperbolic_results:
                    if point.id in combined_scores:
                        combined_scores[point.id] += lambda_h * point.score
                    else:
                        combined_scores[point.id] = lambda_h * point.score
                
                # Sort by combined score
                sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
                
                # Retrieve full points
                point_ids = [id for id, _ in sorted_ids]
                points = self.client.retrieve(
                    collection_name="events",
                    ids=point_ids
                )
                
                # Convert to events
                events_with_scores = []
                for point in points:
                    event = self._reconstruct_event_from_payload(point.id, point.payload)
                    score = combined_scores[point.id]
                    events_with_scores.append((event, score))
                
            else:
                # Single vector search (Euclidean only)
                results = self.client.search(
                    collection_name="events",
                    query_vector=("euclidean", query_embedding.tolist()),
                    limit=k,
                    query_filter=filter_obj
                )
                
                events_with_scores = []
                for point in results:
                    event = self._reconstruct_event_from_payload(point.id, point.payload)
                    events_with_scores.append((event, point.score))
            
            return events_with_scores
            
        except Exception as e:
            logger.error(f"Failed to retrieve events: {e}")
            return []
    
    def retrieve_merged_events_by_query(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        merge_type: Optional[str] = None,
        hyperbolic_query: Optional[np.ndarray] = None,
        lambda_e: float = 0.5,
        lambda_h: float = 0.5
    ) -> List[Tuple[MergedEvent, float]]:
        """
        Retrieve merged events using dual-space similarity search
        
        Args:
            query_embedding: Euclidean query vector
            k: Number of results
            merge_type: Filter by merge type (actor, temporal, conceptual, spatial)
            hyperbolic_query: Optional hyperbolic query vector
            lambda_e: Weight for Euclidean space
            lambda_h: Weight for Hyperbolic space
        
        Returns:
            List of (merged_event, similarity_score) tuples
        """
        try:
            # Build filter
            filter_obj = None
            if merge_type:
                filter_obj = Filter(
                    must=[FieldCondition(key="merge_type", match=MatchValue(value=merge_type))]
                )
            
            # Perform dual-space search if hyperbolic query provided
            if hyperbolic_query is not None:
                # Search both spaces
                euclidean_results = self.client.search(
                    collection_name="merged_events",
                    query_vector=("euclidean", query_embedding.tolist()),
                    limit=k * 2,
                    query_filter=filter_obj
                )
                
                hyperbolic_results = self.client.search(
                    collection_name="merged_events",
                    query_vector=("hyperbolic", hyperbolic_query.tolist()),
                    limit=k * 2,
                    query_filter=filter_obj
                )
                
                # Combine scores
                combined_scores = {}
                id_to_payload = {}
                
                for point in euclidean_results:
                    combined_scores[point.id] = lambda_e * point.score
                    id_to_payload[point.id] = point.payload
                
                for point in hyperbolic_results:
                    if point.id in combined_scores:
                        combined_scores[point.id] += lambda_h * point.score
                    else:
                        combined_scores[point.id] = lambda_h * point.score
                        id_to_payload[point.id] = point.payload
                
                # Sort and get top-k
                sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
                
                # Reconstruct merged events
                merged_events_with_scores = []
                for event_id, score in sorted_results:
                    merged_event = self._reconstruct_merged_event_from_payload(
                        event_id, id_to_payload[event_id]
                    )
                    merged_events_with_scores.append((merged_event, score))
                
            else:
                # Single vector search
                results = self.client.search(
                    collection_name="merged_events",
                    query_vector=("euclidean", query_embedding.tolist()),
                    limit=k,
                    query_filter=filter_obj
                )
                
                merged_events_with_scores = []
                for point in results:
                    merged_event = self._reconstruct_merged_event_from_payload(
                        point.id, point.payload
                    )
                    merged_events_with_scores.append((merged_event, point.score))
            
            return merged_events_with_scores
            
        except Exception as e:
            logger.error(f"Failed to retrieve merged events: {e}")
            return []
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """Get a specific event by ID"""
        # Check cache first
        if event_id in self.cache['events']:
            return self.cache['events'][event_id]
        
        try:
            points = self.client.retrieve(
                collection_name="events",
                ids=[event_id]
            )
            
            if points:
                event = self._reconstruct_event_from_payload(points[0].id, points[0].payload)
                self.cache['events'][event_id] = event
                return event
                
        except Exception as e:
            logger.error(f"Failed to get event {event_id}: {e}")
        
        return None
    
    def get_merged_event(self, merged_event_id: str) -> Optional[MergedEvent]:
        """Get a specific merged event by ID"""
        try:
            points = self.client.retrieve(
                collection_name="merged_events",
                ids=[merged_event_id]
            )
            
            if points:
                return self._reconstruct_merged_event_from_payload(
                    points[0].id, points[0].payload
                )
                
        except Exception as e:
            logger.error(f"Failed to get merged event {merged_event_id}: {e}")
        
        return None
    
    def delete_event(self, event_id: str) -> bool:
        """Delete an event from the database"""
        try:
            self.client.delete(
                collection_name="events",
                points_selector=PointIdsList(points=[event_id])
            )
            # Remove from cache if present
            if event_id in self.cache['events']:
                del self.cache['events'][event_id]
            return True
        except Exception as e:
            logger.error(f"Failed to delete event {event_id}: {e}")
            return False
    
    def get_all_events(self) -> List[Dict]:
        """Get all events from the database"""
        try:
            # Scroll through all events
            results = []
            offset = None
            
            while True:
                batch = self.client.scroll(
                    collection_name="events",
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                points, next_offset = batch
                
                for point in points:
                    # Convert to dict format expected by web app
                    event_dict = {
                        'id': str(point.id),
                        'metadatas': point.payload
                    }
                    results.append(event_dict)
                
                if next_offset is None:
                    break
                offset = next_offset
            
            return {'ids': [r['id'] for r in results], 
                    'metadatas': [r['metadatas'] for r in results]}
            
        except Exception as e:
            logger.error(f"Failed to get all events: {e}")
            return {'ids': [], 'metadatas': []}
    
    def get_episode_events(self, episode_id: str) -> List[Event]:
        """Get all events in an episode"""
        try:
            # Search with episode filter
            filter_obj = Filter(
                must=[FieldCondition(key="episode_id", match=MatchValue(value=episode_id))]
            )
            
            # Scroll through all results
            results = self.client.scroll(
                collection_name="events",
                scroll_filter=filter_obj,
                limit=1000  # Get all in one batch
            )
            
            events = []
            for point in results[0]:  # results is a tuple (points, next_offset)
                event = self._reconstruct_event_from_payload(point.id, point.payload)
                events.append(event)
            
            # Sort by timestamp
            events.sort(key=lambda e: e.created_at)
            return events
            
        except Exception as e:
            logger.error(f"Failed to get episode events: {e}")
            return []
    
    def store_block(self, block: MemoryBlock) -> bool:
        """Store a memory block"""
        try:
            # Prepare payload
            payload = {
                "block_type": block.block_type,
                "block_salience": float(block.block_salience),
                "coherence_score": float(block.coherence_score),
                "event_count": len(block.events),
                "link_count": len(block.links),
                "created_at": block.created_at.isoformat(),
                "updated_at": block.updated_at.isoformat(),
                "embedding_version": block.embedding_version,
                "event_ids": list(block.event_ids),
                "tags": block.tags
            }
            
            # Use block embedding or create placeholder
            if block.block_embedding is not None:
                embedding = block.block_embedding
            else:
                embedding = np.zeros(self.config.memory.embedding_dim)
            
            # Store in Qdrant
            point = PointStruct(
                id=block.id,
                vector={"euclidean": embedding.tolist()},
                payload=payload
            )
            
            self.client.upsert(
                collection_name="memory_blocks",
                points=[point]
            )
            
            # Update cache
            self.cache['blocks'][block.id] = block
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store block: {e}")
            return False
    
    def get_block(self, block_id: str) -> Optional[MemoryBlock]:
        """Get a specific block by ID"""
        if block_id in self.cache['blocks']:
            return self.cache['blocks'][block_id]
        
        try:
            points = self.client.retrieve(
                collection_name="memory_blocks",
                ids=[block_id]
            )
            
            if points:
                block = self._reconstruct_block_from_payload(
                    points[0].id, points[0].payload
                )
                self.cache['blocks'][block_id] = block
                return block
                
        except Exception as e:
            logger.error(f"Failed to get block {block_id}: {e}")
        
        return None
    
    def update_event(self, event: Event) -> bool:
        """Update an event in the database"""
        try:
            # Get existing vectors
            existing = self.client.retrieve(
                collection_name="events",
                ids=[event.id],
                with_vectors=True
            )
            
            if not existing:
                logger.warning(f"Event {event.id} not found for update")
                return False
            
            # Update payload
            payload = existing[0].payload
            payload.update({
                "event_type": event.event_type.value,
                "confidence": float(event.confidence),
                "who": event.five_w1h.who or "",
                "what": event.five_w1h.what or "",
                "when": event.five_w1h.when or "",
                "where": event.five_w1h.where or "",
                "why": event.five_w1h.why or "",
                "how": event.five_w1h.how or "",
                "full_content": event.full_content or "",
            })
            
            # Update with existing vectors
            point = PointStruct(
                id=event.id,
                vector=existing[0].vector,
                payload=payload
            )
            
            self.client.upsert(
                collection_name="events",
                points=[point]
            )
            
            # Update cache
            self.cache['events'][event.id] = event
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update event: {e}")
            return False
    
    def update_embedding(self, event_id: str, euclidean_embedding: np.ndarray,
                        hyperbolic_embedding: Optional[np.ndarray] = None) -> bool:
        """Update embeddings for an event"""
        try:
            # Get existing payload
            existing = self.client.retrieve(
                collection_name="events",
                ids=[event_id]
            )
            
            if not existing:
                return False
            
            vectors = {"euclidean": euclidean_embedding.tolist()}
            if hyperbolic_embedding is not None:
                vectors["hyperbolic"] = hyperbolic_embedding.tolist()
            
            # Update with new vectors
            point = PointStruct(
                id=event_id,
                vector=vectors,
                payload=existing[0].payload
            )
            
            self.client.upsert(
                collection_name="events",
                points=[point]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update embedding: {e}")
            return False
    
    def get_embedding(self, event_id: str) -> Optional[np.ndarray]:
        """Get Euclidean embedding for an event"""
        try:
            points = self.client.retrieve(
                collection_name="events",
                ids=[event_id],
                with_vectors=True
            )
            
            if points and "euclidean" in points[0].vector:
                return np.array(points[0].vector["euclidean"])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return None
    
    def get_dual_embeddings(self, event_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get both Euclidean and Hyperbolic embeddings for an event"""
        try:
            points = self.client.retrieve(
                collection_name="events",
                ids=[event_id],
                with_vectors=True
            )
            
            if points:
                euclidean = np.array(points[0].vector.get("euclidean", []))
                hyperbolic = np.array(points[0].vector.get("hyperbolic", []))
                return euclidean, hyperbolic
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get dual embeddings: {e}")
            return None
    
    def delete_event(self, event_id: str) -> bool:
        """Delete an event from the database"""
        try:
            self.client.delete(
                collection_name="events",
                points_selector=[event_id]
            )
            if event_id in self.cache['events']:
                del self.cache['events'][event_id]
            return True
        except Exception as e:
            logger.error(f"Failed to delete event: {e}")
            return False
    
    def delete_block(self, block_id: str) -> bool:
        """Delete a block from the database"""
        try:
            self.client.delete(
                collection_name="memory_blocks",
                points_selector=[block_id]
            )
            if block_id in self.cache['blocks']:
                del self.cache['blocks'][block_id]
            return True
        except Exception as e:
            logger.error(f"Failed to delete block: {e}")
            return False
    
    def retrieve_all_events(self) -> List[Event]:
        """Retrieve all events from the database"""
        try:
            # Scroll through all events
            results, _ = self.client.scroll(
                collection_name="events",
                limit=10000  # Large batch size
            )
            
            events = []
            for point in results:
                event = self._reconstruct_event_from_payload(point.id, point.payload)
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to retrieve all events: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            events_info = self.client.get_collection("events")
            merged_info = self.client.get_collection("merged_events")
            blocks_info = self.client.get_collection("memory_blocks")
            
            return {
                "total_events": events_info.points_count,
                "total_merged_events": merged_info.points_count,
                "total_blocks": blocks_info.points_count,
                "episodes": len(self.cache['episode_map']),
                "cache_size": {
                    "events": len(self.cache['events']),
                    "blocks": len(self.cache['blocks'])
                }
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Alias for get_statistics for compatibility"""
        return self.get_statistics()
    
    def clear_all(self):
        """Clear all collections (use with caution!)"""
        try:
            self.client.delete_collection("events")
            self.client.delete_collection("merged_events")
            self.client.delete_collection("memory_blocks")
            self._init_collections()
            self.cache = {'blocks': {}, 'events': {}, 'episode_map': {}}
            logger.info("Cleared all collections")
        except Exception as e:
            logger.error(f"Failed to clear collections: {e}")
    
    def _reconstruct_event_from_payload(self, event_id: str, payload: Dict) -> Event:
        """Reconstruct an Event object from Qdrant payload"""
        five_w1h = FiveW1H(
            who=payload.get('who', ''),
            what=payload.get('what', ''),
            when=payload.get('when', ''),
            where=payload.get('where', ''),
            why=payload.get('why', ''),
            how=payload.get('how', '')
        )
        
        event = Event(
            five_w1h=five_w1h,
            id=event_id,
            event_type=EventType(payload.get('event_type', 'observation')),
            episode_id=payload.get('episode_id', ''),
            confidence=payload.get('confidence', 1.0),
            full_content=payload.get('full_content', '')
        )
        
        # Set timestamps if available
        if 'created_at' in payload:
            try:
                event.created_at = datetime.fromisoformat(payload['created_at'])
            except:
                pass
        
        return event
    
    def _reconstruct_merged_event_from_payload(self, event_id: str, payload: Dict) -> MergedEvent:
        """Reconstruct a MergedEvent object from Qdrant payload"""
        merged_event = MergedEvent(
            id=event_id,
            base_event_id=payload.get('base_event_id', '')
        )
        
        merged_event.merge_count = payload.get('merge_count', 1)
        merged_event.confidence_score = payload.get('confidence_score', 1.0)
        merged_event.coherence_score = payload.get('coherence_score', 1.0)
        merged_event.component_ids = set(payload.get('component_ids', []))
        
        # Handle raw_event_ids - could be comma-separated string or list
        raw_ids = payload.get('raw_event_ids', [])
        if isinstance(raw_ids, str):
            # If it's a comma-separated string, split it
            merged_event.raw_event_ids = set(raw_ids.split(',')) if raw_ids else set()
        else:
            # Otherwise treat as list
            merged_event.raw_event_ids = set(raw_ids) if raw_ids else set()
        
        # Reconstruct dominant pattern
        if any(k.startswith('dominant_') for k in payload):
            merged_event.dominant_pattern = {
                field: payload.get(f'dominant_{field}', '')
                for field in ['who', 'what', 'when', 'where', 'why', 'how']
            }
        
        # Add group-level fields if present
        if 'group_why' in payload:
            merged_event.group_why = payload['group_why']
        if 'group_how' in payload:
            merged_event.group_how = payload['group_how']
        
        # Set timestamps
        if 'created_at' in payload:
            try:
                merged_event.created_at = datetime.fromisoformat(payload['created_at'])
            except:
                pass
        if 'last_updated' in payload:
            try:
                merged_event.last_updated = datetime.fromisoformat(payload['last_updated'])
            except:
                pass
        
        return merged_event
    
    def _reconstruct_block_from_payload(self, block_id: str, payload: Dict) -> MemoryBlock:
        """Reconstruct a MemoryBlock object from Qdrant payload"""
        block = MemoryBlock(id=block_id)
        
        block.block_type = payload.get('block_type', 'unknown')
        block.block_salience = payload.get('block_salience', 0.0)
        block.coherence_score = payload.get('coherence_score', 0.0)
        block.embedding_version = payload.get('embedding_version', 0)
        
        # Set timestamps
        if 'created_at' in payload:
            try:
                block.created_at = datetime.fromisoformat(payload['created_at'])
            except:
                pass
        if 'updated_at' in payload:
            try:
                block.updated_at = datetime.fromisoformat(payload['updated_at'])
            except:
                pass
        
        # Reconstruct event list
        event_ids = payload.get('event_ids', [])
        block.event_ids = set(event_ids)
        
        # Get actual events
        block.events = []
        for event_id in event_ids:
            event = self.get_event(event_id)
            if event:
                block.events.append(event)
        
        block.tags = payload.get('tags', [])
        
        return block
    
    def _refresh_cache(self):
        """Refresh the cache from database"""
        try:
            # Get sample of recent blocks for cache
            try:
                results, _ = self.client.scroll(
                    collection_name="memory_blocks",
                    limit=100
                )
                
                for point in results:
                    block = self._reconstruct_block_from_payload(
                        point.id, point.payload
                    )
                    self.cache['blocks'][point.id] = block
                
                logger.info(f"Cache refreshed with {len(self.cache['blocks'])} blocks")
            except:
                logger.info("No blocks to cache yet")
            
        except Exception as e:
            logger.error(f"Failed to refresh cache: {e}")
    
    def update_similarity_cache(self, event_id: str, embeddings: Dict):
        """Update similarity cache with new event embeddings"""
        self.similarity_cache.add_embedding(event_id, embeddings, update_similarities=True)
    
    def batch_update_similarity_cache(self, embeddings_dict: Dict[str, Dict]):
        """Batch update similarity cache with multiple embeddings"""
        self.similarity_cache.batch_add_embeddings(embeddings_dict)
    
    def get_cached_similarity(self, id1: str, id2: str, lambda_e: float = 0.5, 
                            lambda_h: float = 0.5) -> Optional[float]:
        """Get cached similarity between two events"""
        return self.similarity_cache.get_similarity(id1, id2, lambda_e, lambda_h)
    
    def get_top_similar_events(self, event_id: str, k: int = 10, 
                              exclude_ids: Set[str] = None) -> List[Tuple[str, float]]:
        """Get top-k most similar events from cache"""
        return self.similarity_cache.get_top_k_similar(event_id, k, exclude_ids)
    
    def get_similarity_stats(self) -> Dict:
        """Get similarity cache statistics"""
        return self.similarity_cache.get_stats()
    
    def update_provenance(self, event_id: str, provenance_data: Dict) -> bool:
        """Update provenance information for an event"""
        try:
            existing = self.client.retrieve(
                collection_name="events",
                ids=[event_id]
            )
            
            if not existing:
                return False
            
            payload = existing[0].payload
            payload.update(provenance_data)
            payload['version'] = payload.get('version', 1) + 1
            
            # Update point with same vectors
            self.client.set_payload(
                collection_name="events",
                payload=payload,
                points=[event_id]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update provenance: {e}")
            return False
    
    def get_provenance(self, event_id: str) -> Optional[Dict]:
        """Get provenance information for an event"""
        try:
            points = self.client.retrieve(
                collection_name="events",
                ids=[event_id]
            )
            
            if not points:
                return None
            
            payload = points[0].payload
            
            return {
                'version': payload.get('version', 1),
                'last_accessed': payload.get('last_accessed', ''),
                'access_count': payload.get('access_count', 0),
                'euclidean_weight': payload.get('euclidean_weight', 0.5),
                'hyperbolic_weight': payload.get('hyperbolic_weight', 0.5),
                'created_at': payload.get('created_at', '')
            }
            
        except Exception as e:
            logger.error(f"Failed to get provenance: {e}")
            return None
    
    def _setup_collection_aliases(self):
        """Setup collection aliases for compatibility"""
        # Create mock collection objects for dimensional collections
        class MockCollection:
            def __init__(self, store, collection_name, merge_type):
                self.store = store
                self.collection_name = collection_name
                self.merge_type = merge_type
            
            def upsert(self, ids, embeddings, documents=None, metadatas=None):
                # Forward to QdrantStore's merged_events collection with merge_type
                if not metadatas:
                    metadatas = [{}] * len(ids)
                if not documents:
                    documents = ['{}'] * len(ids)
                    
                for i, id in enumerate(ids):
                    if metadatas[i] is None:
                        metadatas[i] = {}
                    metadatas[i]['merge_type'] = self.merge_type
                    
                    # Convert to merged event and store
                    from models.merged_event import MergedEvent
                    import json
                    
                    try:
                        doc_data = json.loads(documents[i]) if isinstance(documents[i], str) else documents[i] if documents[i] else {}
                    except:
                        doc_data = {}
                    
                    merged_event = MergedEvent(id=id, base_event_id=metadatas[i].get('base_event_id', id))
                    # Set default attributes if not present
                    if not hasattr(merged_event, 'confidence_score'):
                        merged_event.confidence_score = 1.0
                    if not hasattr(merged_event, 'coherence_score'):
                        merged_event.coherence_score = 1.0
                    if not hasattr(merged_event, 'raw_event_ids'):
                        merged_event.raw_event_ids = set()
                    
                    # Populate raw_event_ids from metadata
                    if 'component_ids' in metadatas[i] and metadatas[i]['component_ids']:
                        merged_event.raw_event_ids = set(metadatas[i]['component_ids'])
                    elif 'raw_event_ids' in metadatas[i] and metadatas[i]['raw_event_ids']:
                        # Parse comma-separated string if that's what we have
                        raw_ids_str = metadatas[i]['raw_event_ids']
                        if isinstance(raw_ids_str, str):
                            merged_event.raw_event_ids = set(id.strip() for id in raw_ids_str.split(',') if id.strip())
                        elif isinstance(raw_ids_str, list):
                            merged_event.raw_event_ids = set(raw_ids_str)
                    
                    # Store with dual embeddings
                    euclidean = np.array(embeddings[i]) if len(embeddings[i]) == 768 else np.array(embeddings[i][:768])
                    hyperbolic = np.random.randn(64).astype(np.float32) * 0.9  # Generate hyperbolic
                    
                    try:
                        # Pass through all the metadata
                        self.store.store_merged_event(merged_event, euclidean, hyperbolic, self.merge_type, metadata=metadatas[i])
                    except Exception as e:
                        # Log but don't fail - MockCollection is for compatibility only
                        logger.debug(f"MockCollection store failed (non-critical): {e}")
            
            def get(self, **kwargs):
                # Forward to QdrantStore with merge_type filter
                try:
                    # Query the merged_events collection with merge_type filter
                    from qdrant_client.models import Filter, FieldCondition, MatchValue
                    
                    filter_obj = Filter(
                        must=[
                            FieldCondition(
                                key="merge_type",
                                match=MatchValue(value=self.merge_type)
                            )
                        ]
                    )
                    
                    # Scroll through all matching merge groups
                    all_points = []
                    offset = None
                    
                    while True:
                        result = self.store.client.scroll(
                            collection_name="merged_events",
                            scroll_filter=filter_obj,
                            limit=100,
                            offset=offset,
                            with_payload=True,
                            with_vectors=True
                        )
                        
                        points, next_offset = result
                        all_points.extend(points)
                        
                        if next_offset is None:
                            break
                        offset = next_offset
                    
                    # Convert to ChromaDB-like format for compatibility
                    ids = []
                    metadatas = []
                    embeddings = []
                    
                    for point in all_points:
                        ids.append(str(point.id))
                        metadatas.append(point.payload)
                        # Get euclidean embedding if it exists
                        if point.vector and isinstance(point.vector, dict) and 'euclidean' in point.vector:
                            embeddings.append(point.vector['euclidean'])
                        else:
                            embeddings.append(None)
                    
                    return {'ids': ids, 'metadatas': metadatas, 'embeddings': embeddings}
                except Exception as e:
                    logger.warning(f"MockCollection.get failed: {e}")
                    return {'ids': [], 'documents': [], 'metadatas': []}
        
        # Create mock collections
        self.actor_merges_collection = MockCollection(self, 'actor_merges', 'actor')
        self.temporal_merges_collection = MockCollection(self, 'temporal_merges', 'temporal')
        self.conceptual_merges_collection = MockCollection(self, 'conceptual_merges', 'conceptual')
        self.spatial_merges_collection = MockCollection(self, 'spatial_merges', 'spatial')
        
        # Also expose the real collections for direct access
        self.events_collection = 'events'
        self.merged_events_collection = 'merged_events'
        self.memory_blocks_collection = 'memory_blocks'
    
    def __repr__(self) -> str:
        """String representation"""
        stats = self.get_statistics()
        return f"QdrantStore(events={stats.get('total_events', 0)}, merged={stats.get('total_merged_events', 0)}, blocks={stats.get('total_blocks', 0)})"