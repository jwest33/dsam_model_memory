"""
Learned dimension classifier using vector similarity to example queries.

This module provides a vector-based approach to dimension classification,
learning from both synthetic examples and actual query history.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

from models.merge_types import MergeType

logger = logging.getLogger(__name__)


class LearnedDimensionClassifier:
    """
    Classifies queries into dimensions using vector similarity to learned examples.
    
    Instead of regex patterns, this uses:
    - A collection of example queries with known dimension weights
    - Vector similarity to find most relevant examples
    - Weighted blending of similar examples for final classification
    """
    
    def __init__(self, encoder, storage_backend=None, similarity_threshold=0.7):
        """
        Initialize the learned classifier.
        
        Args:
            encoder: Dual-space encoder for embeddings
            storage_backend: Optional storage for persisting learned examples
            similarity_threshold: Minimum similarity to consider an example relevant
        """
        self.encoder = encoder
        self.storage = storage_backend
        self.similarity_threshold = similarity_threshold
        
        # Collection name for storing query examples
        self.collection_name = "query_dimension_examples"
        
        # Initialize or load the examples collection
        self._initialize_collection()
        
        # Cache for performance
        self.example_cache = {}
        
    def _initialize_collection(self):
        """Initialize the vector collection for query examples."""
        if self.storage:
            try:
                # Create collection if it doesn't exist
                self.collection = self.storage.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"description": "Query to dimension mapping examples"}
                )
                
                # Always sync with JSONL file on startup
                self._sync_with_jsonl()
                    
            except Exception as e:
                logger.error(f"Error initializing examples collection: {e}")
                self.collection = None
        else:
            self.collection = None
            
    def _sync_with_jsonl(self):
        """Sync collection with JSONL file, loading only new/updated examples."""
        import hashlib
        
        # Get existing examples from collection
        existing_examples = set()
        try:
            all_items = self.collection.get(include=['metadatas'])
            for metadata in all_items.get('metadatas', []):
                if metadata and 'query' in metadata:
                    # Create a hash of query + weights for comparison
                    example_key = self._get_example_key(metadata)
                    existing_examples.add(example_key)
                    
            logger.info(f"Found {len(existing_examples)} existing examples in collection")
        except Exception as e:
            logger.warning(f"Could not retrieve existing examples: {e}")
            
        # Load and check JSONL file
        jsonl_path = self._find_jsonl_path()
        if not jsonl_path:
            if len(existing_examples) == 0:
                logger.warning("No JSONL file found and collection is empty!")
            return

        # Load JSONL and add only new examples
        try:
            new_count = 0
            skip_count = 0
            
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        example = json.loads(line)
                        
                        # Convert category to weights
                        weights = self._category_to_weights(example.get('category'))
                        
                        example_data = {
                            'query': example['query'],
                            **weights
                        }
                        example_key = self._get_example_key(example_data)
                        
                        if example_key not in existing_examples:
                            # This is a new example, add it
                            metadata = {'category': example.get('category', 'unknown')}
                            self.add_example(
                                query_text=example['query'],
                                dimension_weights=weights,
                                source='synthetic',
                                metadata=metadata
                            )
                            new_count += 1
                        else:
                            skip_count += 1
                            
            if new_count > 0:
                logger.info(f"Added {new_count} new examples from JSONL (skipped {skip_count} existing)")
            else:
                logger.info(f"All {skip_count} examples from JSONL already in collection")
                
        except Exception as e:
            logger.error(f"Error syncing with JSONL file: {e}")
            
    def _find_jsonl_path(self):
        """Find the JSONL file path."""
        possible_paths = [
            Path("data/query_dimension_examples.jsonl"),
            Path("../data/query_dimension_examples.jsonl"),
            Path(__file__).parent.parent / "data" / "query_dimension_examples.jsonl",
            Path.cwd() / "data" / "query_dimension_examples.jsonl"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        return None
        
    def _get_example_key(self, example_data: Dict) -> str:
        """Generate a unique key for an example based on query and weights."""
        import hashlib
        
        # Create a normalized string representation
        key_parts = [
            example_data.get('query', '').lower().strip(),
            f"actor:{example_data.get('actor', 0):.2f}",
            f"temporal:{example_data.get('temporal', 0):.2f}",
            f"conceptual:{example_data.get('conceptual', 0):.2f}",
            f"spatial:{example_data.get('spatial', 0):.2f}"
        ]
        key_string = "|".join(key_parts)
        
        # Return hash for efficient comparison
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _category_to_weights(self, category: str) -> Dict[str, float]:
        """
        Convert a simple category to dimension weights.
        Uses strong weight for primary dimension, small weights for others.
        
        Args:
            category: One of 'temporal', 'conceptual', 'actor', 'spatial'
            
        Returns:
            Dictionary of dimension weights
        """
        # Default equal weights if category unknown
        weights = {
            'actor': 0.1,
            'temporal': 0.1,
            'conceptual': 0.1,
            'spatial': 0.1
        }
        
        # Set primary dimension to have dominant weight
        if category in weights:
            # Primary dimension gets 70%, others share 30%
            for dim in weights:
                if dim == category:
                    weights[dim] = 0.7
                else:
                    weights[dim] = 0.1
        else:
            # Unknown category - use equal weights
            weights = {
                'actor': 0.25,
                'temporal': 0.25,
                'conceptual': 0.25,
                'spatial': 0.25
            }
            
        return weights
        
    def add_example(self, query_text: str, dimension_weights: Dict[str, float], 
                   source: str = "learned", metadata: Optional[Dict] = None):
        """
        Add a query example with its dimension weights.
        
        Args:
            query_text: The query text
            dimension_weights: Dict mapping dimension names to weights (0-1)
            source: Source of the example ("synthetic", "learned", "user_feedback")
            metadata: Optional additional metadata
        """
        if not self.collection or not self.encoder:
            return
            
        try:
            # Encode the query
            euclidean_emb, hyperbolic_emb = self.encoder.encode_dual(query_text)
            
            # Create metadata
            example_metadata = {
                "query": query_text,
                "source": source,
                "timestamp": datetime.utcnow().isoformat(),
                **dimension_weights  # Add dimension weights as metadata fields
            }
            
            if metadata:
                example_metadata.update(metadata)
            
            # Generate unique ID
            import hashlib
            example_id = hashlib.md5(f"{query_text}_{source}_{datetime.utcnow().isoformat()}".encode()).hexdigest()
            
            # Add to collection (using Euclidean embedding for storage)
            self.collection.add(
                embeddings=[euclidean_emb.tolist()],
                metadatas=[example_metadata],
                ids=[example_id]
            )
            
            # Clear cache
            self.example_cache.clear()
            
        except Exception as e:
            logger.error(f"Error adding example: {e}")
            
    def classify_query(self, query_fields: Dict[str, str], 
                       k_neighbors: int = 10) -> Dict[MergeType, float]:
        """
        Classify a query into dimensions using learned examples.
        
        Args:
            query_fields: Query fields dictionary
            k_neighbors: Number of similar examples to consider
            
        Returns:
            Dictionary mapping MergeType to weight (0-1)
        """
        # Extract query text
        query_text = ' '.join([str(v) for v in query_fields.values() if v])
        
        # Check cache
        cache_key = query_text
        if cache_key in self.example_cache:
            return self.example_cache[cache_key]
        
        # Default weights if no collection
        if not self.collection or not self.encoder:
            return self._get_default_weights()
        
        try:
            # Encode query
            euclidean_emb, hyperbolic_emb = self.encoder.encode_dual(query_text)
            
            # Find similar examples
            results = self.collection.query(
                query_embeddings=[euclidean_emb.tolist()],
                n_results=k_neighbors,
                include=['metadatas', 'distances']
            )
            
            if not results or not results['metadatas'][0]:
                return self._get_default_weights()
            
            # Blend weights from similar examples
            blended_weights = self._blend_example_weights(
                results['metadatas'][0],
                results['distances'][0]
            )
            
            # Cache result
            self.example_cache[cache_key] = blended_weights
            
            return blended_weights
            
        except Exception as e:
            logger.error(f"Error classifying query: {e}")
            return self._get_default_weights()
            
    def _blend_example_weights(self, metadatas: List[Dict], distances: List[float]) -> Dict[MergeType, float]:
        """
        Blend dimension weights from similar examples.
        
        Args:
            metadatas: List of metadata dicts from similar examples
            distances: List of distances to examples
            
        Returns:
            Blended dimension weights
        """
        # Convert distances to similarities
        similarities = [1.0 / (1.0 + dist) for dist in distances]
        
        # Filter by threshold
        relevant_examples = [
            (meta, sim) for meta, sim in zip(metadatas, similarities)
            if sim >= self.similarity_threshold
        ]
        
        if not relevant_examples:
            # If no examples above threshold, use closest one
            if metadatas:
                relevant_examples = [(metadatas[0], similarities[0])]
            else:
                return self._get_default_weights()
        
        # Initialize weights
        dimension_names = ['actor', 'temporal', 'conceptual', 'spatial']
        blended = {name: 0.0 for name in dimension_names}
        total_weight = 0.0
        
        # Weighted average based on similarity
        for metadata, similarity in relevant_examples:
            for dim_name in dimension_names:
                if dim_name in metadata:
                    blended[dim_name] += float(metadata[dim_name]) * similarity
            total_weight += similarity
            
        # Normalize
        if total_weight > 0:
            for dim_name in dimension_names:
                blended[dim_name] /= total_weight
                
        # Ensure sum to 1
        total = sum(blended.values())
        if total > 0:
            for dim_name in dimension_names:
                blended[dim_name] /= total
        else:
            # Equal weights if something went wrong
            for dim_name in dimension_names:
                blended[dim_name] = 0.25
                
        # Convert to MergeType enum
        return {
            MergeType.ACTOR: blended['actor'],
            MergeType.TEMPORAL: blended['temporal'],
            MergeType.CONCEPTUAL: blended['conceptual'],
            MergeType.SPATIAL: blended['spatial']
        }
        
    def _get_default_weights(self) -> Dict[MergeType, float]:
        """Get default equal weights."""
        return {
            MergeType.ACTOR: 0.25,
            MergeType.TEMPORAL: 0.25,
            MergeType.CONCEPTUAL: 0.25,
            MergeType.SPATIAL: 0.25
        }
        
    def update_from_feedback(self, query_text: str, corrected_weights: Dict[str, float]):
        """
        Update the classifier based on user feedback.
        
        Args:
            query_text: The query that was classified
            corrected_weights: The correct dimension weights
        """
        self.add_example(
            query_text=query_text,
            dimension_weights=corrected_weights,
            source="user_feedback"
        )
        
    def export_examples(self, filepath: str):
        """Export learned examples to a JSON file."""
        if not self.collection:
            return
            
        try:
            # Get all examples
            all_examples = self.collection.get(include=['metadatas'])
            
            # Format for export
            export_data = {
                "examples": all_examples['metadatas'],
                "exported_at": datetime.utcnow().isoformat(),
                "total_count": len(all_examples['metadatas'])
            }
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            logger.info(f"Exported {export_data['total_count']} examples to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting examples: {e}")
            
    def import_examples(self, filepath: str):
        """Import examples from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            for example in data.get('examples', []):
                # Extract dimension weights
                weights = {
                    'actor': example.get('actor', 0.25),
                    'temporal': example.get('temporal', 0.25),
                    'conceptual': example.get('conceptual', 0.25),
                    'spatial': example.get('spatial', 0.25)
                }
                
                self.add_example(
                    query_text=example.get('query', ''),
                    dimension_weights=weights,
                    source=example.get('source', 'imported')
                )
                
            logger.info(f"Imported {len(data.get('examples', []))} examples from {filepath}")
            
        except Exception as e:
            logger.error(f"Error importing examples: {e}")
