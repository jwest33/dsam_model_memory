"""
Embedding system for converting 5W1H events to vectors

Supports both transformer-based and hash-based embeddings with role awareness.
"""

import hashlib
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import logging
from pathlib import Path
import json

from config import get_config
from models.event import FiveW1H, Event

logger = logging.getLogger(__name__)

class TextEmbedder:
    """Base text embedding system with transformer and hash fallback"""
    
    def __init__(self, config=None):
        """Initialize embedder with config"""
        self.config = config or get_config().embedding
        self.model = None
        self.cache = {}
        
        # Try to load transformer model
        if self.config.use_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.config.model_name)
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                logger.info(f"Loaded transformer model: {self.config.model_name}")
            except ImportError:
                logger.warning("sentence-transformers not available, falling back to hash embeddings")
                self.config.use_transformer = False
                self.embedding_dim = self.config.hash_dim
            except Exception as e:
                logger.warning(f"Failed to load transformer model: {e}, falling back to hash embeddings")
                self.config.use_transformer = False
                self.embedding_dim = self.config.hash_dim
        else:
            self.embedding_dim = self.config.hash_dim
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string"""
        if self.config.cache_embeddings and text in self.cache:
            return self.cache[text].copy()
        
        if self.config.use_transformer and self.model:
            embedding = self.model.encode(text, convert_to_numpy=True)
        else:
            embedding = self._hash_embed(text)
        
        if self.config.cache_embeddings:
            self.cache[text] = embedding.copy()
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts efficiently"""
        if self.config.use_transformer and self.model:
            # Check cache for any existing embeddings
            uncached_texts = []
            uncached_indices = []
            embeddings = np.zeros((len(texts), self.embedding_dim))
            
            for i, text in enumerate(texts):
                if self.config.cache_embeddings and text in self.cache:
                    embeddings[i] = self.cache[text]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Batch encode uncached texts
            if uncached_texts:
                new_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)
                for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                    embeddings[idx] = emb
                    if self.config.cache_embeddings:
                        self.cache[text] = emb.copy()
            
            return embeddings
        else:
            # Hash-based fallback
            return np.array([self.embed_text(text) for text in texts])
    
    def _hash_embed(self, text: str) -> np.ndarray:
        """Create deterministic hash-based embedding"""
        # Use multiple hash functions for better distribution
        embedding = np.zeros(self.embedding_dim)
        
        for i, salt in enumerate(['md5', 'sha1', 'sha256']):
            hash_obj = hashlib.new('sha256')
            hash_obj.update(f"{salt}:{text}".encode('utf-8'))
            hash_bytes = hash_obj.digest()
            
            # Convert hash to float values
            for j in range(0, len(hash_bytes), 4):
                if i * len(hash_bytes) // 4 + j // 4 >= self.embedding_dim:
                    break
                idx = (i * len(hash_bytes) // 4 + j // 4) % self.embedding_dim
                value = int.from_bytes(hash_bytes[j:j+4], 'big') / (2**32)
                embedding[idx] = 2 * value - 1  # Scale to [-1, 1]
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    
    def save_cache(self, path: Path):
        """Save embedding cache to file"""
        if not self.cache:
            return
        
        cache_data = {
            text: emb.tolist() 
            for text, emb in self.cache.items()
        }
        
        with open(path, 'w') as f:
            json.dump(cache_data, f)
        
        logger.info(f"Saved {len(self.cache)} cached embeddings to {path}")
    
    def load_cache(self, path: Path):
        """Load embedding cache from file"""
        if not path.exists():
            return
        
        try:
            with open(path, 'r') as f:
                cache_data = json.load(f)
            
            self.cache = {
                text: np.array(emb)
                for text, emb in cache_data.items()
            }
            
            logger.info(f"Loaded {len(self.cache)} cached embeddings from {path}")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")


class FiveW1HEmbedder:
    """Specialized embedder for 5W1H events with role awareness"""
    
    # Role embedding vectors (learned or predefined)
    ROLE_VECTORS = {
        "who": np.array([1, 0, 0, 0, 0, 0]),
        "what": np.array([0, 1, 0, 0, 0, 0]),
        "when": np.array([0, 0, 1, 0, 0, 0]),
        "where": np.array([0, 0, 0, 1, 0, 0]),
        "why": np.array([0, 0, 0, 0, 1, 0]),
        "how": np.array([0, 0, 0, 0, 0, 1])
    }
    
    def __init__(self, text_embedder: Optional[TextEmbedder] = None):
        """Initialize with text embedder"""
        self.text_embedder = text_embedder or TextEmbedder()
        self.config = get_config().embedding
        
        # Expand role vectors to match embedding dimension
        self.role_embeddings = {}
        for role, vector in self.ROLE_VECTORS.items():
            # Repeat and scale role vector to match dimension
            expanded = np.tile(vector, (self.text_embedder.embedding_dim // 6 + 1))
            self.role_embeddings[role] = expanded[:self.text_embedder.embedding_dim]
            self.role_embeddings[role] = self.role_embeddings[role] / np.linalg.norm(self.role_embeddings[role])
    
    def embed_slot(self, slot_name: str, slot_value: str) -> np.ndarray:
        """Embed a single 5W1H slot with role information"""
        # Get text embedding
        text_emb = self.text_embedder.embed_text(slot_value)
        
        # Add role embedding if configured
        if self.config.add_role_embeddings and slot_name in self.role_embeddings:
            role_emb = self.role_embeddings[slot_name]
            # Weighted combination
            combined = text_emb + self.config.role_embedding_scale * role_emb
            # Normalize
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm
            return combined
        
        return text_emb
    
    def embed_five_w1h(self, five_w1h: FiveW1H) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed a complete 5W1H structure
        
        Returns:
            key: Fused embedding of all slots (for matching)
            value: Content embedding (what + full_content if available)
        """
        # Embed each slot
        slot_embeddings = {
            "who": self.embed_slot("who", five_w1h.who),
            "what": self.embed_slot("what", five_w1h.what),
            "when": self.embed_slot("when", five_w1h.when),
            "where": self.embed_slot("where", five_w1h.where),
            "why": self.embed_slot("why", five_w1h.why),
            "how": self.embed_slot("how", five_w1h.how)
        }
        
        # Create key: weighted average of all slots
        key = np.mean(list(slot_embeddings.values()), axis=0)
        
        # Create value: focus on content (what)
        value = slot_embeddings["what"]
        
        return key, value
    
    def embed_event(self, event: Event) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed a complete event
        
        Returns:
            key: Query embedding for retrieval
            value: Content embedding for storage
        """
        key, value = self.embed_five_w1h(event.five_w1h)
        
        # If there's full content, use it for value
        if event.full_content:
            value = self.text_embedder.embed_text(event.full_content)
        
        # Normalize key (salience weighting now handled by block salience matrix)
        key_norm = np.linalg.norm(key)
        if key_norm > 0:
            key = key / key_norm
        
        return key, value
    
    def embed_partial_query(self, partial: Dict[str, str]) -> np.ndarray:
        """Embed a partial 5W1H query for retrieval"""
        if not partial:
            return np.zeros(self.text_embedder.embedding_dim)
        
        embeddings = []
        for slot_name, slot_value in partial.items():
            if slot_value:
                embeddings.append(self.embed_slot(slot_name, slot_value))
        
        if not embeddings:
            return np.zeros(self.text_embedder.embedding_dim)
        
        # Average the available slots
        query = np.mean(embeddings, axis=0)
        
        # Normalize
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        
        return query
    
    def compute_similarity(
        self,
        query: Union[FiveW1H, Dict[str, str], np.ndarray],
        target: Union[FiveW1H, Event, np.ndarray]
    ) -> float:
        """Compute similarity between query and target"""
        # Convert query to embedding
        if isinstance(query, np.ndarray):
            query_emb = query
        elif isinstance(query, FiveW1H):
            query_emb, _ = self.embed_five_w1h(query)
        elif isinstance(query, dict):
            query_emb = self.embed_partial_query(query)
        else:
            raise ValueError(f"Unsupported query type: {type(query)}")
        
        # Convert target to embedding
        if isinstance(target, np.ndarray):
            target_emb = target
        elif isinstance(target, Event):
            target_emb, _ = self.embed_event(target)
        elif isinstance(target, FiveW1H):
            target_emb, _ = self.embed_five_w1h(target)
        else:
            raise ValueError(f"Unsupported target type: {type(target)}")
        
        return self.text_embedder.cosine_similarity(query_emb, target_emb)
    
    def batch_embed_events(self, events: List[Event]) -> Tuple[np.ndarray, np.ndarray]:
        """Batch embed multiple events efficiently"""
        keys = []
        values = []
        
        for event in events:
            key, value = self.embed_event(event)
            keys.append(key)
            values.append(value)
        
        return np.array(keys), np.array(values)
