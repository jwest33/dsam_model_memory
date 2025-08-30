"""
Baseline implementations for comparative evaluation
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from abc import ABC, abstractmethod
import time
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class BaselineSystem(ABC):
    """Abstract base class for baseline systems"""
    
    def __init__(self, name: str):
        self.name = name
        self.metrics = {
            'store_times': [],
            'retrieve_times': [],
            'memory_usage': []
        }
    
    @abstractmethod
    def store(self, text: str, metadata: Dict[str, Any]) -> bool:
        """Store a memory in the system"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve memories based on query"""
        pass
    
    @abstractmethod
    def clear(self):
        """Clear all stored memories"""
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for this baseline"""
        return {
            'avg_store_time': np.mean(self.metrics['store_times']) if self.metrics['store_times'] else 0,
            'avg_retrieve_time': np.mean(self.metrics['retrieve_times']) if self.metrics['retrieve_times'] else 0,
            'avg_memory_usage': np.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
        }


class TFIDFBaseline(BaselineSystem):
    """TF-IDF baseline for text retrieval"""
    
    def __init__(self, max_features: int = 5000):
        super().__init__("TF-IDF")
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.documents = []
        self.metadata_list = []
        self.tfidf_matrix = None
        self.is_fitted = False
    
    def store(self, text: str, metadata: Dict[str, Any]) -> bool:
        """Store document and metadata"""
        start_time = time.time()
        
        self.documents.append(text)
        self.metadata_list.append(metadata)
        
        # Refit vectorizer with all documents
        if len(self.documents) > 0:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            self.is_fitted = True
        
        self.metrics['store_times'].append(time.time() - start_time)
        return True
    
    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve using TF-IDF similarity"""
        start_time = time.time()
        
        if not self.is_fitted or self.tfidf_matrix is None:
            return []
        
        # Transform query
        query_vec = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            if idx < len(self.documents):
                results.append({
                    'text': self.documents[idx],
                    'metadata': self.metadata_list[idx],
                    'score': float(similarities[idx])
                })
        
        self.metrics['retrieve_times'].append(time.time() - start_time)
        return results
    
    def clear(self):
        """Clear all stored data"""
        self.documents = []
        self.metadata_list = []
        self.tfidf_matrix = None
        self.is_fitted = False
        self.vectorizer = TfidfVectorizer(max_features=5000)


class BM25Baseline(BaselineSystem):
    """BM25 baseline for text retrieval"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        super().__init__("BM25")
        self.k1 = k1
        self.b = b
        self.documents = []
        self.metadata_list = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = {}
        self.idf_scores = {}
        self.N = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.lower().split()
    
    def _calculate_idf(self):
        """Calculate IDF scores for all terms"""
        self.idf_scores = {}
        for term in self.doc_freqs:
            df = self.doc_freqs[term]
            self.idf_scores[term] = np.log((self.N - df + 0.5) / (df + 0.5) + 1)
    
    def store(self, text: str, metadata: Dict[str, Any]) -> bool:
        """Store document and update BM25 statistics"""
        start_time = time.time()
        
        # Tokenize and store
        tokens = self._tokenize(text)
        self.documents.append(tokens)
        self.metadata_list.append(metadata)
        self.doc_lengths.append(len(tokens))
        
        # Update document frequency
        unique_tokens = set(tokens)
        for token in unique_tokens:
            self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
        
        # Update statistics
        self.N = len(self.documents)
        self.avg_doc_length = np.mean(self.doc_lengths)
        self._calculate_idf()
        
        self.metrics['store_times'].append(time.time() - start_time)
        return True
    
    def _calculate_bm25_score(self, query_tokens: List[str], doc_tokens: List[str], doc_length: int) -> float:
        """Calculate BM25 score for a document"""
        score = 0.0
        
        # Count term frequencies in document
        doc_tf = {}
        for token in doc_tokens:
            doc_tf[token] = doc_tf.get(token, 0) + 1
        
        for query_term in query_tokens:
            if query_term in doc_tf:
                tf = doc_tf[query_term]
                idf = self.idf_scores.get(query_term, 0)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                
                score += idf * (numerator / denominator)
        
        return score
    
    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve using BM25 scoring"""
        start_time = time.time()
        
        if not self.documents:
            return []
        
        query_tokens = self._tokenize(query)
        
        # Calculate scores for all documents
        scores = []
        for i, (doc_tokens, doc_length) in enumerate(zip(self.documents, self.doc_lengths)):
            score = self._calculate_bm25_score(query_tokens, doc_tokens, doc_length)
            scores.append((i, score))
        
        # Sort by score and get top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:k]:
            results.append({
                'text': ' '.join(self.documents[idx]),
                'metadata': self.metadata_list[idx],
                'score': float(score)
            })
        
        self.metrics['retrieve_times'].append(time.time() - start_time)
        return results
    
    def clear(self):
        """Clear all stored data"""
        self.documents = []
        self.metadata_list = []
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_freqs = {}
        self.idf_scores = {}
        self.N = 0


class SimpleVectorStoreBaseline(BaselineSystem):
    """Simple dense vector store baseline using ChromaDB"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        super().__init__("SimpleVectorStore")
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            is_persistent=False,
            anonymized_telemetry=False
        ))
        
        # Create collection
        self.collection_name = f"baseline_simple_{int(time.time())}"
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.doc_counter = 0
    
    def store(self, text: str, metadata: Dict[str, Any]) -> bool:
        """Store document in vector database"""
        start_time = time.time()
        
        try:
            # Store in ChromaDB
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[f"doc_{self.doc_counter}"]
            )
            self.doc_counter += 1
            
            self.metrics['store_times'].append(time.time() - start_time)
            return True
        except Exception as e:
            logger.error(f"Error storing in vector database: {e}")
            return False
    
    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve using vector similarity"""
        start_time = time.time()
        
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k, self.doc_counter)
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'score': 1.0 - results['distances'][0][i] if results['distances'] else 0.0
                    })
            
            self.metrics['retrieve_times'].append(time.time() - start_time)
            return formatted_results
        except Exception as e:
            logger.error(f"Error retrieving from vector database: {e}")
            return []
    
    def clear(self):
        """Clear collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.doc_counter = 0
        except Exception as e:
            logger.error(f"Error clearing vector database: {e}")


class EuclideanOnlyBaseline(BaselineSystem):
    """Baseline using only Euclidean space embeddings"""
    
    def __init__(self, embedding_dim: int = 768):
        super().__init__("EuclideanOnly")
        self.embedding_dim = embedding_dim
        self.embeddings = []
        self.texts = []
        self.metadata_list = []
    
    def _get_random_embedding(self, text: str) -> np.ndarray:
        """Generate a deterministic pseudo-random embedding for text"""
        # Use hash for deterministic pseudo-randomness
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self.embedding_dim)
    
    def store(self, text: str, metadata: Dict[str, Any]) -> bool:
        """Store with Euclidean embedding"""
        start_time = time.time()
        
        # Generate embedding (in real implementation, use actual encoder)
        embedding = self._get_random_embedding(text)
        
        self.embeddings.append(embedding)
        self.texts.append(text)
        self.metadata_list.append(metadata)
        
        self.metrics['store_times'].append(time.time() - start_time)
        return True
    
    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve using Euclidean distance"""
        start_time = time.time()
        
        if not self.embeddings:
            return []
        
        # Generate query embedding
        query_embedding = self._get_random_embedding(query)
        
        # Calculate Euclidean distances
        embeddings_matrix = np.array(self.embeddings)
        distances = np.linalg.norm(embeddings_matrix - query_embedding, axis=1)
        
        # Get top-k (smallest distances)
        top_k_indices = np.argsort(distances)[:k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'text': self.texts[idx],
                'metadata': self.metadata_list[idx],
                'score': float(1.0 / (1.0 + distances[idx]))  # Convert distance to similarity
            })
        
        self.metrics['retrieve_times'].append(time.time() - start_time)
        return results
    
    def clear(self):
        """Clear all stored data"""
        self.embeddings = []
        self.texts = []
        self.metadata_list = []


class HyperbolicOnlyBaseline(BaselineSystem):
    """Baseline using only Hyperbolic space embeddings"""
    
    def __init__(self, embedding_dim: int = 64):
        super().__init__("HyperbolicOnly")
        self.embedding_dim = embedding_dim
        self.embeddings = []
        self.texts = []
        self.metadata_list = []
    
    def _project_to_poincare_ball(self, x: np.ndarray) -> np.ndarray:
        """Project point to Poincaré ball"""
        norm = np.linalg.norm(x)
        if norm >= 1.0:
            return x / (norm + 1e-6) * 0.99
        return x
    
    def _poincare_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate distance in Poincaré ball model"""
        x_norm = np.linalg.norm(x)
        y_norm = np.linalg.norm(y)
        
        # Ensure points are in the ball
        if x_norm >= 1.0:
            x = x / (x_norm + 1e-6) * 0.99
        if y_norm >= 1.0:
            y = y / (y_norm + 1e-6) * 0.99
        
        # Poincaré distance formula
        diff_norm = np.linalg.norm(x - y)
        x_norm_sq = np.sum(x * x)
        y_norm_sq = np.sum(y * y)
        
        numerator = 2 * diff_norm ** 2
        denominator = (1 - x_norm_sq) * (1 - y_norm_sq)
        
        if denominator <= 0:
            return float('inf')
        
        cosh_dist = 1 + numerator / denominator
        return float(np.arccosh(max(cosh_dist, 1.0)))
    
    def _get_random_hyperbolic_embedding(self, text: str) -> np.ndarray:
        """Generate a deterministic pseudo-random hyperbolic embedding"""
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.embedding_dim) * 0.5  # Start in ball
        return self._project_to_poincare_ball(embedding)
    
    def store(self, text: str, metadata: Dict[str, Any]) -> bool:
        """Store with Hyperbolic embedding"""
        start_time = time.time()
        
        # Generate hyperbolic embedding
        embedding = self._get_random_hyperbolic_embedding(text)
        
        self.embeddings.append(embedding)
        self.texts.append(text)
        self.metadata_list.append(metadata)
        
        self.metrics['store_times'].append(time.time() - start_time)
        return True
    
    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve using Hyperbolic distance"""
        start_time = time.time()
        
        if not self.embeddings:
            return []
        
        # Generate query embedding
        query_embedding = self._get_random_hyperbolic_embedding(query)
        
        # Calculate Hyperbolic distances
        distances = []
        for emb in self.embeddings:
            dist = self._poincare_distance(query_embedding, emb)
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Get top-k (smallest distances)
        top_k_indices = np.argsort(distances)[:k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'text': self.texts[idx],
                'metadata': self.metadata_list[idx],
                'score': float(1.0 / (1.0 + distances[idx]))  # Convert distance to similarity
            })
        
        self.metrics['retrieve_times'].append(time.time() - start_time)
        return results
    
    def clear(self):
        """Clear all stored data"""
        self.embeddings = []
        self.texts = []
        self.metadata_list = []


class NoAdaptationBaseline(BaselineSystem):
    """Baseline without residual adaptation (ablation study)"""
    
    def __init__(self):
        super().__init__("NoAdaptation")
        self.embeddings = []
        self.texts = []
        self.metadata_list = []
    
    def _get_static_embedding(self, text: str) -> np.ndarray:
        """Get static embedding without any adaptation"""
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(384)  # Fixed dimension
    
    def store(self, text: str, metadata: Dict[str, Any]) -> bool:
        """Store with static embedding"""
        start_time = time.time()
        
        embedding = self._get_static_embedding(text)
        self.embeddings.append(embedding)
        self.texts.append(text)
        self.metadata_list.append(metadata)
        
        self.metrics['store_times'].append(time.time() - start_time)
        return True
    
    def retrieve(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve without adaptation"""
        start_time = time.time()
        
        if not self.embeddings:
            return []
        
        query_embedding = self._get_static_embedding(query)
        embeddings_matrix = np.array(self.embeddings)
        
        # Cosine similarity
        similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'text': self.texts[idx],
                'metadata': self.metadata_list[idx],
                'score': float(similarities[idx])
            })
        
        self.metrics['retrieve_times'].append(time.time() - start_time)
        return results
    
    def clear(self):
        """Clear all stored data"""
        self.embeddings = []
        self.texts = []
        self.metadata_list = []


class BaselineManager:
    """Manager for running and comparing baseline systems"""
    
    def __init__(self):
        self.baselines = {
            'tfidf': TFIDFBaseline(),
            'bm25': BM25Baseline(),
            'simple_vector': SimpleVectorStoreBaseline(),
            'euclidean_only': EuclideanOnlyBaseline(),
            'hyperbolic_only': HyperbolicOnlyBaseline(),
            'no_adaptation': NoAdaptationBaseline()
        }
    
    def get_baseline(self, name: str) -> Optional[BaselineSystem]:
        """Get a specific baseline system"""
        return self.baselines.get(name)
    
    def run_baseline_comparison(self, 
                               test_documents: List[Tuple[str, Dict]],
                               test_queries: List[Tuple[str, List[str]]],
                               k: int = 10) -> Dict[str, Dict]:
        """Run comparison across all baselines"""
        results = {}
        
        for baseline_name, baseline in self.baselines.items():
            logger.info(f"Evaluating baseline: {baseline_name}")
            
            # Clear baseline
            baseline.clear()
            
            # Store documents
            for text, metadata in test_documents:
                baseline.store(text, metadata)
            
            # Run queries
            query_results = []
            for query, relevant_docs in test_queries:
                retrieved = baseline.retrieve(query, k)
                retrieved_ids = [r['metadata'].get('id', '') for r in retrieved]
                
                # Calculate metrics
                precision = len(set(retrieved_ids) & set(relevant_docs)) / k
                recall = len(set(retrieved_ids) & set(relevant_docs)) / len(relevant_docs) if relevant_docs else 0
                
                query_results.append({
                    'precision': precision,
                    'recall': recall,
                    'retrieved': retrieved_ids,
                    'relevant': relevant_docs
                })
            
            # Aggregate results
            results[baseline_name] = {
                'precision': np.mean([r['precision'] for r in query_results]),
                'recall': np.mean([r['recall'] for r in query_results]),
                'performance': baseline.get_performance_metrics(),
                'detailed_results': query_results
            }
        
        return results
