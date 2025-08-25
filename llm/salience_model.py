"""
Block-based salience computation using embedding matrix

This module is now simplified since salience is computed at the block level
using the salience matrix approach rather than individual event salience.
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np

from config import get_config
from llm.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class SalienceModel:
    """
    Model for computing block-level salience.
    Individual event salience is now handled by the MemoryBlock's salience matrix.
    """
    
    def __init__(self, llm_interface: Optional[LLMInterface] = None):
        """Initialize salience model"""
        self.config = get_config()
        self.llm = llm_interface or LLMInterface(self.config.llm)
    
    def evaluate_block_importance(
        self,
        block_summary: str,
        goal: Optional[str] = None,
        context: Optional[str] = None
    ) -> float:
        """
        Evaluate the importance of a memory block using LLM.
        This is used for high-level block importance assessment.
        
        Args:
            block_summary: Summary of the block's contents
            goal: Current goal or context
            context: Additional context
        
        Returns:
            Importance score (0-1)
        """
        if not self.config.llm.use_llm_salience or not self.llm.is_available():
            # Simple fallback: return neutral importance
            return 0.5
        
        try:
            prompt = f"""Rate the importance of this memory block (0-1 scale).
Goal: {goal or "General knowledge"}
Context: {context or "No specific context"}
Block summary: {block_summary}

Consider: relevance, uniqueness, potential utility.
Output only a decimal number."""
            
            response = self.llm.generate(
                prompt=prompt,
                max_tokens=5,
                temperature=0.1,
                repetition_penalty=1.3
            )
            
            score = self.llm.extract_number(response, default=0.5)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.debug(f"LLM block evaluation failed: {e}")
            return 0.5
    
    def compute_novelty_score(
        self,
        content: str,
        existing_contents: List[str]
    ) -> float:
        """
        Compute how novel/unique content is compared to existing content.
        Used during block formation decisions.
        
        Args:
            content: New content to evaluate
            existing_contents: List of existing content strings
        
        Returns:
            Novelty score (0-1, where 1 is completely novel)
        """
        if not existing_contents:
            return 1.0
        
        try:
            from embedding.singleton_embedder import get_text_embedder
            embedder = get_text_embedder()
            
            # Get embedding for new content
            new_embedding = embedder.embed_text(content)
            new_norm = np.linalg.norm(new_embedding)
            
            if new_norm == 0:
                return 0.5
            
            # Compare with existing content
            max_similarity = 0.0
            for existing in existing_contents:
                existing_embedding = embedder.embed_text(existing)
                existing_norm = np.linalg.norm(existing_embedding)
                
                if existing_norm > 0:
                    similarity = np.dot(new_embedding, existing_embedding) / (new_norm * existing_norm)
                    max_similarity = max(max_similarity, similarity)
            
            # Convert similarity to novelty (inverse relationship)
            novelty = 1.0 - max_similarity
            return max(0.0, novelty)
            
        except Exception as e:
            logger.debug(f"Novelty computation failed: {e}")
            # Fallback to simple word overlap
            new_words = set(content.lower().split())
            
            if not new_words:
                return 0.5
            
            max_overlap = 0.0
            for existing in existing_contents:
                existing_words = set(existing.lower().split())
                if existing_words:
                    overlap = len(new_words & existing_words) / len(new_words | existing_words)
                    max_overlap = max(max_overlap, overlap)
            
            return 1.0 - max_overlap
    
    def __repr__(self) -> str:
        """String representation"""
        llm_status = "available" if self.llm.is_available() else "unavailable"
        return f"SalienceModel(llm={llm_status})"
