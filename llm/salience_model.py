"""
Salience computation model using LLM with heuristic fallback

Determines the importance of memories for storage and retrieval.
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np

from config import get_config
from llm.llm_interface import LLMInterface
from models.event import Event

logger = logging.getLogger(__name__)

class SalienceModel:
    """Model for computing memory salience (importance)"""
    
    def __init__(self, llm_interface: Optional[LLMInterface] = None):
        """Initialize salience model"""
        self.config = get_config()
        self.llm = llm_interface or LLMInterface(self.config.llm)
        self.use_llm = self.config.llm.use_llm_salience
    
    def compute_salience(
        self,
        event: Event,
        goal: Optional[str] = None,
        existing_memories: Optional[List[Event]] = None
    ) -> float:
        """
        Compute salience score for an event
        
        Args:
            event: Event to score
            goal: Current goal or context
            existing_memories: Related existing memories for novelty computation
        
        Returns:
            Salience score (0-1)
        """
        # Compute novelty and overlap
        novelty = self._compute_novelty(event, existing_memories)
        overlap = self._compute_overlap(event, existing_memories)
        
        # Try LLM-based scoring if available
        if self.use_llm and self.llm.is_available():
            try:
                salience = self._llm_salience(
                    event=event,
                    goal=goal,
                    novelty=novelty,
                    overlap=overlap
                )
                
                # Apply exponential moving average if event has previous salience
                if event.salience != 0.5:  # Not default
                    alpha = self.config.memory.salience_ema_alpha
                    salience = (1 - alpha) * event.salience + alpha * salience
                
                return salience
                
            except Exception as e:
                logger.warning(f"LLM salience computation failed: {e}, falling back to heuristic")
        
        # Fallback to heuristic
        return self._heuristic_salience(event, goal, novelty, overlap)
    
    def _llm_salience(
        self,
        event: Event,
        goal: Optional[str],
        novelty: float,
        overlap: float
    ) -> float:
        """Compute salience using LLM"""
        # Build prompt
        prompt = self.config.llm.salience_prompt_template.format(
            goal=goal or "General knowledge acquisition",
            query=event.five_w1h.why or "Unknown intent",
            observation=event.five_w1h.what,
            novelty=novelty,
            overlap=overlap
        )
        
        # Get LLM response
        response = self.llm.generate(
            prompt=prompt,
            max_tokens=20,
            temperature=0.3  # Lower temperature for more consistent scoring
        )
        
        # Extract score
        score = self.llm.extract_number(response, default=0.5)
        
        logger.debug(f"LLM salience for '{event.five_w1h.what[:50]}...': {score:.2f}")
        
        return score
    
    def _heuristic_salience(
        self,
        event: Event,
        goal: Optional[str],
        novelty: float,
        overlap: float
    ) -> float:
        """Compute salience using heuristics"""
        # Base factors
        factors = {
            'novelty': novelty * 0.4,
            'low_overlap': (1 - overlap) * 0.3,
            'length': self._length_factor(event) * 0.2,
            'keywords': self._keyword_boost(event, goal) * 0.1
        }
        
        # Type-specific boosts
        if event.event_type.value == "observation":
            factors['type_boost'] = 0.1  # Observations are generally important
        elif event.event_type.value == "user_input":
            factors['type_boost'] = 0.15  # User input is very important
        else:
            factors['type_boost'] = 0.05
        
        # Confidence factor
        factors['confidence'] = event.confidence * 0.1
        
        # Sum factors
        salience = sum(factors.values())
        
        # Clamp to [0, 1]
        salience = max(0.0, min(1.0, salience))
        
        logger.debug(f"Heuristic salience for '{event.five_w1h.what[:50]}...': {salience:.2f}")
        
        return salience
    
    def _compute_novelty(
        self,
        event: Event,
        existing_memories: Optional[List[Event]]
    ) -> float:
        """Compute novelty score (1 = completely new, 0 = duplicate)"""
        if not existing_memories:
            return 1.0
        
        # Compare with existing memories
        max_similarity = 0.0
        
        for memory in existing_memories:
            # Simple text similarity based on 'what' field
            similarity = self._text_similarity(
                event.five_w1h.what,
                memory.five_w1h.what
            )
            max_similarity = max(max_similarity, similarity)
        
        novelty = 1.0 - max_similarity
        return novelty
    
    def _compute_overlap(
        self,
        event: Event,
        existing_memories: Optional[List[Event]]
    ) -> float:
        """Compute overlap with existing memories (semantic redundancy)"""
        if not existing_memories:
            return 0.0
        
        # Take top-k most similar memories
        k = min(5, len(existing_memories))
        
        similarities = []
        for memory in existing_memories:
            sim = self._text_similarity(
                event.five_w1h.what,
                memory.five_w1h.what
            )
            similarities.append(sim)
        
        # Average of top-k similarities
        top_k_sims = sorted(similarities, reverse=True)[:k]
        overlap = np.mean(top_k_sims) if top_k_sims else 0.0
        
        return overlap
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity using word overlap"""
        if not text1 or not text2:
            return 0.0
        
        # Tokenize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _length_factor(self, event: Event) -> float:
        """Score based on content length (longer = more information)"""
        text = event.full_content or event.five_w1h.what
        length = len(text)
        
        # Sigmoid-like scoring
        if length < 50:
            return 0.2
        elif length < 200:
            return 0.5
        elif length < 500:
            return 0.8
        else:
            return 1.0
    
    def _keyword_boost(self, event: Event, goal: Optional[str]) -> float:
        """Boost score if event contains goal-related keywords"""
        if not goal:
            return 0.0
        
        text = event.five_w1h.what.lower()
        goal_words = set(goal.lower().split())
        
        # Check for keyword matches
        matches = sum(1 for word in goal_words if word in text)
        
        if matches == 0:
            return 0.0
        elif matches == 1:
            return 0.5
        else:
            return 1.0
    
    def batch_compute_salience(
        self,
        events: List[Event],
        goal: Optional[str] = None
    ) -> List[float]:
        """Compute salience for multiple events"""
        saliences = []
        
        for i, event in enumerate(events):
            # Use previous events as context for novelty
            existing = events[:i] if i > 0 else None
            
            salience = self.compute_salience(
                event=event,
                goal=goal,
                existing_memories=existing
            )
            saliences.append(salience)
        
        return saliences
    
    def update_salience_ema(
        self,
        current: float,
        new: float,
        alpha: Optional[float] = None
    ) -> float:
        """Update salience using exponential moving average"""
        if alpha is None:
            alpha = self.config.memory.salience_ema_alpha
        
        return (1 - alpha) * current + alpha * new
    
    def __repr__(self) -> str:
        """String representation"""
        mode = "LLM" if self.use_llm and self.llm.is_available() else "Heuristic"
        return f"SalienceModel(mode={mode})"