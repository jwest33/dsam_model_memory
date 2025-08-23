"""
High-level Memory Agent interface

Provides a simple API for memory operations with the 5W1H framework.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid

from config import get_config
from models.event import Event, EventType, FiveW1H
from memory.memory_store import MemoryStore
from llm.llm_interface import LLMInterface
from llm.salience_model import SalienceModel

logger = logging.getLogger(__name__)

class MemoryAgent:
    """High-level interface for memory operations"""
    
    def __init__(self, config=None):
        """Initialize the memory agent"""
        self.config = config or get_config()
        
        # Core components
        self.memory_store = MemoryStore(self.config)
        self.llm = LLMInterface(self.config.llm)
        self.salience_model = SalienceModel(self.llm)
        
        # Episode management
        self.current_episode_id = None
        self.episode_start_time = None
        
        # Agent state
        self.current_goal = None
        self.operation_count = 0
        
        logger.info("Memory Agent initialized")
    
    def remember(
        self,
        who: str,
        what: str,
        where: Optional[str] = None,
        why: Optional[str] = None,
        how: Optional[str] = None,
        event_type: str = "action",
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> Tuple[bool, str, Event]:
        """
        Store a memory using 5W1H structure
        
        Args:
            who: Agent/actor
            what: Action or content
            where: Context or location
            why: Intent or trigger
            how: Mechanism used
            event_type: Type of event (action/observation/user_input/system_event)
            tags: Optional tags for categorization
            **kwargs: Additional metadata
        
        Returns:
            (success, message, event)
        """
        # Auto-manage episodes
        self._manage_episode()
        
        # Create event
        event = Event(
            five_w1h=FiveW1H(
                who=who,
                what=what,
                when=datetime.utcnow().isoformat() + "Z",
                where=where or f"session:{self.current_episode_id}",
                why=why or self.current_goal or "unknown",
                how=how or "direct_input"
            ),
            event_type=EventType(event_type),
            episode_id=self.current_episode_id,
            tags=tags or [],
            **kwargs
        )
        
        # Compute salience
        existing_memories = self.memory_store.processed_memories[-10:]  # Recent memories
        event.salience = self.salience_model.compute_salience(
            event=event,
            goal=self.current_goal,
            existing_memories=existing_memories
        )
        
        # Store in memory
        success, message = self.memory_store.store_event(event)
        
        self.operation_count += 1
        
        logger.info(f"Stored memory: {event.id[:8]} - {message}")
        
        return success, message, event
    
    def recall(
        self,
        query: Optional[Dict[str, str]] = None,
        who: Optional[str] = None,
        what: Optional[str] = None,
        where: Optional[str] = None,
        why: Optional[str] = None,
        how: Optional[str] = None,
        k: int = 5,
        include_raw: bool = False
    ) -> List[Tuple[Event, float]]:
        """
        Recall memories matching a query
        
        Args:
            query: Direct 5W1H query dict
            who/what/where/why/how: Individual query components
            k: Number of results
            include_raw: Include raw memories in search
        
        Returns:
            List of (event, similarity_score) tuples
        """
        # Build query from parameters
        if query is None:
            query = {}
            if who: query["who"] = who
            if what: query["what"] = what
            if where: query["where"] = where
            if why: query["why"] = why
            if how: query["how"] = how
        
        # Retrieve from memory store
        results = self.memory_store.retrieve(
            query=query,
            k=k,
            include_raw=include_raw
        )
        
        logger.info(f"Recalled {len(results)} memories for query: {query}")
        
        return results
    
    def observe(
        self,
        what: str,
        who: Optional[str] = None,
        where: Optional[str] = None,
        action_event_id: Optional[str] = None,
        **kwargs
    ) -> Tuple[bool, str, Event]:
        """
        Record an observation (result of an action)
        
        Args:
            what: Observation content
            who: Observer (defaults to "system")
            where: Location of observation
            action_event_id: ID of action that caused this observation
            **kwargs: Additional metadata
        
        Returns:
            (success, message, event)
        """
        # Find episode from action if provided
        episode_id = self.current_episode_id
        why = "observation"
        
        if action_event_id:
            action_event = self._find_event(action_event_id)
            if action_event:
                episode_id = action_event.episode_id
                why = f"in response to {action_event.five_w1h.what[:50]}"
        
        return self.remember(
            who=who or "system",
            what=what,
            where=where,
            why=why,
            how="observation",
            event_type="observation",
            episode_id=episode_id,
            **kwargs
        )
    
    def chain_events(
        self,
        action: Event,
        observation: Event
    ) -> bool:
        """
        Explicitly link an action with its observation
        
        Args:
            action: Action event
            observation: Observation event
        
        Returns:
            Success status
        """
        # Ensure same episode
        if action.episode_id != observation.episode_id:
            observation.episode_id = action.episode_id
            
            # Update episode map
            if observation.id in self.memory_store.episode_map.get(observation.episode_id, []):
                self.memory_store.episode_map[observation.episode_id].remove(observation.id)
            
            if action.episode_id not in self.memory_store.episode_map:
                self.memory_store.episode_map[action.episode_id] = []
            
            self.memory_store.episode_map[action.episode_id].append(observation.id)
        
        # Update observation metadata
        observation.metadata["caused_by"] = action.id
        action.metadata["resulted_in"] = observation.id
        
        # Analyze causality if LLM available
        if self.llm.is_available():
            causality = self.llm.analyze_causality(
                action=action.five_w1h.what,
                observation=observation.five_w1h.what
            )
            
            action.metadata["causality"] = causality
            observation.metadata["causality"] = causality
        
        logger.info(f"Chained events: {action.id[:8]} -> {observation.id[:8]}")
        
        return True
    
    def get_episode(self, episode_id: Optional[str] = None) -> List[Event]:
        """
        Get all events in an episode
        
        Args:
            episode_id: Episode to retrieve (defaults to current)
        
        Returns:
            List of events in chronological order
        """
        episode_id = episode_id or self.current_episode_id
        
        if not episode_id:
            return []
        
        return self.memory_store.retrieve_episode(episode_id)
    
    def get_causal_chain(
        self,
        event_id: str,
        max_depth: int = 5
    ) -> List[Tuple[Event, Event]]:
        """
        Get causal chain starting from an event
        
        Args:
            event_id: Starting event ID
            max_depth: Maximum chain length
        
        Returns:
            List of (action, observation) pairs
        """
        return self.memory_store.get_causal_chain(event_id, max_depth)
    
    def complete_memory(
        self,
        partial: Dict[str, str],
        context: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Complete missing 5W1H slots using LLM
        
        Args:
            partial: Partial 5W1H dictionary
            context: Additional context
        
        Returns:
            Completed 5W1H dictionary
        """
        if not self.llm.is_available():
            logger.warning("LLM not available for memory completion")
            return partial
        
        return self.llm.complete_5w1h(partial, context)
    
    def summarize_episode(
        self,
        episode_id: Optional[str] = None
    ) -> str:
        """
        Generate a summary of an episode
        
        Args:
            episode_id: Episode to summarize (defaults to current)
        
        Returns:
            Text summary
        """
        events = self.get_episode(episode_id)
        
        if not events:
            return "No events in episode."
        
        if self.llm.is_available():
            # Convert events to simple format for LLM
            event_data = [
                {
                    "who": e.five_w1h.who,
                    "what": e.five_w1h.what,
                    "when": e.five_w1h.when
                }
                for e in events
            ]
            
            return self.llm.summarize_episode(event_data)
        else:
            # Simple summary
            return (
                f"Episode {episode_id or self.current_episode_id} contains {len(events)} events. "
                f"First: {events[0].five_w1h.what[:50]}... "
                f"Last: {events[-1].five_w1h.what[:50]}..."
            )
    
    def set_goal(self, goal: str):
        """Set the current goal for salience computation"""
        self.current_goal = goal
        logger.info(f"Set goal: {goal}")
    
    def start_episode(self, episode_id: Optional[str] = None):
        """Start a new episode"""
        self.current_episode_id = episode_id or f"e{uuid.uuid4().hex[:8]}"
        self.episode_start_time = datetime.utcnow()
        logger.info(f"Started episode: {self.current_episode_id}")
    
    def end_episode(self) -> str:
        """End the current episode and return its ID"""
        episode_id = self.current_episode_id
        self.current_episode_id = None
        self.episode_start_time = None
        logger.info(f"Ended episode: {episode_id}")
        return episode_id
    
    def _manage_episode(self):
        """Auto-manage episode lifecycle"""
        if not self.config.agent.auto_link_episodes:
            return
        
        # Start new episode if needed
        if not self.current_episode_id:
            self.start_episode()
            return
        
        # Check for episode timeout
        if self.episode_start_time:
            elapsed = (datetime.utcnow() - self.episode_start_time).total_seconds()
            if elapsed > self.config.agent.episode_timeout:
                self.end_episode()
                self.start_episode()
    
    def _find_event(self, event_id: str) -> Optional[Event]:
        """Find an event by ID"""
        return self.memory_store._find_event_by_id(event_id)
    
    def save(self):
        """Save memory state to disk"""
        self.memory_store.save()
        logger.info("Saved memory state")
    
    def load(self):
        """Load memory state from disk"""
        self.memory_store._load_from_disk()
        logger.info("Loaded memory state")
    
    def clear(self):
        """Clear all memories"""
        self.memory_store.clear()
        self.current_episode_id = None
        self.episode_start_time = None
        self.operation_count = 0
        logger.info("Cleared all memories")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent and memory statistics"""
        stats = self.memory_store.get_statistics()
        
        stats.update({
            'current_episode': self.current_episode_id,
            'current_goal': self.current_goal,
            'operation_count': self.operation_count,
            'llm_available': self.llm.is_available()
        })
        
        if self.episode_start_time:
            elapsed = (datetime.utcnow() - self.episode_start_time).total_seconds()
            stats['episode_duration'] = elapsed
        
        return stats
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - auto-save"""
        if self.config.storage.auto_save:
            self.save()
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"MemoryAgent(memories={len(self.memory_store)}, "
            f"episode={self.current_episode_id}, "
            f"goal={self.current_goal})"
        )