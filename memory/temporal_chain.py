"""
Temporal Chain Manager for Event Relationships

This module manages temporal relationships between events, tracking how events
update, correct, or supersede each other over time.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import json

from models.event import Event
from models.merged_event import EventRelationship

logger = logging.getLogger(__name__)


class ChainType(Enum):
    """Types of temporal chains"""
    UPDATE_CHAIN = "update_chain"      # Progressive updates to same concept
    CORRECTION_CHAIN = "correction"    # Error corrections
    ITERATION_CHAIN = "iteration"      # Iterative improvements
    CONVERSATION_CHAIN = "conversation"  # Back-and-forth dialogue
    WORKFLOW_CHAIN = "workflow"        # Step-by-step process


class TemporalChain:
    """
    Manages temporal relationships between events, tracking how information
    evolves, gets corrected, or superseded over time.
    """
    
    def __init__(self):
        """Initialize the temporal chain manager"""
        self.chains: Dict[str, List[str]] = {}  # chain_id -> ordered list of event IDs
        self.chain_types: Dict[str, ChainType] = {}  # chain_id -> chain type
        self.event_to_chain: Dict[str, str] = {}  # event_id -> chain_id
        self.latest_state: Dict[str, Dict] = {}  # chain_id -> current state
        self.chain_metadata: Dict[str, Dict] = {}  # chain_id -> metadata
        
        # Relationship tracking
        self.supersedes_map: Dict[str, str] = {}  # event_id -> superseded_by
        self.corrections_map: Dict[str, str] = {}  # event_id -> corrected_by
        self.continues_map: Dict[str, str] = {}  # event_id -> continues_from
        
    def add_event(self, event: Event, chain_context: Optional[Dict] = None) -> str:
        """
        Add an event to the appropriate temporal chain.
        
        Args:
            event: The event to add
            chain_context: Optional context for chain identification
            
        Returns:
            The chain ID the event was added to
        """
        # Identify or create chain
        chain_id = self._identify_chain(event, chain_context)
        
        if chain_id not in self.chains:
            self.chains[chain_id] = []
            self.chain_types[chain_id] = self._determine_chain_type(event, chain_context)
            self.chain_metadata[chain_id] = {
                'created_at': datetime.utcnow().isoformat(),
                'initial_event': event.id,
                'participants': set(),
                'topics': set()
            }
        
        # Determine relationship to previous events
        relationship = self._determine_relationship(event, self.chains[chain_id])
        
        # Update relationship maps
        self._update_relationship_maps(event, self.chains[chain_id], relationship)
        
        # Add event to chain
        self.chains[chain_id].append(event.id)
        self.event_to_chain[event.id] = chain_id
        
        # Update chain metadata
        self._update_chain_metadata(chain_id, event)
        
        # Update latest state
        self._update_latest_state(chain_id, event)
        
        logger.info(f"Added event {event.id} to chain {chain_id} with relationship {relationship.value}")
        
        return chain_id
    
    def _identify_chain(self, event: Event, context: Optional[Dict]) -> str:
        """
        Identify which chain this event belongs to.
        
        Uses various heuristics including episode ID, conversation context,
        and semantic similarity to existing chains.
        """
        # Priority 1: Explicit chain ID in context
        if context and context.get('chain_id'):
            return context['chain_id']
        
        # Priority 2: Episode-based chaining
        if event.episode_id:
            # Check if this episode already has a chain
            for existing_chain_id, metadata in self.chain_metadata.items():
                if metadata.get('episode_id') == event.episode_id:
                    return existing_chain_id
            
            # Create new chain for this episode
            return f"episode_{event.episode_id}"
        
        # Priority 3: Conversation continuation (for chat interfaces)
        if event.five_w1h.where == "web_chat":
            # Look for recent chat chains
            for chain_id, events in self.chains.items():
                if self.chain_types.get(chain_id) == ChainType.CONVERSATION_CHAIN:
                    if events:
                        # Check if this is a continuation (within 5 minutes)
                        last_event_time = self.latest_state.get(chain_id, {}).get('timestamp')
                        if last_event_time:
                            time_diff = event.created_at - datetime.fromisoformat(last_event_time)
                            if time_diff < timedelta(minutes=5):
                                return chain_id
            
            # Create new conversation chain
            return f"conversation_{event.created_at.strftime('%Y%m%d_%H%M%S')}"
        
        # Priority 4: Topic-based chaining
        if event.five_w1h.what:
            topic_key = self._extract_topic_key(event.five_w1h.what)
            
            # Check existing chains for similar topics
            for chain_id, metadata in self.chain_metadata.items():
                if topic_key in metadata.get('topics', set()):
                    # Check temporal proximity
                    last_update = metadata.get('last_updated')
                    if last_update:
                        time_diff = event.created_at - datetime.fromisoformat(last_update)
                        if time_diff < timedelta(hours=1):
                            return chain_id
            
            # Create new topic-based chain
            return f"topic_{topic_key}_{event.created_at.strftime('%Y%m%d_%H%M')}"
        
        # Default: Create unique chain for this event
        return f"unique_{event.id}"
    
    def _determine_chain_type(self, event: Event, context: Optional[Dict]) -> ChainType:
        """Determine the type of chain based on the event and context"""
        # Explicit type in context
        if context and context.get('chain_type'):
            return ChainType(context['chain_type'])
        
        # Chat interactions
        if event.five_w1h.where == "web_chat":
            return ChainType.CONVERSATION_CHAIN
        
        # Look for workflow indicators
        if event.five_w1h.how and any(word in event.five_w1h.how.lower() 
                                      for word in ['step', 'phase', 'stage']):
            return ChainType.WORKFLOW_CHAIN
        
        # Look for correction indicators
        if event.five_w1h.what and any(word in event.five_w1h.what.lower() 
                                       for word in ['fix', 'correct', 'repair']):
            return ChainType.CORRECTION_CHAIN
        
        # Look for iteration indicators
        if event.five_w1h.what and any(word in event.five_w1h.what.lower() 
                                       for word in ['improve', 'optimize', 'refine']):
            return ChainType.ITERATION_CHAIN
        
        # Default to update chain
        return ChainType.UPDATE_CHAIN
    
    def _determine_relationship(self, event: Event, chain_events: List[str]) -> EventRelationship:
        """Determine how this event relates to previous events in the chain"""
        if not chain_events:
            return EventRelationship.INITIAL
        
        # Analyze event content for relationship indicators
        what_lower = event.five_w1h.what.lower() if event.five_w1h.what else ""
        why_lower = event.five_w1h.why.lower() if event.five_w1h.why else ""
        
        # Check for correction
        if any(word in what_lower for word in ['fix', 'correct', 'repair', 'amend']):
            return EventRelationship.CORRECTION
        
        # Check for continuation
        if any(word in what_lower for word in ['continue', 'resume', 'proceed']):
            return EventRelationship.CONTINUATION
        
        # Check for superseding
        if any(phrase in what_lower for phrase in ['instead of', 'replace', 'new approach']):
            return EventRelationship.SUPERSEDES
        
        # Check for update
        if any(word in what_lower for word in ['update', 'modify', 'change']):
            return EventRelationship.UPDATE
        
        # Default to variation
        return EventRelationship.VARIATION
    
    def _update_relationship_maps(self, event: Event, chain_events: List[str], 
                                 relationship: EventRelationship):
        """Update the relationship tracking maps"""
        if not chain_events:
            return
        
        last_event_id = chain_events[-1] if chain_events else None
        
        if relationship == EventRelationship.SUPERSEDES and last_event_id:
            self.supersedes_map[last_event_id] = event.id
            
        elif relationship == EventRelationship.CORRECTION and last_event_id:
            self.corrections_map[last_event_id] = event.id
            
        elif relationship == EventRelationship.CONTINUATION and last_event_id:
            self.continues_map[event.id] = last_event_id
    
    def _update_chain_metadata(self, chain_id: str, event: Event):
        """Update metadata for a chain"""
        metadata = self.chain_metadata[chain_id]
        
        # Update timestamp
        metadata['last_updated'] = event.created_at.isoformat()
        
        # Add participant
        if event.five_w1h.who:
            if 'participants' not in metadata:
                metadata['participants'] = set()
            metadata['participants'].add(event.five_w1h.who)
        
        # Add topics
        if event.five_w1h.what:
            topic_key = self._extract_topic_key(event.five_w1h.what)
            if 'topics' not in metadata:
                metadata['topics'] = set()
            metadata['topics'].add(topic_key)
        
        # Track episode if present
        if event.episode_id and 'episode_id' not in metadata:
            metadata['episode_id'] = event.episode_id
        
        # Update event count
        metadata['event_count'] = len(self.chains[chain_id])
    
    def _update_latest_state(self, chain_id: str, event: Event):
        """Update the latest state for a chain"""
        self.latest_state[chain_id] = {
            'event_id': event.id,
            'timestamp': event.created_at.isoformat(),
            'who': event.five_w1h.who,
            'what': event.five_w1h.what,
            'when': event.five_w1h.when,
            'where': event.five_w1h.where,
            'why': event.five_w1h.why,
            'how': event.five_w1h.how
        }
    
    def _extract_topic_key(self, text: str) -> str:
        """Extract a topic key from text for chain identification"""
        # Simple approach: use first few significant words
        # This can be enhanced with NLP
        words = text.lower().split()
        significant_words = [w for w in words if len(w) > 3][:3]
        return "_".join(significant_words)
    
    def get_chain_for_event(self, event_id: str) -> Optional[List[str]]:
        """Get the complete chain containing this event"""
        chain_id = self.event_to_chain.get(event_id)
        if chain_id:
            return self.chains.get(chain_id, [])
        return None
    
    def get_chain_context(self, chain_id: str, max_events: int = 10) -> List[Dict]:
        """
        Get the context of a chain for LLM consumption.
        
        Args:
            chain_id: The chain to get context for
            max_events: Maximum number of recent events to include
            
        Returns:
            List of event summaries in chronological order
        """
        if chain_id not in self.chains:
            return []
        
        events = self.chains[chain_id]
        recent_events = events[-max_events:] if len(events) > max_events else events
        
        context = []
        for i, event_id in enumerate(recent_events):
            event_context = {
                'event_id': event_id,
                'position': i + 1,
                'total': len(recent_events)
            }
            
            # Add relationship information
            if event_id in self.supersedes_map:
                event_context['superseded_by'] = self.supersedes_map[event_id]
            if event_id in self.corrections_map:
                event_context['corrected_by'] = self.corrections_map[event_id]
            if event_id in self.continues_map:
                event_context['continues_from'] = self.continues_map[event_id]
            
            # Check if this event supersedes or corrects others
            for other_id, superseded_by in self.supersedes_map.items():
                if superseded_by == event_id:
                    event_context['supersedes'] = other_id
                    break
            
            for other_id, corrected_by in self.corrections_map.items():
                if corrected_by == event_id:
                    event_context['corrects'] = other_id
                    break
            
            context.append(event_context)
        
        return context
    
    def get_latest_valid_state(self, chain_id: str) -> Optional[Dict]:
        """
        Get the latest valid state of a chain, accounting for corrections and supersessions.
        
        Args:
            chain_id: The chain to get the latest state for
            
        Returns:
            The latest valid state, or None if chain doesn't exist
        """
        if chain_id not in self.chains:
            return None
        
        events = self.chains[chain_id]
        
        # Work backwards to find the latest non-superseded, non-corrected event
        for event_id in reversed(events):
            if event_id not in self.supersedes_map and event_id not in self.corrections_map:
                # This event hasn't been superseded or corrected
                return self.latest_state.get(chain_id)
        
        # All events have been superseded or corrected, return the very latest
        return self.latest_state.get(chain_id)
    
    def merge_chains(self, chain_id1: str, chain_id2: str) -> str:
        """
        Merge two chains into one.
        
        Args:
            chain_id1: First chain ID
            chain_id2: Second chain ID
            
        Returns:
            The ID of the merged chain
        """
        if chain_id1 not in self.chains or chain_id2 not in self.chains:
            logger.warning(f"Cannot merge chains {chain_id1} and {chain_id2}: one or both don't exist")
            return chain_id1 if chain_id1 in self.chains else chain_id2
        
        # Merge events (chronologically)
        events1 = self.chains[chain_id1]
        events2 = self.chains[chain_id2]
        
        # Create merged list maintaining chronological order
        # This is simplified - in practice, you'd want to sort by actual timestamps
        merged_events = events1 + events2
        
        # Update chain
        self.chains[chain_id1] = merged_events
        
        # Update event-to-chain mapping
        for event_id in events2:
            self.event_to_chain[event_id] = chain_id1
        
        # Merge metadata
        metadata1 = self.chain_metadata.get(chain_id1, {})
        metadata2 = self.chain_metadata.get(chain_id2, {})
        
        if 'participants' in metadata1 and 'participants' in metadata2:
            metadata1['participants'].update(metadata2['participants'])
        if 'topics' in metadata1 and 'topics' in metadata2:
            metadata1['topics'].update(metadata2['topics'])
        
        metadata1['event_count'] = len(merged_events)
        metadata1['merged_from'] = chain_id2
        metadata1['merged_at'] = datetime.utcnow().isoformat()
        
        # Remove chain2
        del self.chains[chain_id2]
        if chain_id2 in self.chain_types:
            del self.chain_types[chain_id2]
        if chain_id2 in self.chain_metadata:
            del self.chain_metadata[chain_id2]
        if chain_id2 in self.latest_state:
            del self.latest_state[chain_id2]
        
        logger.info(f"Merged chain {chain_id2} into {chain_id1}")
        
        return chain_id1
    
    def to_dict(self) -> Dict:
        """Serialize the temporal chain manager to a dictionary"""
        return {
            'chains': self.chains,
            'chain_types': {k: v.value for k, v in self.chain_types.items()},
            'event_to_chain': self.event_to_chain,
            'latest_state': self.latest_state,
            'chain_metadata': {
                k: {
                    **v,
                    'participants': list(v.get('participants', set())),
                    'topics': list(v.get('topics', set()))
                }
                for k, v in self.chain_metadata.items()
            },
            'supersedes_map': self.supersedes_map,
            'corrections_map': self.corrections_map,
            'continues_map': self.continues_map
        }
    
    def from_dict(self, data: Dict):
        """Load the temporal chain manager from a dictionary"""
        self.chains = data.get('chains', {})
        self.chain_types = {
            k: ChainType(v) for k, v in data.get('chain_types', {}).items()
        }
        self.event_to_chain = data.get('event_to_chain', {})
        self.latest_state = data.get('latest_state', {})
        
        # Restore metadata with sets
        self.chain_metadata = {}
        for k, v in data.get('chain_metadata', {}).items():
            self.chain_metadata[k] = {
                **v,
                'participants': set(v.get('participants', [])),
                'topics': set(v.get('topics', []))
            }
        
        self.supersedes_map = data.get('supersedes_map', {})
        self.corrections_map = data.get('corrections_map', {})
        self.continues_map = data.get('continues_map', {})