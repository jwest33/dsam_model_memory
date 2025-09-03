"""
Memory Evaluation System
Evaluates if memories can answer a query or if tools are needed.
Includes time-based validity for different information types.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import re


class MemoryEvaluator:
    """Evaluates memory freshness and determines if tool calls are needed."""
    
    # Time-based validity periods for different information types (in hours)
    VALIDITY_PERIODS = {
        # Very short-lived (minutes to hours)
        'weather_current': 0.5,  # 30 minutes
        'weather_today': 3,      # 3 hours
        'weather_forecast': 24,  # 24 hours
        'stock_price': 0.25,     # 15 minutes
        'cryptocurrency': 0.083, # 5 minutes
        
        # Short-lived (hours to days)
        'news_breaking': 1,      # 1 hour
        'news_daily': 24,        # 24 hours
        'sports_score': 0.5,     # 30 minutes
        'traffic': 0.25,         # 15 minutes
        
        # Medium-lived (days to weeks)
        'news_general': 72,      # 3 days
        'events': 168,           # 1 week
        'product_price': 24,     # 24 hours
        'movie_showtimes': 24,   # 24 hours
        
        # Long-lived (weeks to months)
        'historical_fact': 8760, # 1 year
        'scientific_fact': 2190, # 3 months (for recent research)
        'definition': 8760,      # 1 year
        'tutorial': 730,         # 1 month
        
        # Default
        'default': 24            # 24 hours
    }
    
    @classmethod
    def classify_query_type(cls, query: str) -> str:
        """Classify the type of information being requested."""
        query_lower = query.lower()
        
        # Weather patterns
        if any(word in query_lower for word in ['weather', 'temperature', 'rain', 'snow', 'forecast']):
            if any(word in query_lower for word in ['now', 'current', 'right now']):
                return 'weather_current'
            elif any(word in query_lower for word in ['today', 'tonight']):
                return 'weather_today'
            else:
                return 'weather_forecast'
        
        # Financial patterns
        if any(word in query_lower for word in ['stock', 'share price', 'market']):
            return 'stock_price'
        if any(word in query_lower for word in ['bitcoin', 'ethereum', 'crypto']):
            return 'cryptocurrency'
        
        # News patterns
        if any(word in query_lower for word in ['breaking', 'just happened', 'latest']):
            return 'news_breaking'
        if any(word in query_lower for word in ['news', 'headlines']):
            if 'today' in query_lower:
                return 'news_daily'
            return 'news_general'
        
        # Sports
        if any(word in query_lower for word in ['score', 'game', 'match']):
            return 'sports_score'
        
        # Traffic
        if any(word in query_lower for word in ['traffic', 'commute', 'congestion']):
            return 'traffic'
        
        # Events and prices
        if any(word in query_lower for word in ['event', 'concert', 'show']):
            return 'events'
        if any(word in query_lower for word in ['price', 'cost', 'how much']):
            return 'product_price'
        if any(word in query_lower for word in ['movie', 'cinema', 'showtime']):
            return 'movie_showtimes'
        
        # Knowledge
        if any(word in query_lower for word in ['what is', 'define', 'meaning']):
            return 'definition'
        if any(word in query_lower for word in ['how to', 'tutorial', 'guide']):
            return 'tutorial'
        if any(word in query_lower for word in ['history', 'historical', 'when did']):
            return 'historical_fact'
        if any(word in query_lower for word in ['scientific', 'research', 'study']):
            return 'scientific_fact'
        
        return 'default'
    
    @classmethod
    def is_memory_valid(cls, memory_timestamp: str, query_type: str) -> Tuple[bool, str]:
        """
        Check if a memory is still valid based on its age and type.
        Returns (is_valid, reason)
        """
        try:
            # Parse the timestamp
            if isinstance(memory_timestamp, str):
                memory_time = datetime.fromisoformat(memory_timestamp.replace('Z', '+00:00'))
            else:
                memory_time = memory_timestamp
            
            # Get current time
            current_time = datetime.now(memory_time.tzinfo) if memory_time.tzinfo else datetime.now()
            
            # Calculate age in hours
            age_hours = (current_time - memory_time).total_seconds() / 3600
            
            # Get validity period
            validity_hours = cls.VALIDITY_PERIODS.get(query_type, cls.VALIDITY_PERIODS['default'])
            
            if age_hours <= validity_hours:
                return True, f"Memory is {age_hours:.1f} hours old, still valid for {query_type} (limit: {validity_hours} hours)"
            else:
                return False, f"Memory is {age_hours:.1f} hours old, too old for {query_type} (limit: {validity_hours} hours)"
                
        except Exception as e:
            return False, f"Could not parse timestamp: {e}"
    
    @classmethod
    def evaluate_memories(cls, query: str, memories: List[Dict]) -> Dict:
        """
        Evaluate if memories can answer the query or if tools are needed.
        Simply checks if ANY memories exist and their age.
        """
        query_type = cls.classify_query_type(query)
        
        valid_memories = []
        invalid_memories = []
        
        # Check ALL memories for temporal validity - no content filtering
        for memory in memories:
            timestamp = memory.get('when_ts') or memory.get('created_at')
            if timestamp:
                is_valid, reason = cls.is_memory_valid(timestamp, query_type)
                if is_valid:
                    valid_memories.append({**memory, 'validity_reason': reason})
                else:
                    invalid_memories.append({**memory, 'validity_reason': reason})
        
        # Determine if we need tools based on whether we have ANY valid memories
        time_sensitive_types = [
            'weather_current', 'weather_today', 'weather_forecast',
            'stock_price', 'cryptocurrency', 'news_breaking', 
            'news_daily', 'sports_score', 'traffic'
        ]
        
        needs_tool = len(valid_memories) == 0 and query_type in time_sensitive_types
        
        if needs_tool:
            reason = f"No recent memories found for {query_type}. Tool needed for fresh data."
        elif valid_memories:
            reason = f"Found {len(valid_memories)} recent memories that may contain {query_type} information."
        else:
            reason = f"Memories exist but may be outdated for {query_type}."
        
        return {
            'needs_tool': needs_tool,
            'reason': reason,
            'query_type': query_type,
            'valid_memories': valid_memories,
            'invalid_memories': invalid_memories
        }
    
    
    
    @classmethod
    def build_evaluation_prompt(cls, query: str, evaluation: Dict) -> str:
        """Build a prompt for the LLM to make the final decision."""
        prompt = f"""Based on the memory evaluation, determine if you should use a tool or answer from memory.

User Query: {query}
Query Type: {evaluation['query_type']}
Evaluation: {evaluation['reason']}

Valid Memories Found: {len(evaluation['valid_memories'])}
Invalid/Outdated Memories: {len(evaluation['invalid_memories'])}

DECISION RULES:
1. If valid memories exist and are recent enough for the query type, answer from memory
2. If memories exist but are outdated for time-sensitive queries (weather, news, stocks), use tools
3. For non-time-sensitive queries (definitions, tutorials), older memories may still be valid
4. When in doubt about freshness for current events, prefer using tools

Based on this evaluation, should you:
A) Answer from memory (memories are sufficient and valid)
B) Use web_search tool (memories are insufficient or outdated)

Respond with ONLY 'A' or 'B' followed by your action."""
        
        return prompt