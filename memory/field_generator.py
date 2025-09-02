"""
Intelligent field generation for 5W1H memory fields.
Generates contextual 'why' fields via LLM and supports multiple 'how' mechanisms.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class MechanismType(Enum):
    """Types of mechanisms for the 'how' field"""
    # User input mechanisms
    CHAT_INTERFACE = "chat_interface"
    VOICE_INPUT = "voice_input"
    API_CALL = "api_call"
    CLI_COMMAND = "cli_command"
    WEB_FORM = "web_form"
    
    # Assistant/system mechanisms
    LLM_GENERATION = "llm_generation"
    TOOL_USE = "tool_use"
    FUNCTION_CALL = "function_call"
    WEB_SEARCH = "web_search"
    DATABASE_QUERY = "database_query"
    CALCULATION = "calculation"
    FILE_OPERATION = "file_operation"
    API_REQUEST = "api_request"
    
    # Observation mechanisms
    SYSTEM_OBSERVATION = "system_observation"
    EVENT_TRIGGER = "event_trigger"
    SCHEDULED_TASK = "scheduled_task"
    ERROR_DETECTION = "error_detection"
    
    # Analysis mechanisms
    PATTERN_RECOGNITION = "pattern_recognition"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"


class FieldGenerator:
    """Generates intelligent 5W1H fields for memories"""
    
    def __init__(self, llm_client=None):
        """
        Initialize the field generator.
        
        Args:
            llm_client: LLM client for generating 'why' fields
        """
        self.llm_client = llm_client
        self.recent_context = []  # Store recent messages for context
        self.max_context_size = 5  # Keep last 5 messages for context
        
    def add_to_context(self, message: str, who: str):
        """
        Add a message to the recent context.
        
        Args:
            message: The message content
            who: Who sent the message (User/Assistant)
        """
        self.recent_context.append({
            'who': who,
            'what': message,
            'when': datetime.now(timezone.utc).isoformat()
        })
        
        # Keep only the most recent messages
        if len(self.recent_context) > self.max_context_size:
            self.recent_context.pop(0)
    
    def generate_why_field(self, current_message: str, who: str, 
                          message_type: str = "query") -> str:
        """
        Generate a contextual 'why' field using LLM analysis.
        
        Args:
            current_message: The current message to analyze
            who: Who is creating this memory
            message_type: Type of message (query, response, observation)
            
        Returns:
            A short, contextual 'why' reason
        """
        if not self.llm_client:
            # Fallback to simple heuristics if no LLM available
            return self._generate_why_heuristic(current_message, who, message_type)
        
        try:
            # Build context from recent messages
            context_str = self._build_context_string()
            
            # Create prompt for LLM
            prompt = f"""Analyze this message and recent context to determine the user's intent or purpose.
Generate a SHORT (3-7 words) 'why' field that captures the core purpose/intent.

Recent context:
{context_str}

Current message from {who}:
"{current_message}"

Message type: {message_type}

Examples of good 'why' fields:
- "seeking API implementation guidance"
- "troubleshooting authentication error"
- "planning vacation itinerary"
- "understanding machine learning concepts"
- "configuring development environment"
- "analyzing performance metrics"
- "requesting data visualization"

Generate a concise 'why' field for this message. Focus on the underlying intent or goal.
Return ONLY the why field text, nothing else:"""

            # Call LLM for generation
            response = self.llm_client.generate(
                prompt, 
                max_tokens=20,
                temperature=0.3,  # Lower temperature for more consistent output
                stop=["\n", "."]  # Stop at newline or period
            )
            
            # Clean and validate response
            why_field = response.strip().strip('"').strip("'").strip()
            
            # Remove any leading/trailing punctuation except hyphens
            why_field = why_field.strip('.,;:!?')
            
            # Ensure it's not too long and not empty
            if not why_field:
                return self._generate_why_heuristic(current_message, who, message_type)
            
            if len(why_field.split()) > 10:
                why_field = ' '.join(why_field.split()[:7])
            
            return why_field
            
        except Exception as e:
            logger.warning(f"LLM generation failed, using heuristic: {e}")
            return self._generate_why_heuristic(current_message, who, message_type)
    
    def _generate_why_heuristic(self, message: str, who: str, 
                                message_type: str) -> str:
        """
        Generate 'why' field using heuristics when LLM is unavailable.
        
        Args:
            message: The message to analyze
            who: Who is creating this memory
            message_type: Type of message
            
        Returns:
            A heuristic-based 'why' field
        """
        message_lower = message.lower()
        
        # Detect common patterns and intents
        if who.lower() in ['user', 'human']:
            # User intents
            if any(word in message_lower for word in ['how do i', 'how to', 'how can']):
                return "seeking implementation guidance"
            elif any(word in message_lower for word in ['what is', 'what are', 'what does']):
                return "requesting explanation"
            elif any(word in message_lower for word in ['why is', 'why does', 'why do']):
                return "understanding reasoning"
            elif any(word in message_lower for word in ['error', 'bug', 'issue', 'problem']):
                return "troubleshooting issue"
            elif any(word in message_lower for word in ['create', 'build', 'make', 'implement']):
                return "building new feature"
            elif any(word in message_lower for word in ['fix', 'repair', 'solve', 'resolve']):
                return "fixing problem"
            elif any(word in message_lower for word in ['optimize', 'improve', 'enhance', 'better']):
                return "improving performance"
            elif any(word in message_lower for word in ['analyze', 'examine', 'investigate']):
                return "analyzing data"
            elif any(word in message_lower for word in ['plan', 'schedule', 'organize']):
                return "planning activity"
            elif '?' in message:
                return "asking question"
            else:
                return "providing information"
                
        else:  # Assistant/System
            if message_type == "response":
                # Analyze what kind of response
                if any(word in message_lower for word in ['here is', 'here are', 'this is']):
                    return "providing solution"
                elif any(word in message_lower for word in ['try', 'you can', 'you should']):
                    return "suggesting approach"
                elif any(word in message_lower for word in ['error', 'warning', 'issue']):
                    return "identifying problem"
                elif any(word in message_lower for word in ['first', 'then', 'finally', 'step']):
                    return "explaining process"
                else:
                    return "assisting user"
            elif message_type == "observation":
                return "system observation"
            else:
                return "processing request"
    
    def _build_context_string(self) -> str:
        """Build a string representation of recent context."""
        if not self.recent_context:
            return "No recent context"
        
        lines = []
        for ctx in self.recent_context[-3:]:  # Last 3 messages for context
            who = ctx['who']
            what = ctx['what'][:100]  # Truncate long messages
            lines.append(f"{who}: {what}")
        
        return "\n".join(lines)
    
    def generate_how_field(self, mechanism: MechanismType, 
                          details: Optional[Dict] = None) -> str:
        """
        Generate a descriptive 'how' field based on mechanism type.
        
        Args:
            mechanism: The mechanism type
            details: Optional details about the mechanism
            
        Returns:
            A descriptive 'how' field
        """
        base_description = mechanism.value.replace('_', ' ')
        
        if details:
            # Add specific details if provided
            if mechanism == MechanismType.TOOL_USE and 'tool_name' in details:
                return f"tool use: {details['tool_name']}"
            elif mechanism == MechanismType.FUNCTION_CALL and 'function' in details:
                return f"function call: {details['function']}"
            elif mechanism == MechanismType.API_REQUEST and 'endpoint' in details:
                return f"API request to {details['endpoint']}"
            elif mechanism == MechanismType.FILE_OPERATION and 'operation' in details:
                return f"file {details['operation']}"
            elif mechanism == MechanismType.WEB_SEARCH and 'query' in details:
                return f"web search for: {details['query'][:30]}"
                
        return base_description
    
    def parse_user_intent(self, message: str) -> Tuple[str, float]:
        """
        Parse user intent from a message.
        
        Args:
            message: The user message
            
        Returns:
            Tuple of (intent_category, confidence_score)
        """
        message_lower = message.lower()
        
        # Define intent patterns with confidence scores
        intents = {
            'technical_question': (['how do i', 'how to', 'implement', 'code', 'program', 'api', 'function'], 0.9),
            'troubleshooting': (['error', 'bug', 'issue', 'problem', 'fix', 'broken', 'not working'], 0.95),
            'explanation': (['what is', 'explain', 'understand', 'why does', 'tell me about'], 0.85),
            'planning': (['plan', 'schedule', 'organize', 'prepare', 'strategy'], 0.8),
            'analysis': (['analyze', 'examine', 'investigate', 'compare', 'evaluate'], 0.85),
            'creation': (['create', 'build', 'make', 'develop', 'design'], 0.85),
            'optimization': (['optimize', 'improve', 'enhance', 'better', 'faster'], 0.8),
            'configuration': (['setup', 'configure', 'install', 'settings'], 0.85),
        }
        
        for intent, (keywords, base_confidence) in intents.items():
            if any(keyword in message_lower for keyword in keywords):
                # Adjust confidence based on message characteristics
                confidence = base_confidence
                if '?' in message:
                    confidence += 0.05
                if len(message.split()) > 10:
                    confidence += 0.05
                return intent, min(confidence, 1.0)
        
        # Default intent
        return 'general_query', 0.5