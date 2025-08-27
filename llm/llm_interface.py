"""
LLM Interface for the memory system

Integrates with llama_server_client for language model operations.
"""

import logging
import json
from typing import Dict, Any, Optional, List, Tuple
import re

from config import get_config

# Import the existing llama_server_client
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from llama_server_client import LlamaServerClient, ensure_llm_server

logger = logging.getLogger(__name__)

class LLMInterface:
    """Interface for LLM operations in the memory system"""
    
    def __init__(self, config=None):
        """Initialize LLM interface"""
        self.config = config or get_config().llm
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM client"""
        try:
            # Ensure server is running
            if ensure_llm_server():
                self.client = LlamaServerClient(
                    base_url=self.config.server_url,
                    timeout=self.config.timeout
                )
                
                # Test connection
                if self.client.health_check():
                    logger.info(f"Connected to LLM server at {self.config.server_url}")
                else:
                    logger.warning("LLM server not responding, will retry on first use")
            else:
                logger.warning("Failed to start LLM server")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        use_chat_endpoint: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            use_chat_endpoint: Use chat completions endpoint (better for instruction models)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        if not self.client:
            self._initialize_client()
            if not self.client:
                logger.error("LLM client not available")
                return ""
        
        try:
            # Add repetition penalty if not provided
            if 'repetition_penalty' not in kwargs:
                kwargs['repetition_penalty'] = getattr(self.config, 'repetition_penalty', 1.2)
            
            if use_chat_endpoint and hasattr(self.client, 'chat_completion'):
                # Use chat completion for better handling of instruction-tuned models
                response = self.client.chat_completion(
                    prompt=prompt,
                    max_tokens=max_tokens or self.config.max_tokens,
                    temperature=temperature or self.config.temperature,
                    stop=stop or [],
                    **kwargs
                )
                
                # Extract from chat completion format
                if 'choices' in response and response['choices']:
                    choice = response['choices'][0]
                    if 'message' in choice:
                        return choice['message'].get('content', '')
                    elif 'text' in choice:
                        return choice['text']
            else:
                # Fallback to regular completion
                response = self.client.completion(
                    prompt=prompt,
                    max_tokens=max_tokens or self.config.max_tokens,
                    temperature=temperature or self.config.temperature,
                    stop=stop or [],
                    **kwargs
                )
                
                # Extract generated text
                if 'choices' in response and response['choices']:
                    return response['choices'][0].get('text', '')
            
            return ""
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response"""
        # Try to find JSON block in response
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Try to parse the entire response as JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None
    
    def extract_number(self, text: str, default: float = 0.5) -> float:
        """Extract a number from LLM response"""
        # Look for decimal numbers
        number_pattern = r'(\d*\.?\d+)'
        matches = re.findall(number_pattern, text)
        
        if matches:
            try:
                value = float(matches[0])
                # Clamp to [0, 1] for typical scores
                return max(0.0, min(1.0, value))
            except ValueError:
                pass
        
        return default
    
    def parse_5w1h(self, text: str) -> Dict[str, str]:
        """Parse 5W1H structure from text"""
        result = {
            "who": "",
            "what": "",
            "when": "",
            "where": "",
            "why": "",
            "how": ""
        }
        
        # Look for labeled sections
        patterns = {
            "who": r"(?:who|actor|agent):\s*(.+?)(?:\n|$)",
            "what": r"(?:what|action|content):\s*(.+?)(?:\n|$)",
            "when": r"(?:when|time|timestamp):\s*(.+?)(?:\n|$)",
            "where": r"(?:where|location|context):\s*(.+?)(?:\n|$)",
            "why": r"(?:why|reason|intent):\s*(.+?)(?:\n|$)",
            "how": r"(?:how|method|mechanism):\s*(.+?)(?:\n|$)"
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result[key] = match.group(1).strip()
        
        return result
    
    def complete_5w1h(
        self,
        partial: Dict[str, str],
        context: Optional[str] = None
    ) -> Dict[str, str]:
        """Complete missing 5W1H slots using LLM"""
        # Build concise prompt
        prompt = "Complete missing 5W1H fields.\n"
        
        if context:
            prompt += f"Context: {context}\n"
        
        prompt += "Given:\n"
        for key, value in partial.items():
            if value:
                prompt += f"{key}: {value}\n"
        
        missing = [k for k in ["who", "what", "when", "where", "why", "how"] if not partial.get(k)]
        prompt += f"\nProvide only: {', '.join(missing)}\nFormat: field: value\nBe concise. No explanations."
        
        # Generate completion
        response = self.generate(prompt, max_tokens=200)
        
        # Parse response
        completed = self.parse_5w1h(response)
        
        # Merge with original
        result = partial.copy()
        for key, value in completed.items():
            if not result.get(key) and value:
                result[key] = value
        
        return result
    
    def analyze_causality(
        self,
        action: str,
        observation: str
    ) -> Dict[str, Any]:
        """Analyze causal relationship between action and observation"""
        prompt = f"""Action: {action}
Observation: {observation}

JSON output only:
{{"is_causal": bool, "confidence": 0-1, "mechanism": "brief description"}}"""
        
        response = self.generate(prompt, max_tokens=150)
        result = self.extract_json(response)
        
        if not result:
            # Fallback parsing
            result = {
                "is_causal": "yes" in response.lower(),
                "confidence": self.extract_number(response, 0.5),
                "mechanism": "unknown"
            }
        
        return result
    
    def suggest_tags(
        self,
        event_description: str,
        max_tags: int = 5
    ) -> List[str]:
        """Suggest tags for an event"""
        prompt = f"""Event: {event_description[:100]}
Output {max_tags} tags, comma-separated:"""
        
        response = self.generate(prompt, max_tokens=50)
        
        # Extract tags
        tags = []
        if ',' in response:
            tags = [tag.strip() for tag in response.split(',')]
        else:
            # Try space-separated
            tags = response.split()
        
        # Clean tags
        tags = [tag.strip('.,;:"\'').lower() for tag in tags]
        tags = [tag for tag in tags if tag and len(tag) < 30]
        
        return tags[:max_tags]
    
    def summarize_episode(
        self,
        events: List[Dict[str, Any]]
    ) -> str:
        """Summarize a sequence of events"""
        if not events:
            return "No events to summarize."
        
        prompt = "Summarize in 1 sentence:\n"
        
        for i, event in enumerate(events[:5], 1):  # Limit to first 5
            prompt += f"{event.get('what', 'Unknown')[:30]}\n"
        
        if len(events) > 5:
            prompt += f"(+{len(events) - 5} more)\n"
        
        prompt += "Summary:"
        
        response = self.generate(prompt, max_tokens=100)
        return response.strip()
    
    def is_available(self) -> bool:
        """Check if LLM is available"""
        if not self.client:
            self._initialize_client()
        
        return self.client is not None and self.client.health_check()
    
    def __repr__(self) -> str:
        """String representation"""
        status = "connected" if self.is_available() else "disconnected"
        return f"LLMInterface(status={status}, url={self.config.server_url})"
