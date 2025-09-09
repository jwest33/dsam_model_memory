"""
LLM-based temporal parser for extracting temporal hints from natural language.
Handles flexible temporal references like "yesterday", "last Tuesday", "two weeks ago", etc.
"""

from typing import Optional, Union, Tuple, Dict
from datetime import datetime, timedelta
import json
import re
import requests

from ..config import cfg


def _call_llm_for_temporal(prompt: str) -> Optional[str]:
    """Call LLM for temporal parsing using the same pattern as llm_extractor."""
    url = f"{cfg.llm_base_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    body = {
        "model": cfg.llm_model,
        "messages": [
            {"role": "system", "content": "You are a temporal information extractor. Extract temporal references from text and return structured JSON."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 200
    }
    try:
        resp = requests.post(url, json=body, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        pass
    return None


class TemporalParser:
    """Parses natural language temporal references into structured temporal hints."""
    
    def __init__(self):
        """Initialize temporal parser."""
        pass
        
    def parse_temporal_reference(self, text: str, reference_date: Optional[datetime] = None) -> Optional[Union[str, Tuple[str, str], Dict]]:
        """
        Parse natural language temporal reference into structured format.
        
        Args:
            text: User query that may contain temporal references
            reference_date: The date to use as "today" (defaults to current date)
            
        Returns:
            None if no temporal reference found, otherwise:
            - Single date string: "2024-01-15"
            - Date range tuple: ("2024-01-10", "2024-01-20")  
            - Relative time dict: {"relative": "last_week"}
        """
        if reference_date is None:
            reference_date = datetime.utcnow()
        
        # Create prompt for LLM to extract temporal information
        prompt = f"""Extract temporal information from the following query and convert it to a structured format.
Today's date is: {reference_date.strftime('%Y-%m-%d')} ({reference_date.strftime('%A')})

Query: "{text}"

Analyze the query for any temporal references such as:
- Specific dates: "January 15th", "2024-01-15"
- Relative dates: "yesterday", "last week", "two days ago", "last Tuesday"
- Date ranges: "last month", "this week", "between Monday and Friday"
- Time periods: "in the morning", "afternoon of January 10th"
- Fuzzy references: "recently", "a while ago", "the other day"

Return a JSON object with one of these formats:
1. For specific date: {{"type": "date", "value": "YYYY-MM-DD"}}
2. For date range: {{"type": "range", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}
3. For standard relative terms: {{"type": "relative", "value": "yesterday|today|last_week|last_month|last_year"}}
4. For custom computed dates: {{"type": "computed", "date": "YYYY-MM-DD", "description": "explanation"}}
5. If no temporal reference: {{"type": "none"}}

Examples:
- "Show me messages from yesterday" -> {{"type": "relative", "value": "yesterday"}}
- "What did we discuss last Tuesday?" -> {{"type": "date", "value": "2024-01-09"}}
- "Find conversations from last month" -> {{"type": "range", "start": "2024-12-01", "end": "2024-12-31"}}
- "Remember that chat from a couple days ago?" -> {{"type": "range", "start": "2024-01-13", "end": "2024-01-14"}}
- "What is the capital of France?" -> {{"type": "none"}}

Return ONLY the JSON object, no other text."""

        try:
            # Get LLM response
            response = _call_llm_for_temporal(prompt)
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return None
                
            result = json.loads(json_match.group())
            
            # Convert to our expected format
            if result.get("type") == "none":
                return None
            elif result.get("type") == "date":
                return result["value"]
            elif result.get("type") == "range":
                return (result["start"], result["end"])
            elif result.get("type") == "relative":
                return {"relative": result["value"]}
            elif result.get("type") == "computed":
                # For computed dates, return as single date
                return result["date"]
            else:
                return None
                
        except Exception as e:
            # If LLM parsing fails, fall back to simple keyword detection
            return self._fallback_parse(text, reference_date)
    
    def _fallback_parse(self, text: str, reference_date: datetime) -> Optional[Union[str, Dict]]:
        """Simple fallback parser using keyword matching."""
        text_lower = text.lower()
        
        # Check for common relative terms
        if any(word in text_lower for word in ["yesterday", "yday"]):
            return {"relative": "yesterday"}
        elif any(word in text_lower for word in ["today", "now", "current"]):
            return {"relative": "today"}
        elif "last week" in text_lower or "past week" in text_lower:
            return {"relative": "last_week"}
        elif "last month" in text_lower or "past month" in text_lower:
            return {"relative": "last_month"}
        elif "last year" in text_lower or "past year" in text_lower:
            return {"relative": "last_year"}
        
        # Check for "X days/weeks/months ago" patterns
        ago_pattern = r'(\d+)\s+(day|week|month|year)s?\s+ago'
        match = re.search(ago_pattern, text_lower)
        if match:
            amount = int(match.group(1))
            unit = match.group(2)
            
            if unit == "day":
                target_date = reference_date - timedelta(days=amount)
                return target_date.strftime('%Y-%m-%d')
            elif unit == "week":
                start_date = reference_date - timedelta(weeks=amount)
                end_date = reference_date - timedelta(weeks=amount-1)
                return (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            elif unit == "month":
                # Approximate month as 30 days
                start_date = reference_date - timedelta(days=amount*30)
                end_date = reference_date - timedelta(days=(amount-1)*30)
                return (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # Check for specific date patterns (YYYY-MM-DD)
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        date_match = re.search(date_pattern, text)
        if date_match:
            return date_match.group()
        
        return None
    
    def extract_and_clean_query(self, text: str) -> Tuple[Optional[Union[str, Tuple[str, str], Dict]], str]:
        """
        Extract temporal hint and return cleaned query text.
        
        Returns:
            (temporal_hint, cleaned_query_text)
        """
        temporal_hint = self.parse_temporal_reference(text)
        
        if temporal_hint:
            # Remove obvious temporal phrases from query for better semantic search
            cleaned_text = text
            temporal_phrases = [
                "yesterday", "today", "last week", "last month", "last year",
                "ago", "recent", "recently", "earlier", "before", "previous",
                r"\d{4}-\d{2}-\d{2}",  # Date patterns
                r"\d+\s+(day|week|month|year)s?\s+ago"  # X time ago patterns
            ]
            
            for phrase in temporal_phrases:
                cleaned_text = re.sub(phrase, "", cleaned_text, flags=re.IGNORECASE)
            
            # Clean up extra spaces
            cleaned_text = " ".join(cleaned_text.split())
            
            # If query becomes too short after cleaning, keep original
            if len(cleaned_text) < 10:
                cleaned_text = text
                
            return temporal_hint, cleaned_text
        
        return None, text
