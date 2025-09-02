"""
Test script for enhanced 5W1H field generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from memory.field_generator import FieldGenerator, MechanismType
from llm.llm_interface import LLMInterface
from config import get_config

def test_field_generation():
    """Test the enhanced field generation"""
    
    print("=" * 60)
    print("Testing Enhanced 5W1H Field Generation")
    print("=" * 60)
    
    # Initialize components
    config = get_config()
    llm_interface = LLMInterface(config.llm)
    field_generator = FieldGenerator(llm_client=llm_interface)
    
    # Test cases
    test_messages = [
        # Technical queries
        {
            'message': "How do I implement a REST API with authentication?",
            'who': 'User',
            'type': 'query',
            'expected_intent': 'technical'
        },
        {
            'message': "I'm getting a 404 error when calling the endpoint",
            'who': 'User',
            'type': 'query',
            'expected_intent': 'troubleshooting'
        },
        # Planning queries
        {
            'message': "Help me plan a trip to Japan next month",
            'who': 'User',
            'type': 'query',
            'expected_intent': 'planning'
        },
        # Analysis queries
        {
            'message': "Analyze the performance metrics from last quarter",
            'who': 'User',
            'type': 'query',
            'expected_intent': 'analysis'
        },
        # Assistant responses
        {
            'message': "To implement a REST API, you'll need to set up endpoints for each resource...",
            'who': 'Assistant',
            'type': 'response',
            'expected_intent': 'guidance'
        },
        {
            'message': "The 404 error indicates the endpoint path doesn't exist. Check your route configuration...",
            'who': 'Assistant',
            'type': 'response',
            'expected_intent': 'solution'
        }
    ]
    
    print("\n1. Testing 'WHY' Field Generation:")
    print("-" * 40)
    
    for test_case in test_messages:
        # Add to context
        field_generator.add_to_context(test_case['message'], test_case['who'])
        
        # Generate why field
        why = field_generator.generate_why_field(
            test_case['message'],
            test_case['who'],
            test_case['type']
        )
        
        print(f"\nMessage: {test_case['message'][:60]}...")
        print(f"Who: {test_case['who']}")
        print(f"Generated 'why': {why}")
        print(f"Expected type: {test_case['expected_intent']}")
    
    print("\n\n2. Testing 'HOW' Field Generation:")
    print("-" * 40)
    
    # Test different mechanisms
    mechanisms = [
        (MechanismType.CHAT_INTERFACE, {'interface': 'web'}),
        (MechanismType.API_CALL, {'endpoint': '/api/memories'}),
        (MechanismType.TOOL_USE, {'tool_name': 'calculator'}),
        (MechanismType.FUNCTION_CALL, {'function': 'search_database'}),
        (MechanismType.WEB_SEARCH, {'query': 'python tutorials'}),
        (MechanismType.FILE_OPERATION, {'operation': 'read'}),
        (MechanismType.LLM_GENERATION, {'model': 'llama-3'}),
        (MechanismType.CLI_COMMAND, {}),
    ]
    
    for mechanism, details in mechanisms:
        how = field_generator.generate_how_field(mechanism, details)
        print(f"\nMechanism: {mechanism.value}")
        print(f"Details: {details}")
        print(f"Generated 'how': {how}")
    
    print("\n\n3. Testing Intent Parsing:")
    print("-" * 40)
    
    test_intents = [
        "How do I fix this error in my code?",
        "What is machine learning?",
        "Create a new user account",
        "Optimize the database query performance",
        "Schedule a meeting for next week",
    ]
    
    for message in test_intents:
        intent, confidence = field_generator.parse_user_intent(message)
        print(f"\nMessage: {message}")
        print(f"Intent: {intent}, Confidence: {confidence:.2f}")
    
    print("\n\n4. Testing Context Building:")
    print("-" * 40)
    
    # Clear context and add a conversation
    field_generator.recent_context = []
    conversation = [
        ("User", "I need help with my Python script"),
        ("Assistant", "What issue are you experiencing?"),
        ("User", "It's throwing an ImportError"),
        ("Assistant", "Let me help you resolve that ImportError"),
    ]
    
    for who, message in conversation:
        field_generator.add_to_context(message, who)
    
    context_str = field_generator._build_context_string()
    print("\nBuilt context:")
    print(context_str)
    
    # Generate why field with context
    final_message = "The module 'requests' is not found"
    why_with_context = field_generator.generate_why_field(
        final_message, "User", "query"
    )
    print(f"\nFinal message: {final_message}")
    print(f"Generated 'why' with context: {why_with_context}")
    
    print("\n" + "=" * 60)
    print("Field Generation Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_field_generation()
