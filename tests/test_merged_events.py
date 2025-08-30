"""
Tests for the Enhanced Merged Event System

This module tests the new merging system with 5W1H decomposition,
temporal chains, and context generation.
"""

import unittest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.event import Event, FiveW1H, EventType
from models.merged_event import MergedEvent, EventRelationship, ComponentVariant
from memory.smart_merger import SmartMerger
from memory.temporal_chain import TemporalChain
from memory.context_generator import MergedEventContextGenerator
from memory.memory_store import MemoryStore


class TestMergedEventSystem(unittest.TestCase):
    """Test the complete merged event system"""
    
    def setUp(self):
        """Set up test environment"""
        self.memory_store = MemoryStore()
        self.smart_merger = SmartMerger()
        self.temporal_chain = TemporalChain()
        self.context_generator = MergedEventContextGenerator(self.temporal_chain)
    
    def test_merged_event_creation(self):
        """Test creating and updating a merged event"""
        # Create base merged event
        merged = MergedEvent(
            id="test_merged_1",
            base_event_id="event_1"
        )
        
        # Add first event
        event1_data = {
            'who': 'Alice',
            'what': 'implemented authentication feature',
            'when': 'yesterday',
            'where': 'backend',
            'why': 'security requirement',
            'how': 'using OAuth2',
            'timestamp': datetime.now() - timedelta(days=1)
        }
        merged.add_raw_event('event_1', event1_data)
        
        # Verify first event
        self.assertEqual(merged.merge_count, 2)  # Base + 1 added
        self.assertIn('Alice', merged.who_variants)
        self.assertEqual(len(merged.who_variants['Alice']), 1)
        
        # Add update event
        event2_data = {
            'who': 'Alice',
            'what': 'fixed authentication bug',
            'when': 'today',
            'where': 'backend',
            'why': 'bug report',
            'how': 'debugging',
            'timestamp': datetime.now()
        }
        merged.add_raw_event('event_2', event2_data, EventRelationship.UPDATE)
        
        # Verify update
        self.assertEqual(merged.merge_count, 3)
        self.assertEqual(len(merged.what_variants), 2)  # Two different actions
        
        # Check temporal timeline
        self.assertEqual(len(merged.when_timeline), 2)
        self.assertEqual(merged.when_timeline[0].event_id, 'event_1')
        self.assertEqual(merged.when_timeline[1].event_id, 'event_2')
        
        # Check dominant pattern
        self.assertIsNotNone(merged.dominant_pattern)
        self.assertEqual(merged.dominant_pattern['who'], 'Alice')
        self.assertEqual(merged.dominant_pattern['where'], 'backend')
    
    def test_smart_merger_relationship_detection(self):
        """Test the smart merger's ability to detect relationships"""
        existing = MergedEvent(
            id="merged_1",
            base_event_id="event_1"
        )
        
        # Add initial event
        existing.add_raw_event('event_1', {
            'who': 'Bob',
            'what': 'created user registration',
            'timestamp': datetime.now() - timedelta(hours=2)
        })
        
        # Test correction detection
        correction_event = Event(
            id='event_2',
            five_w1h=FiveW1H(
                who='Bob',
                what='fixed user registration validation',
                when='',
                where='',
                why='',
                how=''
            ),
            event_type=EventType.ACTION
        )
        
        relationship = self.smart_merger._determine_relationship(existing, correction_event)
        self.assertEqual(relationship, EventRelationship.CORRECTION)
        
        # Test continuation detection
        continuation_event = Event(
            id='event_3',
            five_w1h=FiveW1H(
                who='Bob',
                what='continue implementing user profiles',
                when='',
                where='',
                why='',
                how=''
            ),
            event_type=EventType.ACTION
        )
        
        relationship = self.smart_merger._determine_relationship(existing, continuation_event)
        self.assertEqual(relationship, EventRelationship.CONTINUATION)
    
    def test_temporal_chain_management(self):
        """Test temporal chain creation and management"""
        # Create events
        event1 = Event(
            id='event_1',
            five_w1h=FiveW1H(
                who='Charlie',
                what='started debugging session',
                when='',
                where='web_chat',
                why='',
                how=''
            ),
            event_type=EventType.USER_INPUT,
            episode_id='episode_1'
        )
        
        event2 = Event(
            id='event_2',
            five_w1h=FiveW1H(
                who='Assistant',
                what='provided debugging suggestions',
                when='',
                where='web_chat',
                why='',
                how=''
            ),
            event_type=EventType.ACTION,
            episode_id='episode_1'
        )
        
        # Add to temporal chain
        chain_id1 = self.temporal_chain.add_event(event1)
        chain_id2 = self.temporal_chain.add_event(event2)
        
        # Should be in same chain (same episode)
        self.assertEqual(chain_id1, chain_id2)
        
        # Check chain contents
        chain = self.temporal_chain.get_chain_for_event('event_1')
        self.assertIsNotNone(chain)
        self.assertEqual(len(chain), 2)
        self.assertIn('event_1', chain)
        self.assertIn('event_2', chain)
        
        # Check latest state
        latest = self.temporal_chain.get_latest_valid_state(chain_id1)
        self.assertIsNotNone(latest)
        self.assertEqual(latest['who'], 'Assistant')
    
    def test_context_generation(self):
        """Test context generation for LLM"""
        # Create a complex merged event
        merged = MergedEvent(
            id="merged_complex",
            base_event_id="event_1"
        )
        
        # Add multiple events with variations
        events_data = [
            {
                'who': 'Alice',
                'what': 'implemented feature X',
                'when': '10:00',
                'where': 'frontend',
                'why': 'user request',
                'how': 'React components',
                'timestamp': datetime.now() - timedelta(hours=3)
            },
            {
                'who': 'Bob',
                'what': 'updated feature X with animations',
                'when': '11:00',
                'where': 'frontend',
                'why': 'UX improvement',
                'how': 'CSS transitions',
                'timestamp': datetime.now() - timedelta(hours=2)
            },
            {
                'who': 'Alice',
                'what': 'fixed feature X performance issue',
                'when': '14:00',
                'where': 'frontend',
                'why': 'performance optimization',
                'how': 'memoization',
                'timestamp': datetime.now() - timedelta(hours=1)
            }
        ]
        
        for i, data in enumerate(events_data):
            relationship = EventRelationship.INITIAL if i == 0 else EventRelationship.UPDATE
            merged.add_raw_event(f'event_{i+1}', data, relationship)
        
        # Generate different context formats
        
        # Summary context
        summary = self.context_generator._generate_summary_context(merged)
        self.assertIn('Alice', summary)  # Dominant actor
        self.assertIn('[4 merged events]', summary)  # 1 base + 3 added = 4
        
        # Detailed context
        detailed = self.context_generator._generate_detailed_context(merged, None)
        self.assertIn('[Merged 4 similar events]', detailed)  # 1 base + 3 added = 4
        self.assertIn('Actors:', detailed)
        self.assertIn('Alice (2x), Bob', detailed)
        self.assertIn('Actions:', detailed)
        self.assertIn('Time:', detailed)
        
        # Structured context
        structured = self.context_generator._generate_structured_context(merged, None)
        self.assertIn('EVENT_ID: merged_complex', structured)
        self.assertIn('MERGE_COUNT: 4', structured)  # 1 base + 3 added = 4
        self.assertIn('WHO: Alice', structured)  # Latest
        
        print(f"\nSummary Context:\n{summary}\n")
        print(f"Detailed Context:\n{detailed}\n")
        print(f"Structured Context:\n{structured}\n")
    
    def test_memory_store_integration(self):
        """Test the complete integration with memory store"""
        # Store similar events that should merge
        events = [
            Event(
                id='auth_1',
                five_w1h=FiveW1H(
                    who='Developer',
                    what='implementing authentication system',
                    when='',
                    where='backend',
                    why='security requirements',
                    how='JWT tokens'
                ),
                event_type=EventType.ACTION
            ),
            Event(
                id='auth_2',
                five_w1h=FiveW1H(
                    who='Developer',
                    what='implementing authentication with OAuth',
                    when='',
                    where='backend',
                    why='security requirements',
                    how='OAuth2 flow'
                ),
                event_type=EventType.ACTION
            ),
            Event(
                id='auth_3',
                five_w1h=FiveW1H(
                    who='Developer',
                    what='fixing authentication token expiry',
                    when='',
                    where='backend',
                    why='bug fix',
                    how='refresh tokens'
                ),
                event_type=EventType.ACTION
            )
        ]
        
        # Store events
        for event in events:
            success, message = self.memory_store.store_event(event)
            print(f"Stored {event.id}: {message}")
        
        # Retrieve with context
        results = self.memory_store.retrieve_memories_with_context(
            {'what': 'authentication'},
            k=5,
            context_format='detailed'
        )
        
        # Check results
        self.assertGreater(len(results), 0)
        
        # Print generated contexts
        for item, score, context in results:
            print(f"\n=== Retrieved Item (Score: {score:.3f}) ===")
            if hasattr(item, 'merge_count'):
                print(f"Merged Event with {item.merge_count} events")
            print(f"Context:\n{context}")
    
    def test_merge_quality_analysis(self):
        """Test merge quality analysis"""
        merged = MergedEvent(
            id="test_quality",
            base_event_id="event_1"
        )
        
        # Add consistent events (should have high coherence)
        base_time = datetime.now()
        for i in range(5):
            merged.add_raw_event(f'event_{i}', {
                'who': 'Alice',
                'what': f'working on feature implementation step {i+1}',
                'where': 'backend',
                'timestamp': base_time + timedelta(hours=i)
            })
        
        # Analyze quality
        metrics = self.smart_merger.analyze_merge_quality(merged)
        
        print("\nMerge Quality Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}")
        
        # Should have high coherence
        self.assertGreater(metrics['actor_consistency'], 0.5)
        self.assertGreater(metrics['location_consistency'], 0.9)
        self.assertGreater(metrics['overall_coherence'], 0.6)
    
    def test_temporal_progression(self):
        """Test temporal progression in merged events"""
        merged = MergedEvent(
            id="temporal_test",
            base_event_id="event_1"
        )
        
        # Create a progression of events
        base_time = datetime.now() - timedelta(days=5)
        progression = [
            ('initial implementation', EventRelationship.INITIAL),
            ('updated with improvements', EventRelationship.UPDATE),
            ('fixed critical bug', EventRelationship.CORRECTION),
            ('optimized performance', EventRelationship.UPDATE),
            ('added new features', EventRelationship.CONTINUATION)
        ]
        
        for i, (action, relationship) in enumerate(progression):
            merged.add_raw_event(f'event_{i}', {
                'who': 'Team',
                'what': action,
                'timestamp': base_time + timedelta(days=i)
            }, relationship)
        
        # Generate context with temporal focus
        query_context = {'query': {'when': 'timeline'}}
        context = self.context_generator.generate_context(
            merged, query_context, 'detailed'
        )
        
        print(f"\nTemporal Progression Context:\n{context}")
        
        # Verify timeline is included
        self.assertIn('Time:', context)
        self.assertIn('days', context)  # Should show day range
        
        # Check latest state
        latest = merged.get_latest_state()
        self.assertEqual(latest['what'], 'added new features')


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)
