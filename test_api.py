"""Quick test of the memory API"""

from agent.memory_agent import MemoryAgent
from config import get_config
import json

# Initialize
config = get_config()
memory_agent = MemoryAgent(config)

# Add a test memory
success, msg, event = memory_agent.remember(
    who="Test User",
    what="This is a test memory",
    where="test_environment",
    why="Testing the system",
    how="Manual test",
    event_type="observation"
)
print(f"Memory creation: {success}, {msg}")

# Get statistics
stats = memory_agent.get_statistics()
print(f"Stats: {json.dumps(stats, indent=2, default=str)}")

# Get memories like the API does
raw_memories = []
if hasattr(memory_agent.memory_store, 'raw_memories'):
    for event in memory_agent.memory_store.raw_memories:
        raw_memories.append({
            'id': event.id,
            'who': event.five_w1h.who,
            'what': event.five_w1h.what,
            'when': event.five_w1h.when,
            'where': event.five_w1h.where,
            'why': event.five_w1h.why,
            'how': event.five_w1h.how,
            'type': event.event_type.value,
            'salience': 0.5,  # Now handled by block salience matrix
            'episode_id': event.episode_id
        })

processed_memories = []
if hasattr(memory_agent.memory_store, 'processed_memories'):
    for event in memory_agent.memory_store.processed_memories:
        blocks = memory_agent.memory_store.block_manager.get_blocks_for_event(event.id)
        block_ids = [b.id for b in blocks]
        block_saliences = [b.block_salience for b in blocks]
        
        processed_memories.append({
            'id': event.id,
            'who': event.five_w1h.who,
            'what': event.five_w1h.what,
            'when': event.five_w1h.when,
            'where': event.five_w1h.where,
            'why': event.five_w1h.why,
            'how': event.five_w1h.how,
            'type': event.event_type.value,
            'salience': 0.5,  # Now handled by block salience matrix
            'episode_id': event.episode_id,
            'block_ids': block_ids,
            'block_saliences': block_saliences
        })

memory_blocks = []
if hasattr(memory_agent.memory_store, 'block_manager'):
    for block_id, block in memory_agent.memory_store.block_manager.blocks.items():
        memory_blocks.append({
            'id': block.id,
            'type': block.block_type,
            'event_count': len(block.events),
            'event_ids': list(block.event_ids),
            'salience': block.salience,
            'coherence': block.coherence_score,
            'link_count': len(block.links),
            'aggregate_signature': block.aggregate_signature.to_dict() if block.aggregate_signature else None,
            'created_at': block.created_at.isoformat() if block.created_at else None,
            'updated_at': block.updated_at.isoformat() if block.updated_at else None
        })

result = {
    'raw': raw_memories,
    'processed': processed_memories,
    'blocks': memory_blocks,
    'stats': stats
}

print(f"\nAPI Response structure:")
print(f"- raw: {len(result['raw'])} memories")
print(f"- processed: {len(result['processed'])} memories")
print(f"- blocks: {len(result['blocks'])} blocks")
print(f"- stats: {type(result['stats'])}")