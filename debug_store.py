from memory.memory_store import MemoryStore
from models.event import Event, FiveW1H, EventType
import traceback

store = MemoryStore()
event = Event(
    five_w1h=FiveW1H(who='test', what='test', when='now', where='here', why='test', how='test'),
    event_type=EventType.ACTION
)
try:
    store.store_event(event)
except Exception as e:
    traceback.print_exc()