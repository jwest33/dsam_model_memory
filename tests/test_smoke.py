from agentic_memory.types import RawEvent
from agentic_memory.server.flask_app import app

def test_import():
    assert app is not None
