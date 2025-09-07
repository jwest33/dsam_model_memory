#!/usr/bin/env python
"""
Wrapper script for running JAM Web Application.
"""

from agentic_memory.server.flask_app import app

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5001, debug=False)
