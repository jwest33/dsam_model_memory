#!/usr/bin/env python
"""
Wrapper script for running JAM Web Application with all servers.
This script starts the LLM server, embedding server, and web interface.
"""

import sys
import subprocess

if __name__ == "__main__":
    # Run the CLI command to start all servers
    subprocess.run([sys.executable, "-m", "agentic_memory.cli", "server", "start", "--all"])
