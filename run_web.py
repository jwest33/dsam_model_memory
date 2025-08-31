#!/usr/bin/env python
"""
Quick launcher for the Enhanced Dual-Space Web Interface
"""

import sys
import webbrowser
import signal
import atexit
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

def shutdown_handler():
    """Save memory state on shutdown"""
    print("\nSaving memory state before shutdown...")
    try:
        # Try to import from enhanced version first
        from web_app import memory_agent
    except ImportError:
        print("\nError: Enhanced interface not available")
    
    if memory_agent:
        memory_agent.save()
        print("Memory state saved successfully")

def main():
    """Launch the enhanced web interface"""
    print("=" * 60)
    print("[DSAM] Dual-Space Agentic Memory - Enhanced Web Interface v2.0")
    print("=" * 60)
    
    # Try to use enhanced version if available
    try:
        from web_app import app, initialize_system
        print("\n  Loading enhanced dual-space interface...")
        print("   Features: Space indicators, residual tracking, analytics")
    except ImportError as e:
        print(f"\nError: Enhanced interface not available - {e}")
        print("Please check that web_app.py is properly configured.")
        sys.exit(1)
    
    print("\nStarting Flask server...")
    print("The web interface will open in your browser.")
    print("Press Ctrl+C to stop the server.\n")
    
    # Try to open browser after a short delay
    import threading
    def open_browser():
        import time
        time.sleep(5)
        webbrowser.open('http://localhost:5000')
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Initialize the system
    initialize_system()
    
    # Register shutdown handlers
    atexit.register(shutdown_handler)
    signal.signal(signal.SIGINT, lambda s, f: (shutdown_handler(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda s, f: (shutdown_handler(), sys.exit(0)))
    
    # Run the Flask app
    app.run(debug=False, port=5000, host='127.0.0.1')

if __name__ == '__main__':
    main()
