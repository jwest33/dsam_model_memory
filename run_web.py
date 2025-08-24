#!/usr/bin/env python
"""
Quick launcher for the Flask web interface
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
    from web_app import memory_agent
    if memory_agent:
        memory_agent.save()
        print("Memory state saved successfully")

def main():
    """Launch the web interface"""
    print("=" * 50)
    print("5W1H Memory Framework - Web Interface")
    print("=" * 50)
    print("\nStarting Flask server...")
    print("The web interface will open in your browser.")
    print("Press Ctrl+C to stop the server.\n")
    
    # Try to open browser after a short delay
    import threading
    def open_browser():
        import time
        time.sleep(2)
        webbrowser.open('http://localhost:5000')
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Import and run the Flask app
    from web_app import app, initialize_system
    
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
