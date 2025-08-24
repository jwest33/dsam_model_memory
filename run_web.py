#!/usr/bin/env python
"""
Quick launcher for the Flask web interface
"""

import sys
import webbrowser
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

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
    
    # Run the Flask app
    app.run(debug=False, port=5000, host='127.0.0.1')

if __name__ == '__main__':
    main()