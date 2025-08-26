"""
Simple runner for web-based conversation experiments
Starts the web server and runs experiments
"""

import subprocess
import time
import sys
import os
import signal
import requests
from pathlib import Path

# Set offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

def check_port(port=5000, timeout=30):
    """Check if web server is running on port"""
    print(f"Checking for web server on port {port}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/api/stats", timeout=1)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    
    return False

def cleanup_chromadb():
    """Clean ChromaDB data for fresh start"""
    chromadb_path = Path("state/chromadb")
    if chromadb_path.exists():
        import shutil
        shutil.rmtree(chromadb_path)
        print("✓ Cleaned ChromaDB data")

def main():
    print("="*60)
    print("AUTOMATED EXPERIMENT RUNNER")
    print("="*60)
    
    # Check if web server is already running
    if check_port(5000, timeout=2):
        print("✓ Web server already running")
        web_process = None
    else:
        print("Starting web server...")
        # Start web server in background
        web_process = subprocess.Popen(
            [sys.executable, "run_web.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        if not check_port(5000, timeout=30):
            print("❌ Failed to start web server")
            if web_process:
                web_process.terminate()
            return 1
        
        print("✓ Web server started")
    
    try:
        # Ask user for experiment type
        print("\n" + "="*60)
        print("EXPERIMENT OPTIONS")
        print("="*60)
        print("\n1. Quick test (5 conversations)")
        print("2. Comprehensive test (all features)")
        print("3. Clean data and run comprehensive test")
        
        choice = input("\nSelect option (1-3) [default: 1]: ").strip() or "1"
        
        if choice == "3":
            cleanup_chromadb()
            choice = "2"  # Run comprehensive test after cleanup
        
        # Run experiments
        print("\nStarting conversation experiments...")
        
        # Create input for simulate_conversations.py
        sim_input = choice if choice in ["1", "2"] else "1"
        
        # Run simulate_conversations with input
        sim_process = subprocess.Popen(
            [sys.executable, "simulate_conversations.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Send input and get output
        output, _ = sim_process.communicate(input=sim_input)
        
        # Print output line by line for better formatting
        for line in output.split('\n'):
            if not any(skip in line for skip in ['AttributeError', 'oneDNN', 'tensorflow', 'Batches:']):
                print(line)
        
        if sim_process.returncode == 0:
            print("\n✅ Experiments completed successfully!")
        else:
            print(f"\n⚠️ Experiments finished with code {sim_process.returncode}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    finally:
        # Cleanup
        if web_process:
            print("\nStopping web server...")
            web_process.terminate()
            try:
                web_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                web_process.kill()
            print("✓ Web server stopped")

if __name__ == "__main__":
    sys.exit(main() or 0)
