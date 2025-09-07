#!/usr/bin/env python3
"""
Start a dedicated llama.cpp server for embeddings using Qwen3-Embedding model.
This runs on a different port (8002) from the main LLM server (8000).
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def start_embedding_server():
    """Start the embedding server with the Qwen3-Embedding model."""
    
    # Configuration
    EMBEDDING_MODEL_PATH = r"C:\models\Qwen3-Embedding-0.6B\qwen3-embedding-0.6b-q8_0.gguf"
    EMBEDDING_PORT = 8002
    
    # Find llama.cpp server
    llama_server_paths = [
        r"C:\Users\Jake\llama.cpp\build\bin\Release\llama-server.exe",
        r"C:\llama.cpp\build\bin\Release\llama-server.exe",
        "llama-server.exe",
        "llama-server"
    ]
    
    server_cmd = None
    for path in llama_server_paths:
        if Path(path).exists():
            server_cmd = path
            break
    
    if not server_cmd:
        print("Error: Could not find llama-server executable")
        print("Please ensure llama.cpp is built and accessible")
        return False
    
    print(f"Success: Found llama server at: {server_cmd}")
    
    # Build command for embedding server
    cmd = [
        server_cmd,
        "-m", EMBEDDING_MODEL_PATH,
        "--host", "0.0.0.0",
        "--port", str(EMBEDDING_PORT),
        "--embeddings",  # Enable embeddings mode
        "--pooling", "mean",  # Use mean pooling for OpenAI compatibility
        "-c", "2048",  # Smaller context for embeddings
        "-b", "2048",  # Batch size
        "-ub", "512",  # Micro batch
        "--n-gpu-layers", "-1",  # Use all GPU layers
        "--alias", "qwen3-embedding",
        "--no-mmap",  # Keep in memory
        "--mlock"  # Lock in memory
    ]
    
    print(f"Starting embedding server on port {EMBEDDING_PORT}")
    print(f"Model: {EMBEDDING_MODEL_PATH}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Start the server
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Wait for server to start
        print("Waiting for server to initialize...")
        time.sleep(5)
        
        # Check if server is running
        if process.poll() is None:
            print(f"Embedding server started successfully on port {EMBEDDING_PORT}")
            print(f"Embedding endpoint: http://localhost:{EMBEDDING_PORT}/v1/embeddings")
            print("\nPress Ctrl+C to stop the server")
            
            # Keep running and show output
            try:
                for line in process.stdout:
                    print(line.rstrip())
            except KeyboardInterrupt:
                print("\nStopping embedding server...")
                process.terminate()
                process.wait(timeout=5)
                print("Server stopped")
        else:
            print("Server failed to start")
            return False
            
    except Exception as e:
        print(f"Error starting server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_embedding_server()
