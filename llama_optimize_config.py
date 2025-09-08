#!/usr/bin/env python3
"""
Optimized configuration and keep-alive mechanism for llama.cpp servers.
Keeps models "hot" in memory to eliminate warmup time between requests.
"""

import os
import sys
import time
import requests
import threading
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaKeepAlive:
    """Keeps llama.cpp models warm by sending periodic dummy requests."""
    
    def __init__(self, llm_port: int = 8000, embedding_port: int = 8002):
        self.llm_port = llm_port
        self.embedding_port = embedding_port
        self.running = False
        self.thread = None
        
    def start(self, interval: int = 30):
        """Start the keep-alive thread.
        
        Args:
            interval: Seconds between keep-alive requests (default: 30)
        """
        if self.running:
            logger.info("Keep-alive already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._keep_alive_loop, args=(interval,), daemon=True)
        self.thread.start()
        logger.info(f"Started keep-alive with {interval}s interval")
        
    def stop(self):
        """Stop the keep-alive thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Stopped keep-alive")
        
    def _keep_alive_loop(self, interval: int):
        """Main keep-alive loop."""
        while self.running:
            try:
                # Keep LLM warm with a simple completion
                self._warm_llm()
                
                # Keep embedding model warm
                self._warm_embeddings()
                
            except Exception as e:
                logger.debug(f"Keep-alive error (normal if server restarting): {e}")
            
            # Sleep in small increments so we can stop quickly
            for _ in range(interval):
                if not self.running:
                    break
                time.sleep(1)
    
    def _warm_llm(self):
        """Send a minimal request to keep LLM model warm."""
        try:
            url = f"http://localhost:{self.llm_port}/v1/chat/completions"
            payload = {
                "model": "qwen3-4b-instruct",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 1,
                "temperature": 0
            }
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.debug("LLM keep-alive successful")
        except:
            pass  # Silently ignore if server is down
    
    def _warm_embeddings(self):
        """Send a minimal request to keep embedding model warm."""
        try:
            url = f"http://localhost:{self.embedding_port}/v1/embeddings"
            payload = {"input": ["test"]}
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                logger.debug("Embedding keep-alive successful")
        except:
            pass  # Silently ignore if server is down


def optimize_llama_servers():
    """
    Set optimal environment variables for llama.cpp servers.
    Call this before starting the servers.
    """
    optimizations = {
        # Keep models fully loaded in VRAM
        "AM_GPU_LAYERS": "-1",  # Use all GPU layers
        "AM_NO_MMAP": "true",  # Keep entire model in memory
        "AM_LOCK_MEMORY": "true",  # Lock model in RAM/VRAM
        
        # Optimize batching for throughput
        "AM_LLAMA_BATCH_SIZE": "4096",  # Large batch for better GPU utilization
        "AM_LLAMA_UBATCH_SIZE": "1024",  # Micro-batch size
        "AM_CONTINUOUS_BATCHING": "true",  # Process multiple requests efficiently
        "AM_PARALLEL_SEQUENCES": "8",  # Handle multiple sequences in parallel
        
        # Cache optimization
        "AM_CACHE_TYPE": "f16",  # Use FP16 for cache (faster, less memory)
        "AM_CACHE_REUSE": "true",  # Reuse cache across requests when possible
        
        # Thread optimization (adjust based on your CPU)
        "AM_THREADS": str(os.cpu_count() or 4),
        
        # Flash attention (if supported by your GPU)
        "AM_FLASH_ATTENTION": "false",  # Set to "true" if you have a newer GPU
    }
    
    for key, value in optimizations.items():
        if not os.getenv(key):  # Don't override existing settings
            os.environ[key] = value
            logger.info(f"Set {key}={value}")
    
    logger.info("Optimization settings applied")


def start_optimized_servers():
    """Start servers with optimizations and keep-alive."""
    from llama_server_manager import LLMServerManager, EmbeddingServerManager
    
    # Apply optimizations
    optimize_llama_servers()
    
    # Start servers
    llm_manager = LLMServerManager()
    emb_manager = EmbeddingServerManager()
    
    llm_success = llm_manager.start()
    emb_success = emb_manager.start()
    
    if not (llm_success and emb_success):
        print("Failed to start one or both servers")
        return None, None
    
    print("Both servers started successfully!")
    print("LLM server: http://localhost:8000")
    print("Embedding server: http://localhost:8002")
    
    # Start keep-alive
    keep_alive = LlamaKeepAlive()
    keep_alive.start(interval=30)  # Keep warm every 30 seconds
    
    return (llm_manager, emb_manager), keep_alive


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize llama.cpp servers')
    parser.add_argument('--keep-alive-only', action='store_true',
                       help='Only start keep-alive (servers already running)')
    parser.add_argument('--interval', type=int, default=30,
                       help='Keep-alive interval in seconds (default: 30)')
    
    args = parser.parse_args()
    
    if args.keep_alive_only:
        # Just start keep-alive for existing servers
        keep_alive = LlamaKeepAlive()
        keep_alive.start(interval=args.interval)
        print(f"Keep-alive started with {args.interval}s interval")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            keep_alive.stop()
            print("\nKeep-alive stopped")
    else:
        # Start servers with optimizations
        print("Starting optimized servers with keep-alive...")
        managers, keep_alive = start_optimized_servers()
        
        if managers is None:
            print("Failed to start servers")
            sys.exit(1)
        
        llm_manager, emb_manager = managers
        
        print("\nServers running with optimizations:")
        print("- Model locked in VRAM (no reload between requests)")
        print("- Continuous batching enabled")
        print("- Keep-alive warming models every 30 seconds")
        print("\nPress Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            keep_alive.stop()
            llm_manager.stop()
            emb_manager.stop()
            print("\nServers stopped")


if __name__ == "__main__":
    main()
