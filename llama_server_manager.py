"""
Unified LLama.cpp Server Manager - Handles both LLM and Embedding servers
"""

import os
import subprocess
import time
import signal
import logging
import requests
import psutil
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
import atexit
import platform

logger = logging.getLogger(__name__)


@dataclass
class BaseLlamaServerConfig:
    """Base configuration for llama.cpp servers."""
    # Server executable settings
    server_executable: Optional[str] = None
    server_path: Optional[str] = None
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    context_size: int = 2048
    threads: int = 4
    
    # GPU/VRAM optimization settings
    n_gpu_layers: int = -1  # Use ALL layers on GPU
    no_mmap: bool = True  # Keep entire model in VRAM
    lock_memory: bool = True  # Lock model in memory
    batch_size: int = 2048
    ubatch_size: int = 512
    
    # Model settings (to be overridden)
    model_path: str = ""
    model_alias: str = ""
    
    # Timeouts
    startup_timeout: int = 60
    shutdown_timeout: int = 10
    health_check_interval: float = 1.0


@dataclass
class LLMServerConfig(BaseLlamaServerConfig):
    """Configuration for LLM inference server."""
    # Model settings
    model_path: str = r"C:\models\Qwen3-4B-Instruct-2507\Qwen3-4B-Instruct-2507-F16.gguf"
    model_alias: str = "qwen3-4b-instruct"
    port: int = 8000
    context_size: int = 10000
    batch_size: int = 4096
    ubatch_size: int = 1024
    
    # Performance optimizations
    continuous_batching: bool = True
    parallel_sequences: int = 8
    offload_kqv: bool = True  # Offload K,Q,V to GPU
    flash_attention: bool = False  # May not be supported on all GPUs
    
    @classmethod
    def from_env(cls) -> 'LLMServerConfig':
        """Create config from environment variables."""
        config = cls()
        
        # Model path
        if os.getenv("AM_MODEL_PATH") or os.getenv("LLM_MODEL_PATH"):
            config.model_path = os.getenv("AM_MODEL_PATH", os.getenv("LLM_MODEL_PATH"))
        
        # Server settings
        if os.getenv("AM_LLAMA_PORT"):
            config.port = int(os.getenv("AM_LLAMA_PORT"))
        if os.getenv("AM_CONTEXT_WINDOW"):
            config.context_size = int(os.getenv("AM_CONTEXT_WINDOW"))
        if os.getenv("AM_GPU_LAYERS"):
            config.n_gpu_layers = int(os.getenv("AM_GPU_LAYERS"))
        if os.getenv("AM_LLAMA_BATCH_SIZE"):
            config.batch_size = int(os.getenv("AM_LLAMA_BATCH_SIZE"))
        if os.getenv("AM_LLAMA_UBATCH_SIZE"):
            config.ubatch_size = int(os.getenv("AM_LLAMA_UBATCH_SIZE"))
        if os.getenv("AM_CONTINUOUS_BATCHING"):
            config.continuous_batching = os.getenv("AM_CONTINUOUS_BATCHING").lower() in ("true", "1", "yes")
        if os.getenv("AM_PARALLEL_SEQUENCES"):
            config.parallel_sequences = int(os.getenv("AM_PARALLEL_SEQUENCES"))
            
        return config


@dataclass
class EmbeddingServerConfig(BaseLlamaServerConfig):
    """Configuration for embedding server."""
    # Model settings
    model_path: str = r"C:\models\Qwen3-Embedding-0.6B\qwen3-embedding-0.6b-q8_0.gguf"
    model_alias: str = "qwen3-embedding"
    port: int = 8002
    context_size: int = 2048  # Smaller context for embeddings
    batch_size: int = 2048
    ubatch_size: int = 512
    
    # Embedding specific settings
    pooling_type: str = "mean"  # Required for OpenAI-compatible embeddings
    embedding_enabled: bool = True
    
    @classmethod
    def from_env(cls) -> 'EmbeddingServerConfig':
        """Create config from environment variables."""
        config = cls()
        
        # Model path
        if os.getenv("AM_EMBEDDING_MODEL_PATH"):
            config.model_path = os.getenv("AM_EMBEDDING_MODEL_PATH")
        
        # Port (default to 8002 for embedding server)
        if os.getenv("AM_EMBEDDING_PORT"):
            config.port = int(os.getenv("AM_EMBEDDING_PORT"))
            
        # Pooling type
        if os.getenv("AM_EMBEDDING_POOLING"):
            config.pooling_type = os.getenv("AM_EMBEDDING_POOLING")
            
        return config


class BaseLlamaServerManager:
    """Base class for managing llama.cpp server processes."""
    
    def __init__(self, config: BaseLlamaServerConfig, server_type: str = "base"):
        """Initialize with configuration."""
        self._config = config
        self._process = None
        self._log_file = None
        self._started = False
        self.server_type = server_type
        
        # Register cleanup handler
        atexit.register(self.stop)
    
    def _find_server_executable(self) -> Optional[Path]:
        """Find the llama.cpp server executable."""
        # Check if explicitly set
        if self._config.server_executable:
            path = Path(self._config.server_executable)
            if path.exists():
                return path
        
        # Common paths to check
        search_paths = [
            r"C:\Users\Jake\llama.cpp\build\bin\Release\llama-server.exe",
            r"C:\llama.cpp\build\bin\Release\llama-server.exe",
            "./llama-server.exe",
            "./llama-server",
            "llama-server.exe",
            "llama-server"
        ]
        
        # Add config path if specified
        if self._config.server_path:
            search_paths.insert(0, self._config.server_path)
        
        # Search for executable
        for path_str in search_paths:
            path = Path(path_str)
            if path.exists():
                logger.info(f"Found llama server at: {path}")
                return path
        
        # Try to find in PATH
        import shutil
        llama_path = shutil.which("llama-server") or shutil.which("llama-server.exe")
        if llama_path:
            return Path(llama_path)
        
        return None
    
    def _build_command(self, server_path: Path, model_path: Path) -> list:
        """Build the server command with all parameters."""
        cmd = [
            str(server_path),
            "-m", str(model_path),
            "--host", self._config.host,
            "--port", str(self._config.port),
            "-c", str(self._config.context_size),
            "-t", str(self._config.threads),
            "--alias", self._config.model_alias,
            "-b", str(self._config.batch_size),
            "-ub", str(self._config.ubatch_size)
        ]
        
        # Add GPU layers if specified
        if self._config.n_gpu_layers != 0:
            cmd.extend(["--n-gpu-layers", str(self._config.n_gpu_layers)])
        
        # Add memory optimization flags
        if self._config.lock_memory:
            cmd.append("--mlock")
        if self._config.no_mmap:
            cmd.append("--no-mmap")
        
        # Add embedding-specific settings
        if hasattr(self._config, 'embedding_enabled') and self._config.embedding_enabled:
            cmd.append("--embeddings")
            if hasattr(self._config, 'pooling_type'):
                cmd.extend(["--pooling", self._config.pooling_type])
        
        # Add LLM-specific settings
        if hasattr(self._config, 'continuous_batching') and self._config.continuous_batching:
            cmd.append("--cont-batching")
        if hasattr(self._config, 'parallel_sequences'):
            cmd.extend(["--parallel", str(self._config.parallel_sequences)])
        if hasattr(self._config, 'flash_attention') and self._config.flash_attention:
            cmd.append("--flash-attn")
        
        return cmd
    
    def _kill_existing_server(self) -> bool:
        """Kill any existing server on the configured port."""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    # Get connections using new method name
                    connections = proc.net_connections()
                    for conn in connections:
                        if hasattr(conn, 'laddr') and conn.laddr.port == self._config.port:
                            logger.info(f"Found existing process on port {self._config.port}, terminating...")
                            proc.terminate()
                            proc.wait(timeout=5)
                            return True
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
                    continue
        except Exception as e:
            logger.warning(f"Error checking for existing processes: {e}")
        return False
    
    def is_running(self) -> bool:
        """Check if server is running."""
        if self._process and self._process.poll() is None:
            return True
        
        # Also check if server is accessible
        try:
            response = requests.get(f"http://localhost:{self._config.port}/health", timeout=1)
            return response.status_code == 200
        except:
            return False
    
    def start(self) -> bool:
        """Start the llama.cpp server."""
        if self.is_running():
            logger.info(f"{self.server_type} server already running on port {self._config.port}")
            return True
        
        # Kill any existing server on the port
        self._kill_existing_server()
        time.sleep(1)
        
        # Find server executable
        server_path = self._find_server_executable()
        if not server_path:
            logger.error("Could not find llama-server executable")
            return False
        
        # Check model exists
        model_path = Path(self._config.model_path)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Build command
        cmd = self._build_command(server_path, model_path)
        
        logger.info(f"Starting {self.server_type} server: {' '.join(cmd)}")
        
        try:
            # Create log file
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            self._log_file = open(log_dir / f"llama_{self.server_type}_{self._config.port}.log", "w")
            
            # Start process
            if platform.system() == "Windows":
                self._process = subprocess.Popen(
                    cmd,
                    stdout=self._log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                self._process = subprocess.Popen(
                    cmd,
                    stdout=self._log_file,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid
                )
            
            # Wait for server to be ready
            logger.info(f"Waiting for {self.server_type} server to start (timeout: {self._config.startup_timeout}s)...")
            start_time = time.time()
            
            while time.time() - start_time < self._config.startup_timeout:
                if self._process.poll() is not None:
                    logger.error(f"{self.server_type} server process died during startup")
                    return False
                
                try:
                    response = requests.get(f"http://localhost:{self._config.port}/health", timeout=2)
                    if response.status_code == 200:
                        self._started = True
                        logger.info(f"✅ {self.server_type} server started successfully on port {self._config.port}")
                        return True
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(self._config.health_check_interval)
            
            logger.error(f"{self.server_type} server startup timeout after {self._config.startup_timeout}s")
            self.stop()
            return False
            
        except Exception as e:
            logger.error(f"Failed to start {self.server_type} server: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the llama.cpp server."""
        if not self._process:
            return True
        
        logger.info(f"Stopping {self.server_type} server...")
        
        try:
            if platform.system() == "Windows":
                self._process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            
            self._process.wait(timeout=self._config.shutdown_timeout)
            logger.info(f"{self.server_type} server stopped gracefully")
            
        except subprocess.TimeoutExpired:
            logger.warning(f"{self.server_type} server didn't stop gracefully, forcing...")
            self._process.kill()
            self._process.wait()
        except Exception as e:
            logger.error(f"Error stopping {self.server_type} server: {e}")
        finally:
            self._process = None
            if self._log_file:
                self._log_file.close()
                self._log_file = None
            self._started = False
        
        return True
    
    def restart(self) -> bool:
        """Restart the server."""
        logger.info(f"Restarting {self.server_type} server...")
        self.stop()
        time.sleep(2)
        return self.start()


class LLMServerManager(BaseLlamaServerManager):
    """Manager for LLM inference server with backward compatibility."""
    
    def __init__(self, config: Optional[LLMServerConfig] = None):
        """Initialize LLM server manager."""
        config = config or LLMServerConfig.from_env()
        super().__init__(config, server_type="LLM")
    
    def ensure_running(self) -> bool:
        """Ensure server is running (backward compatibility)."""
        if self.is_running():
            return True
        return self.start()
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status (backward compatibility)."""
        return {
            'running': self.is_running(),
            'port': self._config.port,
            'model': self._config.model_path
        }


class EmbeddingServerManager(BaseLlamaServerManager):
    """Manager for embedding server."""
    
    def __init__(self, config: Optional[EmbeddingServerConfig] = None):
        """Initialize embedding server manager."""
        config = config or EmbeddingServerConfig.from_env()
        super().__init__(config, server_type="Embedding")


# Backward compatibility functions
def get_server_manager() -> LLMServerManager:
    """Get the default LLM server manager (for backward compatibility)."""
    return LLMServerManager()

def get_embedding_manager() -> EmbeddingServerManager:
    """Get the default embedding server manager."""
    return EmbeddingServerManager()


# Convenience functions for command-line usage
def main():
    """Command-line interface for server management."""
    import sys
    import time
    
    if len(sys.argv) < 3:
        print("Usage: python llama_server_manager.py <server_type> <command>")
        print("  server_type: llm | embedding | both")
        print("  command: start | stop | restart | status")
        sys.exit(1)
    
    server_type = sys.argv[1].lower()
    command = sys.argv[2].lower()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Handle different server types
    if server_type == "llm":
        manager = LLMServerManager()
    elif server_type == "embedding":
        manager = EmbeddingServerManager()
    elif server_type == "both":
        llm_manager = LLMServerManager()
        emb_manager = EmbeddingServerManager()
        
        if command == "start":
            llm_success = llm_manager.start()
            emb_success = emb_manager.start()
            if llm_success and emb_success:
                print("\nBoth servers started successfully!")
                print("LLM server: http://localhost:8000")
                print("Embedding server: http://localhost:8002")
                print("\nPress Ctrl+C to stop the servers...")
                try:
                    # Keep the process running
                    while True:
                        time.sleep(1)
                        # Check if servers are still running
                        if not llm_manager.is_running() or not emb_manager.is_running():
                            print("One or both servers stopped unexpectedly")
                            break
                except KeyboardInterrupt:
                    print("\nShutting down servers...")
                    llm_manager.stop()
                    emb_manager.stop()
                    print("Servers stopped")
            sys.exit(0 if (llm_success and emb_success) else 1)
        elif command == "stop":
            llm_manager.stop()
            emb_manager.stop()
            sys.exit(0)
        elif command == "restart":
            llm_success = llm_manager.restart()
            emb_success = emb_manager.restart()
            sys.exit(0 if (llm_success and emb_success) else 1)
        elif command == "status":
            llm_running = llm_manager.is_running()
            emb_running = emb_manager.is_running()
            print(f"LLM Server: {'Running' if llm_running else 'Stopped'}")
            print(f"Embedding Server: {'Running' if emb_running else 'Stopped'}")
            sys.exit(0)
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    else:
        print(f"Unknown server type: {server_type}")
        sys.exit(1)
    
    # Handle single server commands
    if command == "start":
        success = manager.start()
        if success:
            print(f"\n{server_type.upper()} server started successfully!")
            print(f"Server endpoint: http://localhost:{manager._config.port}")
            print("\nPress Ctrl+C to stop the server...")
            try:
                # Keep the process running
                while True:
                    time.sleep(1)
                    if not manager.is_running():
                        print("Server stopped unexpectedly")
                        break
            except KeyboardInterrupt:
                print("\nShutting down server...")
                manager.stop()
                print("Server stopped")
        sys.exit(0 if success else 1)
    elif command == "stop":
        manager.stop()
        sys.exit(0)
    elif command == "restart":
        success = manager.restart()
        sys.exit(0 if success else 1)
    elif command == "status":
        running = manager.is_running()
        print(f"{server_type.upper()} Server: {'Running' if running else 'Stopped'}")
        sys.exit(0)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
