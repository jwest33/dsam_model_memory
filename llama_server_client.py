"""
LLM Server Manager - Handles automatic startup and shutdown of llama.cpp server
"""

import os
import subprocess
import time
import signal
import logging
import requests
import psutil
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import atexit
import platform

logger = logging.getLogger(__name__)


class LlamaServerClient:
    """Client for llama.cpp server API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """Initialize client with server URL."""
        self.base_url = base_url.rstrip('/')
        if not self.base_url.endswith('/v1'):
            self.base_url += '/v1'
        self.timeout = timeout
    
    def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            health_url = self.base_url.replace('/v1', '/health')
            response = requests.get(health_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def completion(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion from prompt."""
        url = f"{self.base_url}/completions"
        payload = {
            "prompt": prompt,
            "temperature": kwargs.get("temperature", 0.3),  # Default to lower temperature
            "max_tokens": kwargs.get("max_tokens", 100),
            "stop": kwargs.get("stop", []),
            "stream": False,
            "repeat_penalty": kwargs.get("repetition_penalty", 1.2)  # Default repetition penalty
        }
        
        # Add any additional parameters
        for key in ["top_p", "top_k", "repeat_penalty", "repetition_penalty", "presence_penalty", "frequency_penalty"]:
            if key in kwargs:
                if key == "repetition_penalty":
                    payload["repeat_penalty"] = kwargs[key]
                else:
                    payload[key] = kwargs[key]
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Completion request failed: {e}")
            raise
    
    def chat_completion(self, prompt: str, system_prompt: str = None, **kwargs) -> Dict[str, Any]:
        """Generate chat completion using the chat endpoint with proper template handling."""
        url = f"{self.base_url}/chat/completions"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.3),
            "max_tokens": kwargs.get("max_tokens", 100),
            "stop": kwargs.get("stop", []),
            "stream": False,
            "repeat_penalty": kwargs.get("repetition_penalty", 1.2)
        }
        
        # Add any additional parameters
        for key in ["top_p", "top_k", "repeat_penalty", "repetition_penalty", "presence_penalty", "frequency_penalty"]:
            if key in kwargs:
                if key == "repetition_penalty":
                    payload["repeat_penalty"] = kwargs[key]
                else:
                    payload[key] = kwargs[key]
        
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Chat completion request failed: {e}")
            raise


class LlamaServerModel:
    """Model interface compatible with llama-cpp-python API."""
    
    def __init__(self, server_url: str = "http://localhost:8000", verbose: bool = False, timeout: int = 30):
        """Initialize model interface."""
        self.client = LlamaServerClient(server_url, timeout)
        self.verbose = verbose
        
        # Check server availability
        if not self.client.health_check():
            raise ConnectionError(f"Cannot connect to llama server at {server_url}")
    
    def __call__(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate completion (compatible with llama-cpp-python)."""
        return self.client.completion(prompt, **kwargs)

@dataclass
class ServerConfig:
    """Configuration for llama.cpp server"""
    
    # Model settings
    model_path: str = r"C:\models\Qwen3-4B-Instruct-2507\Qwen3-4B-Instruct-2507-F16.gguf"
    model_alias: str = "qwen3-4b-instruct-2507-f16"
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    context_size: int = 10000
    n_gpu_layers: int = 35  # -1 for all layers on GPU
    threads: int = 4  # Reduced for better performance (usually physical cores / 2)
    
    # GPU/VRAM optimization settings
    gpu_split: Optional[str] = None  # e.g., "0.5,0.5" for multi-GPU
    main_gpu: int = 0  # Primary GPU for scratch buffers
    tensor_split: Optional[str] = None  # Manual tensor split for multi-GPU
    no_mmap: bool = False  # Disable memory mapping (keeps model in RAM/VRAM)
    lock_memory: bool = True  # Lock model in memory (prevent swapping)
    offload_kqv: bool = True  # Offload K,Q,V to GPU
    flash_attention: bool = False  # Disabled - may not be supported on all GPUs/builds
    batch_size: int = 2048  # Increased for better GPU utilization
    ubatch_size: int = 512  # Keep smaller for memory efficiency
    
    # Remote server settings (NEW)
    remote_host: Optional[str] = None
    remote_port: int = 8000
    use_remote: bool = False
    
    # Paths - can be overridden by environment variable
    if platform.system() == "Windows":
        # Default to searching in PATH or local directory
        server_executable = "llama-server.exe"
    else:
        server_executable = "llama-server"
    
    # Timeouts
    startup_timeout: int = 60
    shutdown_timeout: int = 10
    health_check_interval: float = 1.0
    
    @classmethod
    def from_env(cls) -> 'ServerConfig':
        """Create config from environment variables and ConfigManager"""
        config = cls()
        
        try:
            from agentic_memory.config import get_model_path, get_server_config, cfg
            
            # Get model path with OS handling
            model_path = get_model_path()
            if model_path and model_path != '.':  # Only use if valid path
                config.model_path = model_path
            
            # Get port from config
            server_cfg = get_server_config()
            config.port = server_cfg['llama_port']
            
            # Get model alias from config
            config.model_alias = cfg.llm_model
        except ImportError:
            # Fallback to environment variables if config not available
            pass
        
        # Check for startup timeout override
        if os.getenv("LLM_STARTUP_TIMEOUT"):
            config.startup_timeout = int(os.getenv("LLM_STARTUP_TIMEOUT"))
        
        # Check for remote server configuration (NEW)
        if os.getenv("LLM_SERVER_HOST"):
            config.use_remote = True
            config.remote_host = os.getenv("LLM_SERVER_HOST")
            config.remote_port = int(os.getenv("LLM_SERVER_PORT", str(config.port or 8000)))
            logger.info(f"Configured to use remote LLM server at {config.remote_host}:{config.remote_port}")
            return config
        
        # Original local configuration fallback
        if not config.model_path or config.model_path == '.':
            if os.getenv("LLM_MODEL_PATH"):
                model_path = os.getenv("LLM_MODEL_PATH")
                # Convert Unix-style paths to Windows paths on Windows
                if platform.system() == "Windows" and model_path.startswith("/c/"):
                    # Convert /c/path to C:\path
                    model_path = model_path.replace("/c/", "C:\\").replace("/", "\\")
                config.model_path = model_path
            # If still no valid path, keep the default from the dataclass
        if not config.model_alias and os.getenv("LLM_MODEL"):
            config.model_alias = os.getenv("LLM_MODEL")
        if config.port is None and os.getenv("LLM_PORT"):
            config.port = int(os.getenv("LLM_PORT"))
        elif config.port is None:
            config.port = 8000  # Default fallback
        if os.getenv("LLM_HOST"):
            config.host = os.getenv("LLM_HOST")
        
        # Context and threads
        if os.getenv("LLM_CONTEXT_SIZE"):
            config.context_size = int(os.getenv("LLM_CONTEXT_SIZE"))
        if os.getenv("LLM_THREADS"):
            config.threads = int(os.getenv("LLM_THREADS"))
        
        # GPU/VRAM settings from environment
        if os.getenv("LLM_GPU_LAYERS"):
            config.n_gpu_layers = int(os.getenv("LLM_GPU_LAYERS"))
        if os.getenv("LLM_LOCK_MEMORY"):
            config.lock_memory = os.getenv("LLM_LOCK_MEMORY").lower() in ("true", "1", "yes")
        if os.getenv("LLM_NO_MMAP"):
            config.no_mmap = os.getenv("LLM_NO_MMAP").lower() in ("true", "1", "yes")
        if os.getenv("LLM_OFFLOAD_KQV"):
            config.offload_kqv = os.getenv("LLM_OFFLOAD_KQV").lower() in ("true", "1", "yes")
        if os.getenv("LLM_FLASH_ATTENTION"):
            config.flash_attention = os.getenv("LLM_FLASH_ATTENTION").lower() in ("true", "1", "yes")
        if os.getenv("LLM_BATCH_SIZE"):
            config.batch_size = int(os.getenv("LLM_BATCH_SIZE"))
        if os.getenv("LLM_UBATCH_SIZE"):
            config.ubatch_size = int(os.getenv("LLM_UBATCH_SIZE"))
        if os.getenv("LLM_MAIN_GPU"):
            config.main_gpu = int(os.getenv("LLM_MAIN_GPU"))
        if os.getenv("LLM_GPU_SPLIT"):
            config.gpu_split = os.getenv("LLM_GPU_SPLIT")
        if os.getenv("LLM_TENSOR_SPLIT"):
            config.tensor_split = os.getenv("LLM_TENSOR_SPLIT")
        
        # Check for explicit server path from environment variable
        if os.getenv("LLAMA_SERVER_PATH"):
            config.server_executable = os.getenv("LLAMA_SERVER_PATH")
            logger.info(f"Using llama server from LLAMA_SERVER_PATH: {config.server_executable}")
        
        # Check for llama.cpp directory from environment variable
        elif os.getenv("LLAMA_CPP_PATH"):
            llama_cpp_path = os.getenv("LLAMA_CPP_PATH")
            if platform.system() == "Windows":
                server_path = os.path.join(llama_cpp_path, "build", "bin", "Release", "llama-server.exe")
            else:
                server_path = os.path.join(llama_cpp_path, "build", "bin", "llama-server")
            
            if os.path.exists(server_path):
                config.server_executable = server_path
                logger.info(f"Using llama server from LLAMA_CPP_PATH: {server_path}")
        
        # Auto-detect executable in common locations
        elif platform.system() == "Windows":
            # Check common relative paths (no hardcoded user paths)
            possible_paths = [
                # Local llama.cpp build
                "./llama.cpp/build/bin/Release/llama-server.exe",
                "./llama.cpp/build/bin/Release/server.exe",
                # Alternative build locations
                "./build/bin/Release/llama-server.exe",
                "./build/bin/Release/server.exe",
                # Current directory
                "./llama-server.exe",
                "./server.exe",
                "llama-server.exe",
                "server.exe"
            ]
            
            # Also check user's home directory if needed
            home = Path.home()
            possible_paths.extend([
                str(home / "llama.cpp/build/bin/Release/llama-server.exe"),
                str(home / "llama.cpp/build/bin/Release/server.exe")
            ])
            
            for exe_path in possible_paths:
                if Path(exe_path).exists():
                    config.server_executable = exe_path
                    logger.info(f"Found llama server at: {exe_path}")
                    break
        
        return config
    
    @property
    def base_url(self) -> str:
        """Get the base URL for the server"""
        if self.use_remote:
            return f"http://{self.remote_host}:{self.remote_port}/v1"
        return f"http://localhost:{self.port}/v1"
    
    @property
    def health_url(self) -> str:
        """Get the health check URL"""
        if self.use_remote:
            return f"http://{self.remote_host}:{self.remote_port}/health"
        return f"http://localhost:{self.port}/health"


class LLMServerManager:
    """Manages llama.cpp server lifecycle"""
    
    _instance = None
    _process = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one server instance"""
        if cls._instance is None:
            cls._instance = super(LLMServerManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config: Optional[ServerConfig] = None):
        """Initialize server manager"""
        if self._config is None:
            self._config = config or ServerConfig.from_env()
            if not self._config.use_remote:
                self._setup_shutdown_handler()
    
    def is_running(self) -> bool:
        """Check if server is currently running"""
        
        # If using remote server, check its health
        if self._config.use_remote:
            try:
                response = requests.get(self._config.health_url, timeout=5)
                return response.status_code == 200
            except Exception as e:
                logger.warning(f"Remote LLM server health check failed: {e}")
                return False
        
        # Original local server check
        if self._process and self._process.poll() is None:
            return True
        
        try:
            response = requests.get(self._config.health_url, timeout=1)
            return response.status_code == 200
        except:
            return False
    
    def start(self, force_restart: bool = False) -> bool:
        """
        Start the llama.cpp server (or verify remote server is accessible)
        """
        
        # If using remote server, just verify it's accessible
        if self._config.use_remote:
            logger.info(f"Using remote LLM server at {self._config.base_url}")
            if self.is_running():
                logger.info("Remote LLM server is accessible")
                return True
            else:
                logger.warning(f"Remote LLM server at {self._config.base_url} is not responding")
                logger.info("Waiting for remote server to become available...")
                
                # Wait for remote server to be ready
                start_time = time.time()
                while time.time() - start_time < self._config.startup_timeout:
                    if self.is_running():
                        logger.info("Remote LLM server is now accessible")
                        return True
                    time.sleep(self._config.health_check_interval)
                
                logger.error(f"Remote LLM server did not become available after {self._config.startup_timeout}s")
                return False
        
        # Original local server start code
        if self.is_running():
            if not force_restart:
                logger.info(f"LLM server already running at {self._config.base_url}")
                return True
            else:
                logger.info("Force restarting LLM server...")
                self.stop()
        
        # Check if model file exists
        model_path = Path(self._config.model_path)
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            logger.error("Please download the model or update the path in .env")
            return False
        
        # Check if server executable exists or is in PATH
        server_path = Path(self._config.server_executable)
        if not server_path.exists():
            # Check if it's in PATH
            import shutil
            if shutil.which(self._config.server_executable) is None:
                logger.error(f"Server executable not found: {self._config.server_executable}")
                logger.error("Please ensure llama.cpp server is installed (e.g., via winget install ggml.llamacpp)")
                return False
            # Use the command directly if it's in PATH
            server_cmd = self._config.server_executable
        else:
            server_cmd = str(server_path)
        
        # Build command
        cmd = [
            server_cmd,
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
        
        # Add VRAM optimization flags
        if self._config.lock_memory:
            cmd.append("--mlock")  # Lock model in memory
        
        if self._config.no_mmap:
            cmd.append("--no-mmap")  # Keep entire model in RAM/VRAM
        
        # Note: --offload-kqv might not be available in all versions
        # It's controlled by --n-gpu-layers in newer versions
        
        if self._config.flash_attention:
            cmd.append("--flash-attn")  # Use flash attention
        
        # Multi-GPU settings
        if self._config.main_gpu > 0:
            cmd.extend(["--main-gpu", str(self._config.main_gpu)])
        
        if self._config.gpu_split:
            cmd.extend(["--split-mode", "row"])  # Enable tensor splitting
            cmd.extend(["--tensor-split", self._config.gpu_split])
        
        if self._config.tensor_split:
            cmd.extend(["--tensor-split", self._config.tensor_split])
        
        logger.info(f"Starting LLM server: {' '.join(cmd)}")
        
        try:
            # Start the server process
            if platform.system() == "Windows":
                # Windows-specific: Create new process group
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                # Unix-like systems
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
            
            # Wait for server to be ready
            start_time = time.time()
            logger.info(f"Waiting for server to start (timeout: {self._config.startup_timeout}s)...")
            
            while time.time() - start_time < self._config.startup_timeout:
                if self._wait_for_health():
                    logger.info(f"LLM server started successfully at {self._config.base_url}")
                    return True
                
                # Check if process crashed
                if self._process.poll() is not None:
                    stdout, stderr = self._process.communicate()
                    logger.error(f"Server process terminated unexpectedly with code: {self._process.returncode}")
                    logger.error(f"stdout: {stdout.decode()[:1000]}")
                    logger.error(f"stderr: {stderr.decode()[:1000]}")
                    
                    # Check for common errors
                    stderr_text = stderr.decode()
                    if "file not found" in stderr_text.lower():
                        logger.error("Model file not found - check LLM_MODEL_PATH in .env")
                    elif "llama-server" in stderr_text.lower() and "not recognized" in stderr_text.lower():
                        logger.error("llama-server.exe not found - install with: winget install ggml.llamacpp")
                    
                    return False
                
                time.sleep(self._config.health_check_interval)
            
            logger.error(f"Server startup timeout after {self._config.startup_timeout}s")
            self.stop()
            return False
            
        except Exception as e:
            logger.error(f"Failed to start LLM server: {e}")
            return False
    
    def _wait_for_health(self) -> bool:
        """Wait for server health check to pass"""
        try:
            response = requests.get(self._config.health_url, timeout=2)
            if response.status_code == 200:
                # Also check model endpoint
                if self._config.use_remote:
                    models_url = f"http://{self._config.remote_host}:{self._config.remote_port}/v1/models"
                else:
                    models_url = f"http://localhost:{self._config.port}/v1/models"
                models_response = requests.get(models_url, timeout=2)
                return models_response.status_code == 200
        except:
            pass
        return False
    
    def stop(self) -> bool:
        """
        Stop the LLM server (no-op for remote servers)
        """
        
        # Don't stop remote servers
        if self._config.use_remote:
            logger.info("Using remote LLM server - not stopping")
            return True
        
        if not self._process:
            # Try to find and kill existing process
            existing_pid = self._find_existing_process()
            if existing_pid:
                try:
                    if platform.system() == "Windows":
                        subprocess.run(["taskkill", "/F", "/PID", str(existing_pid)], check=False)
                    else:
                        os.kill(existing_pid, signal.SIGTERM)
                    logger.info(f"Stopped existing LLM server (PID: {existing_pid})")
                    return True
                except Exception as e:
                    logger.error(f"Failed to stop existing server: {e}")
            return False
        
        logger.info("Stopping LLM server...")
        
        try:
            # Try graceful shutdown first
            if platform.system() == "Windows":
                self._process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
            
            # Wait for process to terminate
            try:
                self._process.wait(timeout=self._config.shutdown_timeout)
                logger.info("LLM server stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown failed
                logger.warning("Graceful shutdown timeout, force killing...")
                if platform.system() == "Windows":
                    self._process.kill()
                else:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                self._process.wait()
                logger.info("LLM server force stopped")
            
            self._process = None
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop LLM server: {e}")
            return False
    
    def restart(self) -> bool:
        """Restart the LLM server"""
        logger.info("Restarting LLM server...")
        self.stop()
        time.sleep(1)  # Brief pause between stop and start
        return self.start()
    
    def ensure_running(self) -> bool:
        """Ensure server is running, start if not"""
        if not self.is_running():
            return self.start()
        return True
    
    def _setup_shutdown_handler(self):
        """Register shutdown handler to stop server on exit"""
        def cleanup():
            if self._process and self._process.poll() is None:
                logger.info("Cleaning up LLM server on exit...")
                self.stop()
        atexit.register(cleanup)
    
    def _find_existing_process(self) -> Optional[int]:
        """Find existing llama server process by port"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # Check if process is llama server
                    if proc.info['name'] and 'llama' in proc.info['name'].lower():
                        # Check if it's using our port
                        cmdline = proc.info.get('cmdline', [])
                        if cmdline:
                            cmdline_str = ' '.join(cmdline)
                            if f"--port {self._config.port}" in cmdline_str or f"-p {self._config.port}" in cmdline_str:
                                return proc.info['pid']
                    
                    # Also check network connections for our port
                    connections = proc.connections(kind='inet')
                    for conn in connections:
                        if conn.laddr.port == self._config.port and conn.status == 'LISTEN':
                            return proc.info['pid']
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        except Exception as e:
            logger.debug(f"Error finding existing process: {e}")
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get current server status"""
        is_running = self.is_running()
        
        status = {
            "running": is_running,
            "url": self._config.base_url if is_running else None,
            "model": self._config.model_alias,
            "remote": self._config.use_remote,
        }
        
        if not self._config.use_remote:
            status["model_path"] = self._config.model_path
            status["port"] = self._config.port
        else:
            status["remote_host"] = self._config.remote_host
            status["remote_port"] = self._config.remote_port
        
        if is_running:
            try:
                models_response = requests.get(
                    f"{self._config.base_url.replace('/v1', '')}/v1/models",
                    timeout=2
                )
                if models_response.status_code == 200:
                    status["models"] = models_response.json()
            except:
                pass
        
        if self._process and not self._config.use_remote:
            status["pid"] = self._process.pid
        
        return status
    
    def __enter__(self):
        """Context manager entry"""
        self.ensure_running()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - optionally stop server"""
        # We don't stop here since other parts might need it
        pass


# Convenience functions
def get_server_manager() -> LLMServerManager:
    """Get the singleton server manager instance"""
    return LLMServerManager()


def ensure_llm_server() -> bool:
    """Ensure LLM server is running"""
    manager = get_server_manager()
    return manager.ensure_running()


def stop_llm_server() -> bool:
    """Stop the LLM server"""
    manager = get_server_manager()
    return manager.stop()


# CLI interface for testing
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    manager = LLMServerManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "start":
            if manager.start():
                print("✅ Server started")
                print(f"URL: {manager._config.base_url}")
                # Keep the server running if --detach not specified
                if "--detach" not in sys.argv:
                    print("\nServer is running. Press Ctrl+C to stop.")
                    try:
                        while True:
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print("\n\nShutting down server...")
                        manager.stop()
                        print("✅ Server stopped")
            else:
                print("❌ Failed to start server")
                sys.exit(1)
        
        elif command == "stop":
            if manager.stop():
                print("✅ Server stopped")
            else:
                print("❌ Failed to stop server")
        
        elif command == "restart":
            if manager.restart():
                print("✅ Server restarted")
            else:
                print("❌ Failed to restart server")
        
        elif command == "status":
            status = manager.get_status()
            print(f"Status: {'Running' if status['running'] else 'Stopped'}")
            if status['running']:
                print(f"URL: {status.get('url')}")
                print(f"Model: {status.get('model')}")
                if 'pid' in status:
                    print(f"PID: {status['pid']}")
        
        else:
            print(f"Unknown command: {command}")
            print("Usage: python llama_server_client.py [start|stop|restart|status]")
            print("\nCommands:")
            print("  start         - Start the server (keeps running, Ctrl+C to stop)")
            print("  start --detach - Start the server in background")
            print("  stop          - Stop the server")
            print("  restart       - Restart the server")
            print("  status        - Check server status")
    
    else:
        print("LLM Server Manager")
        print("Usage: python llama_server_client.py [start|stop|restart|status]")
        print("\nCommands:")
        print("  start         - Start the server (keeps running, Ctrl+C to stop)")
        print("  start --detach - Start the server in background")
        print("  stop          - Stop the server")
        print("  restart       - Restart the server")
        print("  status        - Check server status")
        print("\nCurrent status:")
        status = manager.get_status()
        print(f"  {'Running' if status['running'] else 'Stopped'}")
        if status['running']:
            print(f"  URL: {status.get('url')}")
