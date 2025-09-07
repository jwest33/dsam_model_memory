"""
Enhanced CLI module for JAM with API server capabilities
"""

import os
import sys
import json
import asyncio
import logging
import subprocess
import signal
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import click
import psutil
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_memory.router import MemoryRouter
from agentic_memory.config_manager import ConfigManager
from llama_server_client import LLMServerManager, get_server_manager
from llama_api import create_app, APIConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProcessManager:
    """Manages background processes for servers"""
    
    def __init__(self):
        self.processes = {}
        self.pid_dir = Path.home() / ".jam" / "pids"
        self.pid_dir.mkdir(parents=True, exist_ok=True)
    
    def start_process(self, name: str, command: List[str], cwd: Optional[Path] = None) -> bool:
        """Start a background process"""
        pid_file = self.pid_dir / f"{name}.pid"
        
        # Check if already running
        if self.is_running(name):
            logger.info(f"{name} is already running")
            return True
        
        try:
            # Start process
            if sys.platform == "win32":
                # Windows: Use CREATE_NEW_PROCESS_GROUP for background
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
                    cwd=cwd
                )
            else:
                # Unix: Use nohup equivalent
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid,
                    cwd=cwd
                )
            
            # Save PID
            pid_file.write_text(str(process.pid))
            self.processes[name] = process
            logger.info(f"Started {name} with PID {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
            return False
    
    def stop_process(self, name: str) -> bool:
        """Stop a background process"""
        pid_file = self.pid_dir / f"{name}.pid"
        
        if not pid_file.exists():
            logger.info(f"No PID file found for {name}")
            return False
        
        try:
            pid = int(pid_file.read_text())
            
            # Try to terminate process
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False)
            else:
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
                # Force kill if still running
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            
            # Clean up PID file
            pid_file.unlink()
            logger.info(f"Stopped {name} (PID {pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop {name}: {e}")
            return False
    
    def is_running(self, name: str) -> bool:
        """Check if a process is running"""
        pid_file = self.pid_dir / f"{name}.pid"
        
        if not pid_file.exists():
            return False
        
        try:
            pid = int(pid_file.read_text())
            # Check if process exists
            return psutil.pid_exists(pid)
        except:
            return False
    
    def get_status(self, name: str) -> Dict[str, Any]:
        """Get process status"""
        pid_file = self.pid_dir / f"{name}.pid"
        
        if not pid_file.exists():
            return {"running": False}
        
        try:
            pid = int(pid_file.read_text())
            if psutil.pid_exists(pid):
                proc = psutil.Process(pid)
                return {
                    "running": True,
                    "pid": pid,
                    "cpu_percent": proc.cpu_percent(),
                    "memory_mb": proc.memory_info().rss / 1024 / 1024,
                    "create_time": datetime.fromtimestamp(proc.create_time()).isoformat()
                }
        except:
            pass
        
        return {"running": False}


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """JAM - Journalistic Agent Memory CLI"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.group()
def server():
    """Manage servers (LLM, API, Web)"""
    pass


@server.command()
@click.option('--all', 'start_all', is_flag=True, help='Start all servers')
@click.option('--llm', is_flag=True, help='Start LLM server')
@click.option('--api', is_flag=True, help='Start API wrapper')
@click.option('--web', is_flag=True, help='Start web interface')
@click.option('--daemon', '-d', is_flag=True, help='Run in background')
def start(start_all, llm, api, web, daemon):
    """Start servers"""
    process_manager = ProcessManager()
    
    if start_all:
        llm = api = web = True
    
    if not (llm or api or web):
        click.echo("Please specify which servers to start (--llm, --api, --web, or --all)")
        return
    
    # Start LLM server
    if llm:
        click.echo("Starting LLM server...")
        manager = get_server_manager()
        if manager.ensure_running():
            click.echo("LLM server started")
        else:
            click.echo("Failed to start LLM server")
            return
    
    # Start API wrapper
    if api:
        click.echo("Starting API wrapper...")
        if daemon:
            # Run in background
            command = [sys.executable, "-m", "llama_api", "start"]
            if process_manager.start_process("jam-api", command):
                click.echo("API wrapper started in background")
            else:
                click.echo("Failed to start API wrapper")
        else:
            # Run in foreground
            import uvicorn
            from llama_api import create_app, APIConfig
            
            config = APIConfig()
            app = create_app(config)
            click.echo(f"Starting API wrapper on {config.host}:{config.port}")
            uvicorn.run(app, host=config.host, port=config.port)
    
    # Start web interface
    if web:
        click.echo("Starting web interface...")
        if daemon:
            command = [sys.executable, "-m", "agentic_memory.server.flask_app"]
            if process_manager.start_process("jam-web", command):
                click.echo("Web interface started in background")
            else:
                click.echo("Failed to start web interface")
        else:
            # Run in foreground
            from agentic_memory.server.flask_app import app
            click.echo("Starting web interface on port 5001")
            app.run(host="0.0.0.0", port=5001)


@server.command()
@click.option('--all', 'stop_all', is_flag=True, help='Stop all servers')
@click.option('--llm', is_flag=True, help='Stop LLM server')
@click.option('--api', is_flag=True, help='Stop API wrapper')
@click.option('--web', is_flag=True, help='Stop web interface')
def stop(stop_all, llm, api, web):
    """Stop servers"""
    process_manager = ProcessManager()
    
    if stop_all:
        llm = api = web = True
    
    if not (llm or api or web):
        click.echo("Please specify which servers to stop (--llm, --api, --web, or --all)")
        return
    
    # Stop web interface
    if web:
        click.echo("Stopping web interface...")
        if process_manager.stop_process("jam-web"):
            click.echo("Web interface stopped")
        else:
            click.echo("Web interface was not running")
    
    # Stop API wrapper
    if api:
        click.echo("Stopping API wrapper...")
        if process_manager.stop_process("jam-api"):
            click.echo("API wrapper stopped")
        else:
            click.echo("API wrapper was not running")
    
    # Stop LLM server
    if llm:
        click.echo("Stopping LLM server...")
        manager = get_server_manager()
        if manager.stop():
            click.echo("LLM server stopped")
        else:
            click.echo("LLM server was not running")


@server.command()
def status():
    """Check server status"""
    process_manager = ProcessManager()
    
    # LLM server status
    manager = get_server_manager()
    llm_status = manager.get_status()
    
    click.echo("\nServer Status:")
    click.echo("-" * 40)
    
    # LLM Server
    if llm_status["running"]:
        click.echo(f"LLM Server: Running")
        click.echo(f"   URL: {llm_status.get('url')}")
        click.echo(f"   Model: {llm_status.get('model')}")
        if 'pid' in llm_status:
            click.echo(f"   PID: {llm_status['pid']}")
    else:
        click.echo("LLM Server: Stopped")
    
    # API Wrapper
    api_status = process_manager.get_status("jam-api")
    if api_status["running"]:
        click.echo(f"API Wrapper: Running")
        click.echo(f"   PID: {api_status['pid']}")
        click.echo(f"   CPU: {api_status['cpu_percent']:.1f}%")
        click.echo(f"   Memory: {api_status['memory_mb']:.1f} MB")
    else:
        click.echo("API Wrapper: Stopped")
    
    # Web Interface
    web_status = process_manager.get_status("jam-web")
    if web_status["running"]:
        click.echo(f"Web Interface: Running")
        click.echo(f"   PID: {web_status['pid']}")
        click.echo(f"   CPU: {web_status['cpu_percent']:.1f}%")
        click.echo(f"   Memory: {web_status['memory_mb']:.1f} MB")
    else:
        click.echo("Web Interface: Stopped")
    
    click.echo("-" * 40)


@server.command()
def restart():
    """Restart all servers"""
    click.echo("Restarting servers...")
    
    # Stop all
    ctx = click.get_current_context()
    ctx.invoke(stop, stop_all=True)
    time.sleep(2)
    
    # Start all
    ctx.invoke(start, start_all=True, daemon=True)


@cli.group()
def memory():
    """Manage memory operations"""
    pass


@memory.command()
@click.argument('text')
@click.option('--actor', default='user', help='Actor/source of the memory')
def add(text, actor):
    """Add a memory"""
    config = ConfigManager()
    router = MemoryRouter(config)
    
    result = router.ingest(text, metadata={'actor': actor})
    if result:
        click.echo(f"Memory added: {result.id}")
    else:
        click.echo("Failed to add memory")


@memory.command()
@click.argument('query')
@click.option('--limit', default=5, help='Number of results')
@click.option('--format', 'output_format', type=click.Choice(['json', 'text']), default='text')
def search(query, limit, output_format):
    """Search memories"""
    config = ConfigManager()
    router = MemoryRouter(config)
    
    results = router.retrieve(query, limit=limit)
    
    if output_format == 'json':
        click.echo(json.dumps([r.to_dict() for r in results], indent=2))
    else:
        for i, result in enumerate(results, 1):
            click.echo(f"\n{i}. {result.what}")
            click.echo(f"   Who: {result.who}")
            click.echo(f"   When: {result.when}")
            if result.where:
                click.echo(f"   Where: {result.where}")
            click.echo(f"   Score: {result.score:.3f}")


@memory.command()
def stats():
    """Show memory statistics"""
    config = ConfigManager()
    router = MemoryRouter(config)
    
    stats = router.get_stats()
    
    click.echo("\nMemory Statistics:")
    click.echo("-" * 40)
    click.echo(f"Total memories: {stats.get('total_memories', 0)}")
    click.echo(f"Unique actors: {stats.get('unique_actors', 0)}")
    click.echo(f"Date range: {stats.get('date_range', 'N/A')}")
    click.echo(f"Database size: {stats.get('db_size_mb', 0):.2f} MB")
    click.echo(f"Index size: {stats.get('index_size_mb', 0):.2f} MB")


@cli.group()
def api():
    """API client commands"""
    pass


@api.command()
@click.argument('prompt')
@click.option('--url', default='http://localhost:8001', help='API wrapper URL')
@click.option('--max-tokens', default=100, help='Maximum tokens to generate')
@click.option('--temperature', default=0.3, help='Temperature for sampling')
def complete(prompt, url, max_tokens, temperature):
    """Send completion request to API"""
    import requests
    
    try:
        response = requests.post(
            f"{url}/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        )
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and result['choices']:
            click.echo(result['choices'][0]['text'])
        else:
            click.echo(json.dumps(result, indent=2))
            
    except Exception as e:
        click.echo(f"Error: {e}")


@api.command()
@click.argument('message')
@click.option('--url', default='http://localhost:8001', help='API wrapper URL')
@click.option('--system', help='System prompt')
@click.option('--max-tokens', default=100, help='Maximum tokens to generate')
def chat(message, url, system, max_tokens):
    """Send chat request to API"""
    import requests
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": message})
    
    try:
        response = requests.post(
            f"{url}/chat/completions",
            json={
                "messages": messages,
                "max_tokens": max_tokens
            }
        )
        response.raise_for_status()
        result = response.json()
        
        if 'choices' in result and result['choices']:
            click.echo(result['choices'][0]['message']['content'])
        else:
            click.echo(json.dumps(result, indent=2))
            
    except Exception as e:
        click.echo(f"Error: {e}")


@api.command()
@click.option('--url', default='http://localhost:8001', help='API wrapper URL')
def health(url):
    """Check API health"""
    import requests
    
    try:
        response = requests.get(f"{url}/health")
        response.raise_for_status()
        result = response.json()
        
        click.echo(f"Status: {result['status']}")
        click.echo(f"LLM Server: {result['llama_server']}")
        click.echo(f"Uptime: {result['api_wrapper']['uptime_seconds']:.0f} seconds")
        click.echo(f"Requests: {result['api_wrapper']['requests_processed']}")
        
    except Exception as e:
        click.echo(f"API is not responding: {e}")


@cli.command()
def config():
    """Show configuration"""
    config = ConfigManager()
    
    click.echo("\nConfiguration:")
    click.echo("-" * 40)
    click.echo(f"LLM Base URL: {config.llm_base_url}")
    click.echo(f"LLM Model: {config.llm_model}")
    click.echo(f"Context Window: {config.context_window}")
    click.echo(f"Database: {config.db_path}")
    click.echo(f"Index: {config.index_path}")
    click.echo("\nRetrieval Weights:")
    click.echo(f"  Semantic: {config.weights['semantic']}")
    click.echo(f"  Lexical: {config.weights['lexical']}")
    click.echo(f"  Recency: {config.weights['recency']}")
    click.echo(f"  Actor: {config.weights['actor']}")
    click.echo(f"  Spatial: {config.weights['spatial']}")
    click.echo(f"  Usage: {config.weights['usage']}")


@cli.command()
def version():
    """Show version information"""
    click.echo("JAM - Journalistic Agent Memory")
    click.echo("Version: 1.0.0")
    click.echo("Python: " + sys.version.split()[0])
    
    # Check component versions
    try:
        import agentic_memory
        click.echo(f"Agentic Memory: {getattr(agentic_memory, '__version__', 'unknown')}")
    except:
        pass
    
    try:
        import fastapi
        click.echo(f"FastAPI: {fastapi.__version__}")
    except:
        pass
    
    try:
        import uvicorn
        click.echo(f"Uvicorn: {uvicorn.__version__}")
    except:
        pass


def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    main()
