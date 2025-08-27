"""
Comprehensive benchmark dataset generator for memory system testing
Creates realistic, diverse memories with synthetic timestamps for performance benchmarking
"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import os
import sys
import warnings
from pathlib import Path
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum

# Set offline mode
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from agent.memory_agent import MemoryAgent
from memory.memory_store import MemoryStore
from models.event import Event, FiveW1H, EventType
from config import get_config, LLMConfig
from llm.llm_interface import LLMInterface
import subprocess
import requests
import threading

class PersonaType(Enum):
    DEVELOPER = "developer"
    DATA_SCIENTIST = "data_scientist"
    DEVOPS = "devops"
    STUDENT = "student"
    MANAGER = "manager"
    DESIGNER = "designer"
    RESEARCHER = "researcher"
    QA_ENGINEER = "qa_engineer"

class LocationType(Enum):
    CODEBASE = "codebase"
    TERMINAL = "terminal"
    IDE = "ide"
    BROWSER = "browser"
    DOCUMENTATION = "documentation"
    GITHUB = "github"
    SLACK = "slack"
    NOTEBOOK = "jupyter_notebook"
    DATABASE = "database"
    CLOUD_CONSOLE = "cloud_console"
    API_CLIENT = "api_client"
    DEBUGGER = "debugger"

class ActivityType(Enum):
    CODING = "coding"
    DEBUGGING = "debugging"
    MEETING = "meeting"
    LEARNING = "learning"
    PLANNING = "planning"
    REVIEWING = "reviewing"
    TESTING = "testing"
    DEPLOYING = "deploying"
    DOCUMENTING = "documenting"
    RESEARCHING = "researching"

@dataclass
class ConversationScenario:
    topic: str
    category: str
    complexity: str  # simple, medium, complex
    space_preference: str  # euclidean, hyperbolic, balanced
    personas: List[PersonaType]
    activities: List[ActivityType]
    keywords: List[str]

class BenchmarkDatasetGenerator:
    def __init__(self, output_dir: str = "./benchmark_datasets", use_dual_llm: bool = False, 
                 alternative_model: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.config = get_config()
        self.memory_agent = MemoryAgent(self.config)
        
        # Setup LLMs for conversation generation
        self.use_dual_llm = use_dual_llm
        self.alternative_model = alternative_model or r"C:\models\Qwen3-4B-Instruct-2507\Qwen3-4B-Instruct-2507-UD-Q8_K_XL.gguf"
        self.llm_servers = []
        self._setup_llms()
        
        # Initialize personas with characteristics
        self.personas = self._initialize_personas()
        
        # Initialize conversation scenarios
        self.scenarios = self._initialize_scenarios()
        
        # Initialize temporal patterns
        self.temporal_patterns = self._initialize_temporal_patterns()
        
        # Statistics tracking
        self.stats = {
            'total_memories': 0,
            'by_persona': {},
            'by_category': {},
            'by_complexity': {},
            'by_time_period': {},
            'space_distribution': {'euclidean': 0, 'hyperbolic': 0, 'balanced': 0}
        }
    
    def _setup_llms(self):
        """Setup LLM interfaces for conversation generation"""
        if self.use_dual_llm:
            print("Setting up dual LLM servers for realistic conversation generation...")
            print("  Tip: Loading models can take 30-60s. Consider using --no-mmap for faster loading.")
            
            # Start both servers in parallel using threads
            import threading
            
            results = {'user': False, 'assistant': False}
            
            def start_user_server():
                results['user'] = self._start_llm_server(
                    self.alternative_model, 
                    port=8001, 
                    gpu_layers=35,  # Full GPU loading for Qwen 4B
                    use_mmap=True  # Keep model mapped in memory
                )
            
            def start_assistant_server():
                results['assistant'] = self._start_llm_server(
                    self.alternative_model, 
                    port=8002, 
                    gpu_layers=35,  # Full GPU loading for Qwen 4B  
                    use_mmap=True  # Keep model mapped in memory
                )
            
            # Start both servers in parallel
            user_thread = threading.Thread(target=start_user_server)
            assistant_thread = threading.Thread(target=start_assistant_server)
            
            user_thread.start()
            assistant_thread.start()
            
            # Wait for both to complete
            user_thread.join()
            assistant_thread.join()
            
            user_server_started = results['user']
            assistant_server_started = results['assistant']
            
            if user_server_started and assistant_server_started:
                # Create LLM interfaces WITHOUT calling ensure_llm_server
                # We already started the servers ourselves
                from llama_server_client import LlamaServerClient
                
                user_config = LLMConfig(
                    server_url="http://localhost:8001",
                    temperature=0.7,
                    max_tokens=150
                )
                assistant_config = LLMConfig(
                    server_url="http://localhost:8002",
                    temperature=0.5,
                    max_tokens=200
                )
                
                # Create interfaces but use direct clients to avoid ensure_llm_server
                self.user_llm = LLMInterface.__new__(LLMInterface)
                self.user_llm.config = user_config
                self.user_llm.client = LlamaServerClient(
                    base_url="http://localhost:8001",
                    timeout=30
                )
                
                self.assistant_llm = LLMInterface.__new__(LLMInterface)
                self.assistant_llm.config = assistant_config
                self.assistant_llm.client = LlamaServerClient(
                    base_url="http://localhost:8002",
                    timeout=30
                )
                
                print("  ✓ Dual LLM servers ready for realistic conversation generation")
            else:
                print("  Failed to start dual LLM servers, falling back to templates")
                self.user_llm = None
                self.assistant_llm = None
        else:
            # Try to use single LLM if available
            try:
                print("Checking for existing LLM server...")
                response = requests.get("http://localhost:8000/health", timeout=2)
                if response.status_code == 200:
                    print("  Using existing LLM server for conversation generation")
                    self.user_llm = LLMInterface(self.config.llm)
                    self.assistant_llm = self.user_llm  # Same LLM for both
                else:
                    print("  No LLM server found, using template-based generation")
                    self.user_llm = None
                    self.assistant_llm = None
            except:
                print("  No LLM server available, using template-based generation")
                self.user_llm = None
                self.assistant_llm = None
    
    def _start_llm_server(self, model_path: str, port: int, gpu_layers: int = 20, use_mmap: bool = True):
        """Start an LLM server on specified port"""
        try:
            # Check if server already running
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=2)
                if response.status_code == 200:
                    print(f"  LLM server already running on port {port}")
                    return True
            except:
                pass
            
            # Check if model exists
            if not Path(model_path).exists():
                print(f"  Warning: Model not found at {model_path}")
                return False
            
            # Find server executable - same approach as llama_server_client.py
            import shutil
            import platform
            
            if platform.system() == "Windows":
                server_executable = "llama-server.exe"
            else:
                server_executable = "llama-server"
            
            # Check for server in current directory first
            server_path = Path(server_executable)
            if not server_path.exists():
                # Check if it's in PATH
                server_cmd = shutil.which(server_executable)
                if server_cmd is None:
                    # Try alternative names
                    for alt_name in ["server.exe", "./server.exe"]:
                        if Path(alt_name).exists():
                            server_cmd = str(Path(alt_name))
                            break
                        alt_cmd = shutil.which(alt_name)
                        if alt_cmd:
                            server_cmd = alt_cmd
                            break
                    
                    if not server_cmd:
                        print(f"  Warning: {server_executable} not found.")
                        print(f"  Install with: winget install ggml.llamacpp")
                        return False
            else:
                server_cmd = str(server_path)
            
            # Build command - using same format as llama_server_client.py
            cmd = [
                server_cmd,
                "-m", model_path,  # Use -m instead of --model
                "--host", "0.0.0.0",
                "--port", str(port),
                "-c", "2048",  # Reduced context for faster loading
                "-t", "4",  # Fewer threads to avoid contention between servers
                "--alias", f"qwen-{port}",
                "-b", "256",  # Smaller batch size for faster loading
                "-ub", "256"  # Smaller ubatch size
            ]
            
            # Add GPU layers - use -ngl shorthand and ensure full GPU loading
            # For 4B model, typically needs ~30-35 layers for full GPU
            if gpu_layers > 0:
                cmd.extend(["-ngl", str(gpu_layers)])  # Use shorthand
            else:
                cmd.extend(["-ngl", "99"])  # Load ALL layers to GPU
            
            # Memory management flags
            if use_mmap:
                # Keep model memory-mapped (default, better for persistent servers)
                pass  
            else:
                cmd.append("--no-mmap")  # Load entire model to RAM/VRAM at once
            
            # Keep model in GPU memory
            cmd.append("--keep-in-gpu")  # Keep model in GPU between requests (if supported)
            
            # Use flash attention for efficiency
            cmd.append("--flash-attn")  
            
            # Don't offload KV cache to save VRAM switching
            cmd.append("--no-kv-offload")  # Keep KV cache in VRAM too
            
            print(f"  Starting LLM server on port {port}...")
            print(f"  Model: {Path(model_path).name}")
            print(f"  GPU layers: {gpu_layers if gpu_layers > 0 else 'ALL'}")
            print(f"  Memory mode: {'mmap' if use_mmap else 'no-mmap (faster load)'}")
            
            # Start server
            if platform.system() == "Windows":
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid
                )
            
            self.llm_servers.append(process)
            
            # Wait for server to be ready with detailed progress
            print(f"  Loading model into memory...")
            last_status = ""
            for i in range(60):  # 60 second timeout
                # Check if process crashed
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    print(f"\n  ❌ Server crashed! Exit code: {process.returncode}")
                    if stderr:
                        error_msg = stderr.decode()
                        if "out of memory" in error_msg.lower():
                            print(f"  Error: Out of memory - try reducing GPU layers")
                        elif "not found" in error_msg.lower():
                            print(f"  Error: Model file not found")
                        else:
                            print(f"  Error: {error_msg[:500]}")
                    return False
                
                try:
                    response = requests.get(f"http://localhost:{port}/health", timeout=1)
                    if response.status_code == 200:
                        print(f"\n  ✅ LLM server ready on port {port}!")
                        return True
                except:
                    # Show progress dots
                    if i < 10:
                        status = "Loading model"
                    elif i < 20:
                        status = "Initializing inference"
                    elif i < 30:
                        status = "Setting up API"
                    else:
                        status = "Almost ready"
                    
                    if status != last_status:
                        print(f"\n    {status}", end="", flush=True)
                        last_status = status
                    else:
                        print(".", end="", flush=True)
                    
                    time.sleep(1)
            
            print(f"  Warning: LLM server on port {port} didn't respond in time")
            return False
            
        except Exception as e:
            print(f"  Error starting LLM server: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def __del__(self):
        """Cleanup LLM servers on exit"""
        for process in self.llm_servers:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
    
    def _initialize_personas(self) -> Dict[PersonaType, Dict]:
        """Initialize personas with their characteristics"""
        return {
            PersonaType.DEVELOPER: {
                'name_pool': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
                'typical_activities': [ActivityType.CODING, ActivityType.DEBUGGING, ActivityType.REVIEWING],
                'typical_locations': [LocationType.IDE, LocationType.TERMINAL, LocationType.CODEBASE],
                'communication_style': 'technical',
                'query_patterns': ['concrete', 'technical', 'code-focused']
            },
            PersonaType.DATA_SCIENTIST: {
                'name_pool': ['Sarah', 'Mike', 'Lisa', 'Tom', 'Emma', 'Jack'],
                'typical_activities': [ActivityType.RESEARCHING, ActivityType.CODING, ActivityType.TESTING],
                'typical_locations': [LocationType.NOTEBOOK, LocationType.DATABASE, LocationType.DOCUMENTATION],
                'communication_style': 'analytical',
                'query_patterns': ['statistical', 'exploratory', 'model-focused']
            },
            PersonaType.DEVOPS: {
                'name_pool': ['Alex', 'Sam', 'Pat', 'Jordan', 'Taylor', 'Morgan'],
                'typical_activities': [ActivityType.DEPLOYING, ActivityType.TESTING, ActivityType.PLANNING],
                'typical_locations': [LocationType.TERMINAL, LocationType.CLOUD_CONSOLE, LocationType.GITHUB],
                'communication_style': 'process-oriented',
                'query_patterns': ['infrastructure', 'automation', 'monitoring']
            },
            PersonaType.STUDENT: {
                'name_pool': ['Chris', 'Jamie', 'Robin', 'Casey', 'Drew', 'Quinn'],
                'typical_activities': [ActivityType.LEARNING, ActivityType.CODING, ActivityType.RESEARCHING],
                'typical_locations': [LocationType.BROWSER, LocationType.IDE, LocationType.DOCUMENTATION],
                'communication_style': 'inquisitive',
                'query_patterns': ['exploratory', 'conceptual', 'example-seeking']
            },
            PersonaType.MANAGER: {
                'name_pool': ['David', 'Jennifer', 'Robert', 'Michelle', 'William', 'Linda'],
                'typical_activities': [ActivityType.PLANNING, ActivityType.MEETING, ActivityType.REVIEWING],
                'typical_locations': [LocationType.SLACK, LocationType.GITHUB, LocationType.BROWSER],
                'communication_style': 'strategic',
                'query_patterns': ['high-level', 'process', 'timeline-focused']
            },
            PersonaType.DESIGNER: {
                'name_pool': ['Olivia', 'Ethan', 'Sophia', 'Mason', 'Ava', 'Lucas'],
                'typical_activities': [ActivityType.PLANNING, ActivityType.REVIEWING, ActivityType.DOCUMENTING],
                'typical_locations': [LocationType.BROWSER, LocationType.DOCUMENTATION, LocationType.API_CLIENT],
                'communication_style': 'creative',
                'query_patterns': ['visual', 'user-focused', 'iterative']
            },
            PersonaType.RESEARCHER: {
                'name_pool': ['Nathan', 'Grace', 'Ryan', 'Lily', 'Andrew', 'Zoe'],
                'typical_activities': [ActivityType.RESEARCHING, ActivityType.DOCUMENTING, ActivityType.TESTING],
                'typical_locations': [LocationType.NOTEBOOK, LocationType.DOCUMENTATION, LocationType.BROWSER],
                'communication_style': 'methodical',
                'query_patterns': ['theoretical', 'comparative', 'evidence-based']
            },
            PersonaType.QA_ENGINEER: {
                'name_pool': ['Kevin', 'Amy', 'Brian', 'Nicole', 'Steven', 'Laura'],
                'typical_activities': [ActivityType.TESTING, ActivityType.DEBUGGING, ActivityType.DOCUMENTING],
                'typical_locations': [LocationType.DEBUGGER, LocationType.API_CLIENT, LocationType.TERMINAL],
                'communication_style': 'detail-oriented',
                'query_patterns': ['edge-cases', 'validation', 'regression']
            }
        }
    
    def _initialize_scenarios(self) -> List[ConversationScenario]:
        """Initialize diverse conversation scenarios"""
        scenarios = []
        
        # Technical scenarios (Euclidean-favoring) - Concrete implementation questions
        technical_topics = [
            # Programming Languages & Frameworks
            ("Python exception handling", ["try-except", "errors", "traceback", "logging"]),
            ("JavaScript async/await patterns", ["promises", "callbacks", "async", "error-handling"]),
            ("TypeScript type system", ["interfaces", "generics", "union-types", "type-guards"]),
            ("Go concurrency patterns", ["goroutines", "channels", "mutex", "waitgroups"]),
            ("Rust memory management", ["ownership", "borrowing", "lifetimes", "references"]),
            ("Java Spring Boot setup", ["annotations", "dependency-injection", "controllers", "services"]),
            ("C++ template metaprogramming", ["templates", "SFINAE", "concepts", "compile-time"]),
            ("Ruby on Rails migrations", ["ActiveRecord", "schema", "rollback", "indexes"]),
            
            # Frontend Development
            ("React hooks implementation", ["useState", "useEffect", "custom-hooks", "dependencies"]),
            ("Vue.js component communication", ["props", "emit", "vuex", "composition-api"]),
            ("Angular dependency injection", ["providers", "services", "injector", "modules"]),
            ("Next.js SSR/SSG", ["getServerSideProps", "getStaticProps", "dynamic-routes", "API-routes"]),
            ("CSS Grid layouts", ["grid-template", "grid-areas", "responsive", "alignment"]),
            ("WebSocket implementation", ["connection", "events", "reconnection", "heartbeat"]),
            ("Progressive Web Apps", ["service-workers", "manifest", "caching", "offline"]),
            
            # Backend Development
            ("REST API error handling", ["status-codes", "error-messages", "validation", "middleware"]),
            ("GraphQL resolver optimization", ["N+1", "dataloader", "batching", "caching"]),
            ("Database transaction management", ["ACID", "isolation-levels", "deadlocks", "rollback"]),
            ("Message queue implementation", ["RabbitMQ", "Kafka", "pub-sub", "dead-letter"]),
            ("Authentication implementation", ["JWT", "OAuth", "sessions", "refresh-tokens"]),
            ("Rate limiting strategies", ["token-bucket", "sliding-window", "distributed", "redis"]),
            ("File upload handling", ["multipart", "streaming", "validation", "storage"]),
            
            # Data Engineering
            ("SQL query optimization", ["indexes", "explain-plan", "joins", "partitioning"]),
            ("NoSQL data modeling", ["denormalization", "sharding", "consistency", "CAP-theorem"]),
            ("ETL pipeline debugging", ["data-quality", "transformations", "scheduling", "monitoring"]),
            ("Stream processing", ["Apache-Kafka", "windowing", "watermarks", "exactly-once"]),
            ("Data warehouse design", ["star-schema", "snowflake", "dimensions", "facts"]),
            ("Apache Spark optimization", ["partitions", "shuffling", "caching", "broadcast"]),
            
            # DevOps & Infrastructure
            ("Docker container debugging", ["logs", "exec", "networking", "volumes"]),
            ("Kubernetes deployment issues", ["pods", "services", "ingress", "configmaps"]),
            ("Terraform state management", ["backend", "locking", "import", "refresh"]),
            ("CI/CD pipeline setup", ["GitHub-Actions", "Jenkins", "artifacts", "stages"]),
            ("AWS Lambda cold starts", ["provisioned-concurrency", "layers", "runtime", "memory"]),
            ("Nginx configuration", ["proxy-pass", "load-balancing", "SSL", "rate-limiting"]),
            ("Monitoring and alerting", ["Prometheus", "Grafana", "metrics", "thresholds"]),
            
            # Testing & QA
            ("Unit test mocking", ["mock", "stub", "spy", "dependency-injection"]),
            ("Integration test setup", ["test-containers", "fixtures", "cleanup", "isolation"]),
            ("E2E test automation", ["Selenium", "Cypress", "selectors", "waits"]),
            ("Performance testing", ["JMeter", "load-testing", "bottlenecks", "profiling"]),
            ("API testing strategies", ["Postman", "assertions", "environments", "data-driven"]),
            
            # Security
            ("XSS prevention", ["sanitization", "CSP", "encoding", "validation"]),
            ("SQL injection prevention", ["prepared-statements", "parameterization", "escaping", "ORM"]),
            ("CORS configuration", ["origins", "credentials", "preflight", "headers"]),
            ("Secrets management", ["vault", "environment-variables", "encryption", "rotation"]),
            
            # Mobile Development
            ("React Native navigation", ["react-navigation", "stack", "tabs", "deep-linking"]),
            ("iOS Swift concurrency", ["async-await", "actors", "tasks", "MainActor"]),
            ("Android Kotlin coroutines", ["suspend", "flow", "dispatcher", "scope"]),
            ("Flutter state management", ["provider", "bloc", "riverpod", "setState"])
        ]
        
        for topic, keywords in technical_topics:
            scenarios.append(ConversationScenario(
                topic=topic,
                category="technical",
                complexity=random.choice(["simple", "medium", "complex"]),
                space_preference="euclidean",
                personas=[PersonaType.DEVELOPER, PersonaType.DEVOPS, PersonaType.QA_ENGINEER, PersonaType.DATA_SCIENTIST],
                activities=[ActivityType.CODING, ActivityType.DEBUGGING, ActivityType.TESTING],
                keywords=keywords
            ))
        
        # Conceptual scenarios (Hyperbolic-favoring) - Abstract understanding
        conceptual_topics = [
            # Architecture & Design
            ("Microservices vs Monolith", ["boundaries", "coupling", "deployment", "complexity"]),
            ("Event-driven architecture", ["events", "eventual-consistency", "CQRS", "event-sourcing"]),
            ("Domain-driven design", ["bounded-context", "aggregates", "entities", "value-objects"]),
            ("Clean architecture principles", ["layers", "dependencies", "use-cases", "ports-adapters"]),
            ("Hexagonal architecture", ["ports", "adapters", "domain", "infrastructure"]),
            ("SOLID principles", ["single-responsibility", "open-closed", "liskov", "dependency-inversion"]),
            ("Design patterns", ["factory", "observer", "strategy", "decorator"]),
            
            # Development Methodologies
            ("Agile vs Waterfall", ["iterations", "flexibility", "documentation", "risk"]),
            ("DevOps philosophy", ["culture", "automation", "measurement", "sharing"]),
            ("Test-driven development", ["red-green-refactor", "design", "coverage", "benefits"]),
            ("Continuous integration principles", ["automation", "feedback", "versioning", "collaboration"]),
            ("Pair programming benefits", ["knowledge-sharing", "quality", "mentoring", "collaboration"]),
            
            # Data & ML Concepts
            ("Machine learning paradigms", ["supervised", "unsupervised", "reinforcement", "transfer"]),
            ("Deep learning architectures", ["CNN", "RNN", "transformer", "attention"]),
            ("Data mesh principles", ["domains", "products", "self-serve", "governance"]),
            ("Feature engineering strategies", ["selection", "extraction", "scaling", "encoding"]),
            ("MLOps practices", ["versioning", "monitoring", "deployment", "reproducibility"]),
            
            # System Design
            ("Scalability patterns", ["horizontal", "vertical", "caching", "sharding"]),
            ("High availability concepts", ["redundancy", "failover", "load-balancing", "disaster-recovery"]),
            ("CAP theorem implications", ["consistency", "availability", "partition-tolerance", "trade-offs"]),
            ("Performance optimization philosophy", ["bottlenecks", "profiling", "caching", "algorithms"]),
            ("Security by design", ["defense-in-depth", "least-privilege", "zero-trust", "encryption"]),
            
            # Team & Process
            ("Code review best practices", ["feedback", "standards", "automation", "culture"]),
            ("Technical debt management", ["identification", "prioritization", "refactoring", "prevention"]),
            ("Documentation philosophy", ["audience", "maintenance", "clarity", "examples"]),
            ("Knowledge sharing strategies", ["documentation", "mentoring", "presentations", "pairing"]),
            
            # Technology Trends
            ("Serverless architecture benefits", ["scaling", "cost", "maintenance", "limitations"]),
            ("Edge computing concepts", ["latency", "bandwidth", "privacy", "processing"]),
            ("Blockchain fundamentals", ["consensus", "immutability", "decentralization", "smart-contracts"]),
            ("Quantum computing implications", ["qubits", "superposition", "algorithms", "applications"]),
            ("AI ethics considerations", ["bias", "transparency", "privacy", "accountability"])
        ]
        
        for topic, keywords in conceptual_topics:
            scenarios.append(ConversationScenario(
                topic=topic,
                category="conceptual",
                complexity=random.choice(["medium", "complex"]),
                space_preference="hyperbolic",
                personas=[PersonaType.MANAGER, PersonaType.RESEARCHER, PersonaType.STUDENT, PersonaType.DESIGNER],
                activities=[ActivityType.PLANNING, ActivityType.LEARNING, ActivityType.RESEARCHING],
                keywords=keywords
            ))
        
        # Mixed scenarios (Balanced) - Practical implementation with theory
        mixed_topics = [
            # Full-Stack Projects
            ("E-commerce platform development", ["cart", "payments", "inventory", "search"]),
            ("Social media app architecture", ["feeds", "notifications", "messaging", "scaling"]),
            ("SaaS application design", ["multi-tenancy", "billing", "authentication", "API"]),
            ("Real-time collaboration tools", ["WebRTC", "conflict-resolution", "presence", "sync"]),
            ("Content management system", ["workflow", "permissions", "versioning", "plugins"]),
            
            # Data Projects
            ("Data pipeline architecture", ["ingestion", "transformation", "storage", "serving"]),
            ("Analytics platform design", ["collection", "processing", "visualization", "insights"]),
            ("Recommendation system implementation", ["collaborative", "content-based", "hybrid", "evaluation"]),
            ("Search engine optimization", ["indexing", "ranking", "relevance", "performance"]),
            ("Time-series data handling", ["storage", "compression", "querying", "forecasting"]),
            
            # Infrastructure Projects
            ("Multi-region deployment", ["latency", "data-replication", "failover", "compliance"]),
            ("Disaster recovery planning", ["RTO", "RPO", "backups", "testing"]),
            ("Cost optimization strategies", ["monitoring", "right-sizing", "reserved-instances", "spot"]),
            ("Security implementation", ["WAF", "encryption", "compliance", "auditing"]),
            ("Platform migration", ["assessment", "planning", "execution", "validation"]),
            
            # Process Improvements
            ("CI/CD implementation", ["pipelines", "testing", "deployment", "rollback"]),
            ("Monitoring strategy", ["metrics", "logs", "traces", "alerts"]),
            ("Performance optimization project", ["benchmarking", "profiling", "optimization", "validation"]),
            ("Technical documentation system", ["automation", "versioning", "search", "maintenance"]),
            ("Developer productivity tools", ["automation", "tooling", "environment", "workflow"]),
            
            # Team Projects
            ("Onboarding process design", ["documentation", "mentoring", "tools", "feedback"]),
            ("Code quality improvement", ["linting", "formatting", "reviews", "metrics"]),
            ("Knowledge base creation", ["organization", "search", "contribution", "maintenance"]),
            ("Incident response planning", ["detection", "triage", "resolution", "post-mortem"]),
            ("Technical training program", ["curriculum", "delivery", "assessment", "improvement"])
        ]
        
        for topic, keywords in mixed_topics:
            scenarios.append(ConversationScenario(
                topic=topic,
                category="mixed",
                complexity="complex",
                space_preference="balanced",
                personas=[PersonaType.DEVELOPER, PersonaType.DATA_SCIENTIST, PersonaType.DESIGNER, PersonaType.DEVOPS],
                activities=[ActivityType.CODING, ActivityType.PLANNING, ActivityType.REVIEWING],
                keywords=keywords
            ))
        
        return scenarios
    
    def _initialize_temporal_patterns(self) -> Dict[str, Dict]:
        """Initialize temporal patterns for realistic timestamp generation"""
        return {
            'workday': {
                'weight': 0.5,
                'hour_distribution': {
                    9: 0.15, 10: 0.20, 11: 0.15,  # Morning
                    13: 0.10, 14: 0.15, 15: 0.10,  # Afternoon
                    16: 0.10, 17: 0.05             # Late afternoon
                },
                'days': [0, 1, 2, 3, 4]  # Monday to Friday
            },
            'evening': {
                'weight': 0.25,
                'hour_distribution': {
                    18: 0.10, 19: 0.15, 20: 0.25,
                    21: 0.25, 22: 0.15, 23: 0.10
                },
                'days': [0, 1, 2, 3, 4, 5, 6]  # All days
            },
            'weekend': {
                'weight': 0.15,
                'hour_distribution': {
                    10: 0.10, 11: 0.15, 12: 0.10,
                    14: 0.15, 15: 0.20, 16: 0.15,
                    17: 0.10, 18: 0.05
                },
                'days': [5, 6]  # Saturday and Sunday
            },
            'late_night': {
                'weight': 0.10,
                'hour_distribution': {
                    0: 0.20, 1: 0.30, 2: 0.30, 3: 0.20
                },
                'days': [0, 1, 2, 3, 4, 5, 6]  # All days
            }
        }
    
    def generate_synthetic_timestamp(self, base_time: datetime, variance_hours: int = 72) -> str:
        """Generate a realistic synthetic timestamp"""
        # Select temporal pattern
        pattern_name = random.choices(
            list(self.temporal_patterns.keys()),
            weights=[p['weight'] for p in self.temporal_patterns.values()]
        )[0]
        pattern = self.temporal_patterns[pattern_name]
        
        # Generate random offset within variance
        days_offset = random.randint(-variance_hours // 24, variance_hours // 24)
        target_date = base_time + timedelta(days=days_offset)
        
        # Adjust to match pattern's preferred days
        while target_date.weekday() not in pattern['days']:
            target_date += timedelta(days=1)
        
        # Select hour based on distribution
        hour = random.choices(
            list(pattern['hour_distribution'].keys()),
            weights=list(pattern['hour_distribution'].values())
        )[0]
        
        # Add random minutes and seconds
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        final_time = target_date.replace(hour=hour, minute=minute, second=second)
        return final_time.isoformat()
    
    def generate_conversation_exchange(
        self, 
        scenario: ConversationScenario,
        persona: PersonaType,
        timestamp: str,
        context: Optional[str] = None
    ) -> List[Event]:
        """Generate a conversation exchange with memories"""
        events = []
        persona_info = self.personas[persona]
        
        # Select random characteristics
        who = random.choice(persona_info['name_pool'])
        activity = random.choice(persona_info['typical_activities'])
        location = random.choice(persona_info['typical_locations'])
        
        # Generate conversation content based on scenario
        exchanges = self._generate_exchange_content(scenario, persona, context)
        
        for i, (what, why, how) in enumerate(exchanges):
            # Alternate between user input and observations to simulate conversation
            event_type = EventType.USER_INPUT if i % 2 == 0 else EventType.OBSERVATION
            
            event = Event(
                event_type=event_type,
                five_w1h=FiveW1H(
                    who=who,
                    what=what,
                    when=timestamp,
                    where=location.value,
                    why=why,
                    how=how
                ),
                episode_id=f"bench_{scenario.category}_{hashlib.md5(f'{timestamp}{i}'.encode()).hexdigest()[:8]}"
            )
            events.append(event)
            
            # Update statistics
            self.stats['total_memories'] += 1
            self.stats['by_persona'][persona.value] = self.stats['by_persona'].get(persona.value, 0) + 1
            self.stats['by_category'][scenario.category] = self.stats['by_category'].get(scenario.category, 0) + 1
            self.stats['space_distribution'][scenario.space_preference] += 1
        
        return events
    
    def _generate_exchange_content(
        self,
        scenario: ConversationScenario,
        persona: PersonaType,
        context: Optional[str]
    ) -> List[Tuple[str, str, str]]:
        """Generate content for conversation exchanges using LLMs"""
        exchanges = []
        persona_info = self.personas[persona]
        
        # Generate 2-5 exchanges per conversation
        num_exchanges = random.randint(2, 5)
        conversation_history = []
        
        print(f"    Generating {num_exchanges} exchanges", end="", flush=True)
        
        for i in range(num_exchanges):
            print(".", end="", flush=True)  # Progress dot for each exchange
            
            # Use LLM to generate user query if available
            if hasattr(self, 'user_llm') and self.user_llm:
                user_query = self._generate_user_query_llm(
                    scenario, persona, persona_info, conversation_history
                )
            else:
                user_query = self._generate_user_query_template(scenario, persona_info)
            
            # Use LLM to generate assistant response if available
            if hasattr(self, 'assistant_llm') and self.assistant_llm:
                assistant_response = self._generate_assistant_response_llm(
                    user_query, scenario, conversation_history
                )
            else:
                assistant_response = f"Response about {random.choice(scenario.keywords)} in {scenario.topic}"
            
            # Store in conversation history for context
            conversation_history.append({"user": user_query, "assistant": assistant_response})
            
            # Generate metadata
            why = self._generate_why(scenario)
            how = self._generate_how(persona_info)
            
            # Add both user and assistant exchanges
            exchanges.append((user_query, why, how))
            if i < num_exchanges - 1:  # Don't add assistant response for last exchange
                exchanges.append((assistant_response, f"responding to: {why}", "providing assistance"))
        
        print(" done", flush=True)  # Complete the line
        return exchanges
    
    def _generate_user_query_llm(
        self, 
        scenario: ConversationScenario, 
        persona: PersonaType,
        persona_info: Dict,
        conversation_history: List[Dict]
    ) -> str:
        """Generate user query using LLM"""
        
        # Build conversation context
        context = ""
        if conversation_history:
            context = "Previous conversation:\n"
            for exchange in conversation_history[-2:]:  # Last 2 exchanges
                context += f"User: {exchange['user'][:100]}...\n"
                context += f"Assistant: {exchange['assistant'][:100]}...\n"
        
        # Create prompt based on scenario type
        if scenario.space_preference == "euclidean":
            style = "specific, technical, and concrete. Include code snippets, error messages, or specific implementation details"
        elif scenario.space_preference == "hyperbolic":
            style = "conceptual and abstract. Focus on understanding principles, relationships, and theoretical aspects"
        else:
            style = "practical and balanced. Mix implementation concerns with broader architectural considerations"
        
        prompt = f"""You are a {persona.value} named {random.choice(persona_info['name_pool'])} working on {scenario.topic}.
{context}
Generate a realistic technical question about {random.choice(scenario.keywords)} that is {style}.
The question should be something a {persona_info['communication_style']} person would ask.

Question:"""
        
        try:
            response = self.user_llm.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.7,
                stop=["\n\n", "Answer:", "Response:"]
            )
            return response.strip() or self._generate_user_query_template(scenario, persona_info)
        except Exception as e:
            print(f"  LLM generation failed: {e}, using template")
            return self._generate_user_query_template(scenario, persona_info)
    
    def _generate_assistant_response_llm(
        self,
        user_query: str,
        scenario: ConversationScenario,
        conversation_history: List[Dict]
    ) -> str:
        """Generate assistant response using LLM"""
        
        prompt = f"""You are a helpful technical assistant responding to a question about {scenario.topic}.

User Question: {user_query}

Provide a concise, helpful response focusing on {', '.join(scenario.keywords[:3])}.
Keep the response under 100 words.

Response:"""
        
        try:
            response = self.assistant_llm.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.5,
                stop=["\n\nUser:", "\n\nQuestion:"]
            )
            return response.strip() or f"Here's information about {random.choice(scenario.keywords)} in {scenario.topic}"
        except Exception as e:
            return f"Information about {random.choice(scenario.keywords)} in the context of {scenario.topic}"
    
    def _generate_user_query_template(self, scenario: ConversationScenario, persona_info: Dict) -> str:
        """Fallback template-based query generation"""
        if scenario.space_preference == "euclidean":
            templates = [
                f"I'm getting an error with {random.choice(scenario.keywords)}. The stack trace shows...",
                f"Show me how to implement {random.choice(scenario.keywords)} with error handling",
                f"What's the correct syntax for {random.choice(scenario.keywords)}?",
                f"My {random.choice(scenario.keywords)} is failing. Here's my code:",
                f"I need to write a function that handles {random.choice(scenario.keywords)}"
            ]
        elif scenario.space_preference == "hyperbolic":
            templates = [
                f"Explain when I should use {random.choice(scenario.keywords)} vs alternatives",
                f"What problem does {random.choice(scenario.keywords)} solve?",
                f"How does {random.choice(scenario.keywords)} fit into the broader architecture?",
                f"What are the trade-offs of using {random.choice(scenario.keywords)}?",
                f"Why was {random.choice(scenario.keywords)} designed this way?"
            ]
        else:
            templates = [
                f"I'm refactoring our {random.choice(scenario.keywords)} implementation. What's the best approach?",
                f"How can I optimize {random.choice(scenario.keywords)} for production?",
                f"What's the recommended pattern for {random.choice(scenario.keywords)} in our stack?",
                f"How do I test {random.choice(scenario.keywords)} properly?"
            ]
        return random.choice(templates)
    
    def _generate_why(self, scenario: ConversationScenario) -> str:
        """Generate 'why' metadata"""
        why_templates = [
            f"fixing production bug in {scenario.topic}",
            f"implementing user story #{random.randint(100, 999)}",
            f"responding to code review feedback",
            f"investigating {scenario.complexity} issue",
            f"migrating legacy {scenario.category} code",
            f"adding new feature to {scenario.topic}",
            f"troubleshooting customer issue",
            f"preparing technical documentation",
            f"optimizing {random.choice(scenario.keywords)}",
            f"learning about {scenario.topic}"
        ]
        return random.choice(why_templates)
    
    def _generate_how(self, persona_info: Dict) -> str:
        """Generate 'how' metadata"""
        return f"{persona_info['communication_style']} query via chat interface"
    
    def generate_dataset(
        self,
        num_conversations: int = 100,
        time_span_days: int = 30,
        dataset_name: str = None
    ) -> Dict:
        """Generate a complete benchmark dataset"""
        if dataset_name is None:
            dataset_name = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n{'='*60}")
        print(f"DATASET GENERATION STARTED")
        print(f"{'='*60}")
        print(f"Dataset name: {dataset_name}")
        print(f"Target conversations: {num_conversations}")
        print(f"Time span: {time_span_days} days")
        print(f"Generation method: {'LLM-powered' if self.user_llm else 'Template-based'}")
        print(f"{'='*60}\n")
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_span_days)
        
        all_events = []
        conversation_metadata = []
        start_generation = time.time()
        
        # Generate conversations
        for conv_idx in range(num_conversations):
            conv_start = time.time()
            
            # Select random scenario
            scenario = random.choice(self.scenarios)
            
            # Select random persona
            persona = random.choice(scenario.personas)
            
            # Progress header for each conversation
            print(f"[Conv {conv_idx + 1}/{num_conversations}] ", end="")
            print(f"{scenario.topic[:30]}... ({persona.value})")
            
            # Generate timestamp
            base_time = start_time + timedelta(
                seconds=random.randint(0, int((end_time - start_time).total_seconds()))
            )
            timestamp = self.generate_synthetic_timestamp(base_time, variance_hours=24)
            
            # Generate conversation events
            print(f"  → Generating exchanges...", end="", flush=True)
            events = self.generate_conversation_exchange(scenario, persona, timestamp)
            print(f" ✓ ({len(events)} events)")
            
            # Store events
            stored_count = 0
            for event in events:
                # Store in memory system
                success, message = self.memory_agent.memory_store.store_event(event)
                if success:
                    stored_count += 1
                    all_events.append({
                        'event_id': event.id,  # Changed from event_id to id
                        'event_type': event.event_type.value,
                        'five_w1h': {
                            'who': event.five_w1h.who,
                            'what': event.five_w1h.what,
                            'when': event.five_w1h.when,
                            'where': event.five_w1h.where,
                            'why': event.five_w1h.why,
                            'how': event.five_w1h.how
                        },
                        'episode_id': event.episode_id,
                        'scenario': scenario.topic,
                        'category': scenario.category,
                        'complexity': scenario.complexity,
                        'space_preference': scenario.space_preference
                    })
            
            print(f"  → Stored {stored_count}/{len(events)} events in memory system")
            
            # Store conversation metadata
            conversation_metadata.append({
                'conversation_id': conv_idx,
                'scenario': scenario.topic,
                'category': scenario.category,
                'complexity': scenario.complexity,
                'space_preference': scenario.space_preference,
                'persona': persona.value,
                'timestamp': timestamp,
                'num_events': len(events)
            })
            
            # Time tracking
            conv_time = time.time() - conv_start
            print(f"  → Conversation time: {conv_time:.1f}s")
            
            # Periodic summary
            if (conv_idx + 1) % 10 == 0:
                elapsed = time.time() - start_generation
                rate = (conv_idx + 1) / elapsed
                remaining = (num_conversations - conv_idx - 1) / rate if rate > 0 else 0
                print(f"\n  *** Progress: {conv_idx + 1}/{num_conversations} conversations")
                print(f"      Total events: {len(all_events)}")
                print(f"      Elapsed: {elapsed:.1f}s | Rate: {rate:.1f} conv/s")
                print(f"      Estimated remaining: {remaining:.1f}s\n")
            elif (conv_idx + 1) % 5 == 0:
                # Shorter progress update
                print(f"  [Progress: {conv_idx + 1}/{num_conversations}]")
            
            print()  # Blank line between conversations
        
        # Compile dataset
        dataset = {
            'metadata': {
                'name': dataset_name,
                'created_at': datetime.now().isoformat(),
                'num_conversations': num_conversations,
                'num_events': len(all_events),
                'time_span_days': time_span_days,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            },
            'statistics': self.stats,
            'conversations': conversation_metadata,
            'events': all_events
        }
        
        # Save dataset
        print(f"\n{'='*60}")
        print(f"SAVING DATASET")
        print(f"{'='*60}")
        dataset_path = self.output_dir / f"{dataset_name}.json"
        print(f"  Writing to: {dataset_path}")
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        file_size_mb = dataset_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {file_size_mb:.2f} MB")
        
        # Final statistics
        total_time = time.time() - start_generation
        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE!")
        print(f"{'='*60}")
        print(f"Summary:")
        print(f"  Total conversations: {num_conversations}")
        print(f"  Total events: {len(all_events)}")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"  Average per conversation: {total_time/num_conversations:.2f}s")
        
        print(f"\nBreakdown by category:")
        for cat, count in sorted(self.stats['by_category'].items()):
            print(f"  {cat:15s}: {count:5d} events")
        
        print(f"\nBreakdown by persona:")
        for persona, count in sorted(self.stats['by_persona'].items()):
            print(f"  {persona:20s}: {count:5d} events")
        
        print(f"\nSpace distribution:")
        total_space = sum(self.stats['space_distribution'].values())
        for space, count in self.stats['space_distribution'].items():
            pct = (count / total_space * 100) if total_space > 0 else 0
            print(f"  {space:12s}: {count:5d} ({pct:.1f}%)")
        
        print(f"\nDataset saved to: {dataset_path}")
        print(f"{'='*60}")
        
        return dataset
    
    def generate_query_set(self, dataset: Dict, num_queries: int = 50) -> List[Dict]:
        """Generate benchmark queries based on the dataset"""
        queries = []
        
        # Extract keywords from events
        all_keywords = set()
        for event in dataset['events']:
            if event['five_w1h']['what']:
                # Extract meaningful words
                words = event['five_w1h']['what'].lower().split()
                all_keywords.update(w for w in words if len(w) > 3)
        
        keyword_list = list(all_keywords)
        
        # Generate diverse queries
        query_types = [
            ('concrete', 'euclidean', ['how to', 'implement', 'code for', 'example of']),
            ('abstract', 'hyperbolic', ['concept of', 'philosophy behind', 'relationship between', 'principles of']),
            ('mixed', 'balanced', ['best practices for', 'approach to', 'strategy for', 'optimize'])
        ]
        
        for i in range(num_queries):
            query_type, expected_space, prefixes = random.choice(query_types)
            prefix = random.choice(prefixes)
            keyword = random.choice(keyword_list)
            
            query = {
                'query_id': i,
                'query': f"{prefix} {keyword}",
                'type': query_type,
                'expected_space': expected_space,
                'timestamp': datetime.now().isoformat()
            }
            queries.append(query)
        
        # Save query set
        query_set_path = self.output_dir / f"{dataset['metadata']['name']}_queries.json"
        with open(query_set_path, 'w') as f:
            json.dump(queries, f, indent=2)
        
        print(f"\nGenerated {num_queries} benchmark queries")
        print(f"  Saved to: {query_set_path}")
        
        return queries

def main():
    """Main function for generating benchmark datasets"""
    print("="*60)
    print("BENCHMARK DATASET GENERATOR")
    print("="*60)
    
    # Check for LLM generation preference
    print("\nLLM Generation Options:")
    print("1. Use template-based generation (fast, no LLM required)")
    print("2. Use single LLM for both roles (good quality, faster loading)")
    print("3. Use dual LLMs for maximum realism (best quality, slower loading)")
    
    llm_choice = input("\nSelect LLM option (1-3) [default: 1]: ").strip() or "1"
    
    if llm_choice == "3":
        # Check if Qwen model exists
        qwen_model = r"C:\models\Qwen3-4B-Instruct-2507\Qwen3-4B-Instruct-2507-UD-Q8_K_XL.gguf"
        if not Path(qwen_model).exists():
            alt_path = input(f"\nQwen model not found at {qwen_model}\nEnter path to model (or press Enter to use templates): ").strip()
            if alt_path and Path(alt_path).exists():
                generator = BenchmarkDatasetGenerator(use_dual_llm=True, alternative_model=alt_path)
            else:
                print("Using template-based generation instead")
                generator = BenchmarkDatasetGenerator(use_dual_llm=False)
        else:
            generator = BenchmarkDatasetGenerator(use_dual_llm=True)
    elif llm_choice == "2":
        generator = BenchmarkDatasetGenerator(use_dual_llm=False)
    else:
        generator = BenchmarkDatasetGenerator(use_dual_llm=False)
    
    print("\nSelect dataset size:")
    print("1. Small (100 conversations, ~500 memories)")
    print("2. Medium (500 conversations, ~2,500 memories)")
    print("3. Large (1000 conversations, ~5,000 memories)")
    print("4. Extra Large (2000 conversations, ~10,000 memories)")
    print("5. Custom")
    
    choice = input("\nEnter choice (1-5) [default: 2]: ").strip() or "2"
    
    if choice == "1":
        num_conversations = 100
        time_span_days = 7
    elif choice == "2":
        num_conversations = 500
        time_span_days = 30
    elif choice == "3":
        num_conversations = 1000
        time_span_days = 60
    elif choice == "4":
        num_conversations = 2000
        time_span_days = 90
    elif choice == "5":
        num_conversations = int(input("Number of conversations: "))
        time_span_days = int(input("Time span (days): "))
    else:
        print("Invalid choice")
        return
    
    # Generate dataset
    dataset = generator.generate_dataset(
        num_conversations=num_conversations,
        time_span_days=time_span_days
    )
    
    # Generate query set
    num_queries = min(100, num_conversations // 5)
    generator.generate_query_set(dataset, num_queries=num_queries)
    
    print("\n" + "="*60)
    print("Benchmark dataset generation complete!")
    print("="*60)

if __name__ == "__main__":
    main()
