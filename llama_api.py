"""
FastAPI wrapper for llama.cpp server endpoints with additional features.
Provides a clean API interface that can run in the background.
"""

import os
import sys
import asyncio
import logging
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from contextlib import asynccontextmanager
import signal
import uvicorn

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import httpx
import tempfile
import shutil
from pathlib import Path

from llama_server_manager import LLMServerManager

# Import document parser and memory router
sys.path.insert(0, str(Path(__file__).parent))
from agentic_memory.document_parser import DocumentParser, SentenceChunker, ParagraphChunker, SemanticChunker
from agentic_memory.router import MemoryRouter
from agentic_memory.config_manager import ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Pydantic models for request/response
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = Field(default=100, gt=0, le=4096)
    temperature: Optional[float] = Field(default=0.3, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=40, ge=0)
    repetition_penalty: Optional[float] = Field(default=1.2, ge=0.0, le=2.0)
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    echo: Optional[bool] = False
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=100, gt=0, le=4096)
    temperature: Optional[float] = Field(default=0.3, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=40, ge=0)
    repetition_penalty: Optional[float] = Field(default=1.2, ge=0.0, le=2.0)
    stop: Optional[List[str]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    memory_search: Optional[bool] = Field(default=False, description="Enable memory search augmentation")
    memory_token_budget: Optional[int] = Field(default=None, description="Token budget for memory search")


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"
    context_size: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentIngestionRequest(BaseModel):
    chunking_strategy: Optional[str] = Field(default="semantic", pattern="^(semantic|paragraph|sentence)$")
    max_chunk_size: Optional[int] = Field(default=2000, gt=100, le=20480)
    chunk_overlap: Optional[int] = Field(default=200, ge=0, le=1000)
    metadata: Optional[Dict[str, Any]] = None


class DocumentIngestionResponse(BaseModel):
    file_name: str
    file_type: str
    chunks_created: int
    memories_created: int


class MemorySearchRequest(BaseModel):
    query: str
    token_budget: Optional[int] = None
    initial_candidates: Optional[int] = Field(default=100, gt=0, le=1000)
    weights: Optional[Dict[str, float]] = None


class DocumentIngestionFinalResponse(BaseModel):
    file_name: str
    file_type: str
    chunks_created: int
    memories_created: int
    total_words: int
    total_chars: int
    errors: List[str]
    success: bool


class APIConfig(BaseModel):
    """Configuration for the API wrapper"""
    host: str = "0.0.0.0"
    port: int = 8001  # Different from llama server port
    llama_server_url: str = "http://localhost:8000"
    enable_cors: bool = True
    cors_origins: List[str] = ["*"]
    api_key: Optional[str] = None
    log_requests: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 300  # 5 minutes
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds


class LlamaAPIWrapper:
    """Main API wrapper class"""
    
    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self.server_manager = LLMServerManager()
        self.client: Optional[httpx.AsyncClient] = None
        self.request_count = 0
        self.start_time = datetime.now()
        self.cache: Dict[str, Any] = {}
        self.memory_router: Optional[MemoryRouter] = None
        self.document_parser: Optional[DocumentParser] = None
        
    async def startup(self):
        """Initialize resources on startup"""
        # Ensure llama server is running
        if not self.server_manager.ensure_running():
            logger.error("Failed to start llama.cpp server")
            raise RuntimeError("Cannot start llama.cpp server")
        
        # Create async HTTP client
        self.client = httpx.AsyncClient(
            base_url=f"{self.config.llama_server_url}/v1",
            timeout=httpx.Timeout(60.0)
        )
        
        # Initialize memory router and document parser
        try:
            config_manager = ConfigManager()
            self.memory_router = MemoryRouter(config_manager)
            self.document_parser = DocumentParser()
            logger.info("Memory router and document parser initialized")
        except Exception as e:
            logger.warning(f"Could not initialize memory components: {e}")
        
        logger.info(f"API wrapper connected to llama server at {self.config.llama_server_url}")
        
    async def shutdown(self):
        """Cleanup resources on shutdown"""
        if self.client:
            await self.client.aclose()
        logger.info("API wrapper shutting down")
    
    async def forward_request(self, path: str, method: str = "POST", **kwargs) -> Dict[str, Any]:
        """Forward request to llama server with error handling"""
        try:
            if method == "GET":
                response = await self.client.get(path, **kwargs)
            else:
                response = await self.client.post(path, **kwargs)
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from llama server: {e}")
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            logger.error(f"Error forwarding request: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def completion(self, request: CompletionRequest) -> Dict[str, Any]:
        """Handle completion requests with caching and metrics"""
        self.request_count += 1
        
        # Check cache if enabled
        cache_key = f"completion:{hash(request.prompt[:100])}"
        if self.config.cache_enabled and cache_key in self.cache:
            cached = self.cache[cache_key]
            if (datetime.now() - cached['time']).seconds < self.config.cache_ttl:
                return cached['response']
        
        # Forward to llama server
        response = await self.forward_request(
            "/completions",
            json=request.dict(exclude_none=True)
        )
        
        # Cache response
        if self.config.cache_enabled:
            self.cache[cache_key] = {
                'response': response,
                'time': datetime.now()
            }
        
        return response
    
    async def chat_completion(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Handle chat completion requests with optional memory search augmentation"""
        self.request_count += 1
        
        # Convert messages to format expected by llama server
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # If memory search is enabled, augment the context
        if request.memory_search and self.memory_router:
            # Extract query from recent messages (last 3 user messages)
            query_parts = []
            for msg in reversed(request.messages):
                if msg.role == "user":
                    query_parts.append(msg.content)
                    if len(query_parts) >= 3:
                        break
            
            if query_parts:
                # Combine recent user messages as query
                search_query = "\n".join(reversed(query_parts))
                
                # Perform memory search
                memory_result = self.memory_router.search_memories(
                    query=search_query,
                    token_budget=request.memory_token_budget,
                    initial_candidates=100
                )
                
                # Format memories as context
                if memory_result.get('memories'):
                    memory_context = "## Relevant Memories:\n\n"
                    for mem in memory_result['memories'][:10]:  # Limit to top 10
                        memory_context += f"**Memory {mem['memory_id']} (score: {mem['total_score']}):**\n"
                        memory_context += f"- When: {mem['when']}\n"
                        memory_context += f"- Who: {mem['who']}\n"
                        memory_context += f"- What: {mem['what']}\n"
                        if mem.get('why'):
                            memory_context += f"- Why: {mem['why']}\n"
                        if mem.get('how'):
                            memory_context += f"- How: {mem['how']}\n"
                        memory_context += f"- Context: {mem['raw_text'][:300]}...\n\n"
                    
                    # Insert memory context as a system message
                    memory_msg = {
                        "role": "system",
                        "content": f"The following memories from your knowledge base may be relevant to this conversation:\n\n{memory_context}"
                    }
                    
                    # Insert after initial system message or at beginning
                    if messages and messages[0]["role"] == "system":
                        messages.insert(1, memory_msg)
                    else:
                        messages.insert(0, memory_msg)
        
        # Prepare request without memory_search fields
        forward_params = request.dict(exclude={'messages', 'memory_search', 'memory_token_budget'}, exclude_none=True)
        
        response = await self.forward_request(
            "/chat/completions",
            json={
                "messages": messages,
                **forward_params
            }
        )
        
        return response
    
    async def embeddings(self, request: EmbeddingRequest) -> Dict[str, Any]:
        """Handle embedding requests"""
        self.request_count += 1
        
        # Ensure input is a list
        inputs = request.input if isinstance(request.input, list) else [request.input]
        
        response = await self.forward_request(
            "/embeddings",
            json={"input": inputs}
        )
        
        return response
    
    async def models(self) -> Dict[str, Any]:
        """Get available models"""
        response = await self.forward_request("/models", method="GET")
        return response
    
    async def health(self) -> Dict[str, Any]:
        """Health check with extended metrics"""
        # Check llama server health
        llama_health = self.server_manager.is_running()
        
        # Calculate uptime
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "status": "healthy" if llama_health else "unhealthy",
            "llama_server": "running" if llama_health else "stopped",
            "api_wrapper": {
                "uptime_seconds": uptime,
                "requests_processed": self.request_count,
                "cache_size": len(self.cache),
                "version": "1.0.0"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def stats(self) -> Dict[str, Any]:
        """Get detailed statistics"""
        server_status = self.server_manager.get_status()
        
        stats = {
            "server": server_status,
            "api": {
                "requests": self.request_count,
                "uptime": (datetime.now() - self.start_time).total_seconds(),
                "cache_entries": len(self.cache)
            }
        }
        
        # Add memory stats if available
        if self.memory_router:
            try:
                memory_stats = self.memory_router.get_stats()
                stats["memory"] = memory_stats
            except:
                pass
        
        return stats
    
    async def search_memories(self, request: MemorySearchRequest) -> Dict[str, Any]:
        """Search memories using the same process as the analyzer"""
        if not self.memory_router:
            raise HTTPException(status_code=503, detail="Memory search not available")
        
        # Call router's search_memories method
        result = self.memory_router.search_memories(
            query=request.query,
            weights=request.weights,
            token_budget=request.token_budget,
            initial_candidates=request.initial_candidates
        )
        
        return result
    
    async def ingest_document(self, file_path: str, request: DocumentIngestionRequest) -> DocumentIngestionResponse:
        """Ingest a document into memory"""
        if not self.document_parser or not self.memory_router:
            raise HTTPException(status_code=503, detail="Document processing not available")
        
        # Select chunking strategy
        if request.chunking_strategy == "sentence":
            chunker = SentenceChunker(
                max_chunk_size=request.max_chunk_size,
                chunk_overlap=request.chunk_overlap
            )
        elif request.chunking_strategy == "paragraph":
            chunker = ParagraphChunker(
                max_chunk_size=request.max_chunk_size,
                chunk_overlap=request.chunk_overlap
            )
        else:
            chunker = SemanticChunker(
                max_chunk_size=request.max_chunk_size,
                chunk_overlap=request.chunk_overlap
            )
        
        # Update parser with chunking strategy
        self.document_parser.chunking_strategy = chunker
        
        # Parse document
        parsed_doc = self.document_parser.parse(file_path)
        
        # Ingest chunks into memory
        memories_created = 0
        for chunk in parsed_doc.chunks:
            try:
                # Create memory text with metadata
                memory_text = chunk.to_memory_text()
                
                # Add document metadata
                metadata = {
                    "source": "document",
                    "file_name": Path(file_path).name,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    **(request.metadata or {})
                }
                
                # Ingest into memory
                memory = self.memory_router.ingest(memory_text, metadata=metadata)
                if memory:
                    memories_created += 1
            except Exception as e:
                logger.error(f"Error ingesting chunk {chunk.chunk_index}: {e}")
        
        return DocumentIngestionResponse(
            file_name=Path(file_path).name,
            file_type=parsed_doc.file_type,
            chunks_created=len(parsed_doc.chunks),
            memories_created=memories_created,
            total_words=parsed_doc.total_words,
            total_chars=parsed_doc.total_chars,
            errors=parsed_doc.extraction_errors,
            success=parsed_doc.success
        )


# Create FastAPI app
def create_app(config: Optional[APIConfig] = None) -> FastAPI:
    """Create and configure FastAPI application"""
    
    config = config or APIConfig()
    wrapper = LlamaAPIWrapper(config)
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        await wrapper.startup()
        yield
        # Shutdown
        await wrapper.shutdown()
    
    app = FastAPI(
        title="Llama API Wrapper",
        description="Enhanced API wrapper for llama.cpp server with caching, metrics, and management features",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware if enabled
    if config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Request logging middleware
    if config.log_requests:
        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = datetime.now()
            response = await call_next(request)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
            return response
    
    # Routes
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "name": "Llama API Wrapper",
            "version": "1.0.0",
            "endpoints": [
                "/completions",
                "/chat/completions",
                "/embeddings",
                "/models",
                "/health",
                "/stats",
                "/docs"
            ]
        }
    
    @app.post("/v1/completions")
    @app.post("/completions")
    async def completions(request: CompletionRequest):
        """Text completion endpoint"""
        return await wrapper.completion(request)
    
    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """Chat completion endpoint"""
        return await wrapper.chat_completion(request)
    
    @app.post("/v1/embeddings")
    @app.post("/embeddings")
    async def embeddings(request: EmbeddingRequest):
        """Embeddings endpoint"""
        return await wrapper.embeddings(request)
    
    @app.get("/v1/models")
    @app.get("/models")
    async def models():
        """List available models"""
        return await wrapper.models()
    
    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return await wrapper.health()
    
    @app.get("/stats")
    async def stats():
        """Statistics endpoint"""
        return await wrapper.stats()
    
    @app.post("/admin/restart")
    async def restart_server():
        """Restart llama server (admin endpoint)"""
        if wrapper.server_manager.restart():
            return {"status": "success", "message": "Server restarted"}
        else:
            raise HTTPException(status_code=500, detail="Failed to restart server")
    
    @app.post("/admin/stop")
    async def stop_server():
        """Stop llama server (admin endpoint)"""
        if wrapper.server_manager.stop():
            return {"status": "success", "message": "Server stopped"}
        else:
            raise HTTPException(status_code=500, detail="Failed to stop server")
    
    @app.post("/documents/upload")
    async def upload_document(
        file: UploadFile = File(...),
        chunking_strategy: str = Form("semantic"),
        max_chunk_size: int = Form(2000),
        chunk_overlap: int = Form(200)
    ):
        """Upload and ingest a document into memory"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            # Copy uploaded file
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Create request object
            request = DocumentIngestionRequest(
                chunking_strategy=chunking_strategy,
                max_chunk_size=max_chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Process document
            result = await wrapper.ingest_document(tmp_path, request)
            
            return result
        finally:
            # Clean up temporary file
            try:
                Path(tmp_path).unlink()
            except:
                pass
    
    @app.post("/documents/ingest")
    async def ingest_document_path(
        file_path: str,
        request: DocumentIngestionRequest
    ):
        """Ingest a document from a file path"""
        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        result = await wrapper.ingest_document(file_path, request)
        return result
    
    @app.post("/v1/memory/search")
    @app.post("/memory/search")
    async def memory_search(request: MemorySearchRequest):
        """Search the memory database using the same process as the analyzer"""
        result = await wrapper.search_memories(request)
        return result
    
    return app


# CLI interface
def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Llama API Wrapper Server")
    parser.add_argument("command", choices=["start", "stop", "status"], 
                       help="Command to execute")
    parser.add_argument("--host", default="0.0.0.0", 
                       help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001,
                       help="Port to listen on (default: 8001)")
    parser.add_argument("--llama-url", default="http://localhost:8000",
                       help="Llama server URL (default: http://localhost:8000)")
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of worker processes")
    parser.add_argument("--daemon", action="store_true",
                       help="Run as daemon in background")
    
    args = parser.parse_args()
    
    if args.command == "start":
        # Create config from args
        config = APIConfig(
            host=args.host,
            port=args.port,
            llama_server_url=args.llama_url
        )
        
        # Create app
        app = create_app(config)
        
        # Run server
        logger.info(f"Starting API wrapper on {args.host}:{args.port}")
        
        if args.daemon:
            # Run in background (would need proper daemon implementation)
            logger.info("Daemon mode not fully implemented yet")
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level="info"
        )
    
    elif args.command == "stop":
        # Would need PID file management for proper stop
        logger.info("Stop command not implemented yet")
        
    elif args.command == "status":
        # Check if API is running
        try:
            import requests
            response = requests.get(f"http://{args.host}:{args.port}/health")
            if response.status_code == 200:
                print("API wrapper is running")
                print(response.json())
            else:
                print("API wrapper returned error")
        except:
            print("API wrapper is not running")


if __name__ == "__main__":
    main()