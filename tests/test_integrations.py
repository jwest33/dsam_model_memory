"""
Test module to verify all integrations and imports for JAM
Run this to check if all components are properly connected
"""

import sys
import traceback
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def test_import(module_name, description):
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        print(f"{GREEN}✓{RESET} {description}")
        return True
    except ImportError as e:
        print(f"{RED}✗{RESET} {description}")
        print(f"  {YELLOW}Error: {e}{RESET}")
        return False
    except Exception as e:
        print(f"{RED}✗{RESET} {description}")
        print(f"  {YELLOW}Unexpected error: {e}{RESET}")
        return False

def test_function(func, description):
    """Test if a function executes without error"""
    try:
        func()
        print(f"{GREEN}✓{RESET} {description}")
        return True
    except Exception as e:
        print(f"{RED}✗{RESET} {description}")
        print(f"  {YELLOW}Error: {e}{RESET}")
        return False

def main():
    print(f"\n{BLUE}={'='*60}{RESET}")
    print(f"{BLUE}JAM Integration Test Suite{RESET}")
    print(f"{BLUE}={'='*60}{RESET}\n")
    
    total_tests = 0
    passed_tests = 0
    
    # 1. Core Dependencies
    print(f"{BLUE}1. Core Dependencies:{RESET}")
    core_deps = [
        ("flask", "Flask web framework"),
        ("fastapi", "FastAPI framework"),
        ("click", "Click CLI framework"),
        ("requests", "Requests library"),
        ("numpy", "NumPy"),
        ("faiss", "FAISS vector store"),
        ("sentence_transformers", "Sentence Transformers"),
        ("werkzeug", "Werkzeug utilities"),
        ("pydantic", "Pydantic models"),
        ("httpx", "HTTPX async client"),
        ("uvicorn", "Uvicorn ASGI server"),
    ]
    
    for module, desc in core_deps:
        total_tests += 1
        if test_import(module, desc):
            passed_tests += 1
    
    # 2. Document Parser Dependencies
    print(f"\n{BLUE}2. Document Parser Dependencies (Optional):{RESET}")
    parser_deps = [
        ("PyPDF2", "PDF parsing support"),
        ("docx", "DOCX parsing support"),
        ("markdown", "Markdown parsing support"),
        ("bs4", "BeautifulSoup HTML parsing"),
        ("pandas", "Pandas for CSV/data files"),
        ("pytesseract", "OCR support for images"),
        ("PIL", "Image processing support"),
        ("yaml", "YAML parsing support"),
        ("toml", "TOML parsing support"),
    ]
    
    optional_available = []
    for module, desc in parser_deps:
        if test_import(module, desc):
            optional_available.append(module)
    
    # 3. Core JAM Modules
    print(f"\n{BLUE}3. Core JAM Modules:{RESET}")
    jam_modules = [
        ("agentic_memory.router", "Memory Router"),
        ("agentic_memory.storage.sql_store", "SQL Store"),
        ("agentic_memory.storage.faiss_index", "FAISS Index"),
        ("agentic_memory.config", "Configuration"),
        ("agentic_memory.config_manager", "Config Manager"),
        ("agentic_memory.extraction.llm_extractor", "LLM Extractor"),
        ("agentic_memory.extraction.multi_part_extractor", "Multi-part Extractor"),
        ("agentic_memory.retrieval", "Retrieval System"),
        ("agentic_memory.block_builder", "Block Builder"),
        ("agentic_memory.tokenization", "Tokenization"),
        ("agentic_memory.types", "Type Definitions"),
        ("agentic_memory.tools.tool_handler", "Tool Handler"),
        ("agentic_memory.cluster.concept_cluster", "Concept Clustering"),
    ]
    
    for module, desc in jam_modules:
        total_tests += 1
        if test_import(module, desc):
            passed_tests += 1
    
    # 4. New Document Parser Module
    print(f"\n{BLUE}4. Document Parser Module:{RESET}")
    total_tests += 1
    if test_import("agentic_memory.document_parser", "Document Parser"):
        passed_tests += 1
        
        # Test parser classes
        try:
            from agentic_memory.document_parser import (
                DocumentParser, 
                SemanticChunker,
                ParagraphChunker, 
                SentenceChunker,
                DocumentChunk,
                ParsedDocument
            )
            print(f"{GREEN}✓{RESET} Document parser classes available")
            passed_tests += 1
            total_tests += 1
        except Exception as e:
            print(f"{RED}✗{RESET} Document parser classes")
            print(f"  {YELLOW}Error: {e}{RESET}")
            total_tests += 1
    
    # 5. Server Components
    print(f"\n{BLUE}5. Server Components:{RESET}")
    
    # Flask app
    total_tests += 1
    try:
        from agentic_memory.server.flask_app import app
        print(f"{GREEN}✓{RESET} Flask app")
        passed_tests += 1
        
        # Check if document routes exist
        with app.test_client() as client:
            # Check if route exists (don't actually call it)
            if '/documents' in [str(rule) for rule in app.url_map.iter_rules()]:
                print(f"{GREEN}✓{RESET} Document upload route registered")
            else:
                print(f"{YELLOW}⚠{RESET} Document upload route not found")
    except Exception as e:
        print(f"{RED}✗{RESET} Flask app")
        print(f"  {YELLOW}Error: {e}{RESET}")
    
    # 6. API Wrapper
    print(f"\n{BLUE}6. API Wrapper:{RESET}")
    total_tests += 1
    try:
        from llama_api import create_app, APIConfig, LlamaAPIWrapper
        print(f"{GREEN}✓{RESET} API wrapper imports")
        passed_tests += 1
        
        # Check document ingestion models
        from llama_api import DocumentIngestionRequest, DocumentIngestionResponse
        print(f"{GREEN}✓{RESET} Document ingestion models")
        passed_tests += 1
        total_tests += 1
    except Exception as e:
        print(f"{RED}✗{RESET} API wrapper")
        print(f"  {YELLOW}Error: {e}{RESET}")
    
    # 7. CLI Module
    print(f"\n{BLUE}7. CLI Module:{RESET}")
    total_tests += 1
    try:
        from agentic_memory.cli import cli, document, memory, server
        print(f"{GREEN}✓{RESET} CLI commands imported")
        passed_tests += 1
        
        # Check if document commands are registered
        if 'document' in [cmd.name for cmd in cli.commands.values()]:
            print(f"{GREEN}✓{RESET} Document CLI commands registered")
        else:
            print(f"{YELLOW}⚠{RESET} Document CLI commands not found")
    except Exception as e:
        print(f"{RED}✗{RESET} CLI module")
        print(f"  {YELLOW}Error: {e}{RESET}")
    
    # 9. Template Files
    print(f"\n{BLUE}9. Template Files:{RESET}")
    template_dir = Path("agentic_memory/server/templates")
    required_templates = [
        "base.html",
        "index.html",
        "memories.html",
        "documents.html",
        "settings.html",
        "clusters.html"
    ]
    
    for template in required_templates:
        total_tests += 1
        template_path = template_dir / template
        if template_path.exists():
            print(f"{GREEN}✓{RESET} Template: {template}")
            passed_tests += 1
        else:
            print(f"{RED}✗{RESET} Template: {template} (missing)")
    
    # 10. File Permissions
    print(f"\n{BLUE}10. File Permissions:{RESET}")
    db_path = Path("amemory.sqlite3")
    if db_path.exists():
        if db_path.stat().st_size > 0:
            print(f"{GREEN}✓{RESET} Database file exists and has data")
        else:
            print(f"{YELLOW}⚠{RESET} Database file exists but is empty")
    else:
        print(f"{YELLOW}⚠{RESET} Database file not yet created (will be created on first use)")
    
    # Summary
    print(f"\n{BLUE}={'='*60}{RESET}")
    print(f"{BLUE}Test Summary:{RESET}")
    print(f"{BLUE}={'='*60}{RESET}")
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    if success_rate == 100:
        print(f"{GREEN}All tests passed! ({passed_tests}/{total_tests}){RESET}")
        print(f"{GREEN}✓ System is ready to run!{RESET}")
    elif success_rate >= 80:
        print(f"{GREEN}Most tests passed ({passed_tests}/{total_tests} - {success_rate:.1f}%){RESET}")
        print(f"{YELLOW}⚠ System should work but some features may be limited{RESET}")
    else:
        print(f"{RED}Many tests failed ({passed_tests}/{total_tests} - {success_rate:.1f}%){RESET}")
        print(f"{RED}✗ Please install missing dependencies{RESET}")
    
    if optional_available:
        print(f"\n{BLUE}Optional parsers available:{RESET} {', '.join(optional_available)}")
    
    print(f"\n{BLUE}Quick Start:{RESET}")
    print("1. Start LLM server: python llama_server_manager.py both")
    print("2. Start web interface: python -m agentic_memory.server.flask_app")
    print("3. Or start all: python -m agentic_memory.cli server start --all")
    
    return success_rate == 100

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)