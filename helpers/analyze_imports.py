#!/usr/bin/env python
"""
Analyze Python imports to find missing dependencies
"""
import ast
import os
import sys
import importlib.metadata
from pathlib import Path
from typing import Set, Dict, List, Tuple

# Standard library modules to ignore
STDLIB_MODULES = {
    'os', 'sys', 'json', 'asyncio', 'logging', 'subprocess', 'signal',
    'pathlib', 'typing', 'datetime', 'time', 'random', 'collections',
    'functools', 'itertools', 'contextlib', 'warnings', 'traceback',
    're', 'math', 'statistics', 'decimal', 'fractions', 'numbers',
    'string', 'textwrap', 'unicodedata', 'codecs', 'encodings',
    'io', 'pickle', 'copy', 'copyreg', 'types', 'weakref',
    'abc', 'dataclasses', 'enum', 'inspect', 'importlib',
    'ast', 'dis', 'token', 'tokenize', 'keyword', 'builtins',
    'sqlite3', 'urllib', 'http', 'email', 'html', 'xml',
    'configparser', 'argparse', 'getopt', 'optparse', 'shlex',
    'tempfile', 'glob', 'fnmatch', 'shutil', 'zipfile', 'tarfile',
    'hashlib', 'hmac', 'secrets', 'uuid', 'socket', 'ssl',
    'threading', 'multiprocessing', 'concurrent', 'queue',
    'unittest', 'doctest', 'pdb', 'profile', 'timeit',
    'platform', 'locale', 'gettext', 'base64', 'binascii',
    'struct', 'array', 'mmap', 'csv', 'operator', 'gc'
}

# Package name mappings (import name -> package name)
PACKAGE_MAPPINGS = {
    'cv2': 'opencv-python',
    'sklearn': 'scikit-learn',
    'yaml': 'PyYAML',
    'PIL': 'Pillow',
    'bs4': 'beautifulsoup4',
    'dotenv': 'python-dotenv',
    'docx': 'python-docx',
    'sentence_transformers': 'sentence-transformers',
    'duckduckgo_search': 'duckduckgo-search',
    'rank_bm25': 'rank-bm25',
    'llama_cpp': 'llama-cpp-python',
    'faiss': 'faiss-cpu',
    'PyPDF2': 'PyPDF2',
    'umap': 'umap-learn',
}

def find_imports(path: Path) -> Set[str]:
    """Find all imports in Python files"""
    imports = set()

    for root, dirs, files in os.walk(path):
        # Skip virtual environments and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and 'venv' not in d]

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        tree = ast.parse(content)

                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for n in node.names:
                                    module = n.name.split('.')[0]
                                    imports.add(module)
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    module = node.module.split('.')[0]
                                    imports.add(module)
                except (SyntaxError, UnicodeDecodeError) as e:
                    print(f"⚠️  Error parsing {filepath}: {e}")
                except Exception as e:
                    print(f"⚠️  Unexpected error in {filepath}: {e}")

    return imports

def check_installed(module: str) -> Tuple[bool, str]:
    """Check if a module is installed and get its version"""
    # Try different package names
    package_names = [
        module,
        module.replace('_', '-'),
        PACKAGE_MAPPINGS.get(module, module)
    ]

    for pkg_name in package_names:
        try:
            version = importlib.metadata.version(pkg_name)
            return True, version
        except importlib.metadata.PackageNotFoundError:
            continue

    return False, ""

def load_requirements(filepath: str) -> Dict[str, str]:
    """Load requirements from requirements.txt"""
    requirements = {}
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse package name from requirement line
                    if '>=' in line:
                        pkg = line.split('>=')[0].strip()
                    elif '==' in line:
                        pkg = line.split('==')[0].strip()
                    elif '>' in line:
                        pkg = line.split('>')[0].strip()
                    elif '<' in line:
                        pkg = line.split('<')[0].strip()
                    elif '[' in line:
                        pkg = line.split('[')[0].strip()
                    else:
                        pkg = line.strip()
                    requirements[pkg.lower()] = line
    return requirements

def main():
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    print("🔍 Analyzing Python imports in the project...\n")

    # Find all imports
    project_imports = find_imports(Path('agentic_memory'))
    project_imports.update(find_imports(Path('data')))

    # Add imports from standalone scripts
    for script in Path('.').glob('*.py'):
        if script.name != 'analyze_imports.py':
            try:
                with open(script, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for n in node.names:
                                project_imports.add(n.name.split('.')[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                project_imports.add(node.module.split('.')[0])
            except:
                pass

    # Find local project modules
    local_modules = set()
    for root, dirs, files in os.walk('.'):
        if '.venv' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py') and file != 'analyze_imports.py':
                module_name = file[:-3]
                local_modules.add(module_name)

    # Also add subdirectory modules and known local imports
    local_modules.update(['agentic_memory', 'data', 'extraction', 'storage',
                         'cluster', 'tools', 'server', 'embedding', 'attention',
                         'block_builder', 'config', 'config_manager', 'retrieval',
                         'settings_manager', 'tokenization', 'router', 'types',
                         'llama_agent_websearch', 'llama_api', 'llama_embedder',
                         'llama_server_manager', 'memory_search_tool', 'ddgs'])

    # Filter out standard library, local modules, and false positives
    third_party = {imp for imp in project_imports
                   if imp not in STDLIB_MODULES
                   and imp not in local_modules
                   and not imp.startswith('agentic_memory')
                   and not imp.startswith('data')
                   and imp not in ['__future__', 'atexit', 'mimetypes', 'tkinter', 'base']
                   and imp != '__main__'}

    # Load current requirements
    requirements = load_requirements('requirements.txt')

    # Check installation status
    installed = []
    missing = []
    in_requirements = []
    not_in_requirements = []

    for module in sorted(third_party):
        is_installed, version = check_installed(module)
        package_name = PACKAGE_MAPPINGS.get(module, module).lower()

        if is_installed:
            installed.append((module, version))
            if package_name in requirements or package_name.replace('-', '_') in requirements:
                in_requirements.append(module)
            else:
                not_in_requirements.append(module)
        else:
            missing.append(module)

    # Display results
    print("📦 INSTALLED PACKAGES:")
    print("-" * 50)
    for module, version in installed:
        status = "✓" if module in in_requirements else "⚠️ "
        print(f"{status} {module:30} {version}")

    if missing:
        print("\n❌ MISSING PACKAGES (imported but not installed):")
        print("-" * 50)
        for module in missing:
            package_name = PACKAGE_MAPPINGS.get(module, module)
            print(f"  {module:30} → pip install {package_name}")

    if not_in_requirements:
        print("\n⚠️  INSTALLED BUT NOT IN requirements.txt:")
        print("-" * 50)
        for module in not_in_requirements:
            package_name = PACKAGE_MAPPINGS.get(module, module)
            print(f"  {module:30} → Add: {package_name}")

    # Find unused requirements
    print("\n📝 PACKAGES IN requirements.txt:")
    print("-" * 50)
    imported_packages = {PACKAGE_MAPPINGS.get(m, m).lower() for m in third_party}
    imported_packages.update({m.lower() for m in third_party})

    used_count = 0
    unused_count = 0
    for req_pkg, req_line in sorted(requirements.items()):
        is_used = req_pkg in imported_packages or req_pkg.replace('-', '_') in imported_packages
        if is_used:
            print(f"✓ {req_line}")
            used_count += 1
        else:
            print(f"? {req_line} (not directly imported)")
            unused_count += 1

    print(f"\n📊 SUMMARY:")
    print(f"  Total imports found: {len(third_party)}")
    print(f"  Installed: {len(installed)}")
    print(f"  Missing: {len(missing)}")
    print(f"  Requirements used: {used_count}/{len(requirements)}")

    if missing:
        print(f"\n💡 To install missing packages:")
        packages = [PACKAGE_MAPPINGS.get(m, m) for m in missing]
        print(f"  pip install {' '.join(packages)}")

    # Generate complete requirements.txt
    print("\n" + "=" * 50)
    print("📋 COMPLETE REQUIREMENTS.TXT (copy and paste):")
    print("=" * 50)

    # Collect all needed packages with versions
    all_packages = {}

    # Add installed packages that are imported
    for module, version in installed:
        package_name = PACKAGE_MAPPINGS.get(module, module)
        if package_name == 'sklearn':
            package_name = 'scikit-learn'
        # Use existing version spec from requirements.txt if available
        if package_name.lower() in requirements:
            all_packages[package_name] = requirements[package_name.lower()]
        elif package_name.replace('-', '_').lower() in requirements:
            all_packages[package_name] = requirements[package_name.replace('-', '_').lower()]
        else:
            # Add with current version
            all_packages[package_name] = f"{package_name}>={version}"

    # Add packages from requirements.txt that are indirect dependencies
    indirect_deps = {
        'jinja2': 'jinja2>=3.1.4',
        'pytest': 'pytest>=7.4.0',
        'pytest-asyncio': 'pytest-asyncio>=0.21.1',
        'scipy': 'scipy>=1.11.0',
        'structlog': 'structlog>=24.1.0',
        'rank-bm25': 'rank-bm25>=0.2.2',
        'umap-learn': 'umap-learn>=0.5.9',
        'python-dotenv': 'python-dotenv>=1.0.0',
        'tiktoken': 'tiktoken>=0.5.0',
    }

    for pkg, spec in indirect_deps.items():
        if pkg.lower() in requirements:
            all_packages[pkg] = requirements[pkg.lower()]
        else:
            all_packages[pkg] = spec

    # Sort and print
    sorted_packages = sorted(all_packages.items(), key=lambda x: x[0].lower())

    print("# Core Framework")
    framework_pkgs = ['flask', 'jinja2', 'fastapi', 'uvicorn', 'pydantic', 'structlog']
    for name, spec in sorted_packages:
        if any(name.lower().startswith(f) for f in framework_pkgs):
            print(spec)

    print("\n# Scientific Computing")
    science_pkgs = ['numpy', 'scipy', 'scikit-learn', 'umap-learn', 'pandas']
    for name, spec in sorted_packages:
        if any(p in name.lower() for p in science_pkgs):
            print(spec)

    print("\n# Deep Learning")
    dl_pkgs = ['torch', 'networkx']
    for name, spec in sorted_packages:
        if any(p in name.lower() for p in dl_pkgs):
            print(spec)

    print("\n# Vector Search & Text Processing")
    search_pkgs = ['faiss', 'rank-bm25', 'tiktoken']
    for name, spec in sorted_packages:
        if any(p in name.lower() for p in search_pkgs):
            print(spec)

    print("\n# LLM Integration")
    llm_pkgs = ['llama-cpp']
    for name, spec in sorted_packages:
        if any(p in name.lower() for p in llm_pkgs):
            print(spec)

    print("\n# Web & API")
    web_pkgs = ['requests', 'httpx', 'click', 'aiohttp']
    for name, spec in sorted_packages:
        if any(p in name.lower() for p in web_pkgs):
            print(spec)

    print("\n# Document Parsing")
    doc_pkgs = ['pypdf2', 'python-docx', 'markdown', 'beautifulsoup', 'pytesseract',
                'pillow', 'pyyaml', 'toml', 'trafilatura']
    for name, spec in sorted_packages:
        if any(p in name.lower() for p in doc_pkgs):
            print(spec)

    print("\n# Utilities")
    util_pkgs = ['psutil', 'python-dotenv', 'tqdm', 'matplotlib', 'duckduckgo']
    for name, spec in sorted_packages:
        if any(p in name.lower() for p in util_pkgs):
            print(spec)

    print("\n# Development & Testing")
    dev_pkgs = ['pytest']
    for name, spec in sorted_packages:
        if any(p in name.lower() for p in dev_pkgs):
            print(spec)

    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
