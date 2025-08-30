#!/usr/bin/env python3
"""
Virtual Environment Setup Script for Dual-Space Memory System
Automatically creates and configures a virtual environment with all dependencies
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def get_venv_path():
    """Get the virtual environment path"""
    return Path(".venv")

def get_python_executable():
    """Get the appropriate Python executable for the platform"""
    venv_path = get_venv_path()
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"

def get_pip_executable():
    """Get the appropriate pip executable for the platform"""
    venv_path = get_venv_path()
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "pip.exe"
    else:
        return venv_path / "bin" / "pip"

def create_venv():
    """Create virtual environment"""
    venv_path = get_venv_path()
    
    if venv_path.exists():
        print(f"Virtual environment already exists at {venv_path}")
        return True
    
    print(f"Creating virtual environment at {venv_path}...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        print(f"Virtual environment created successfully at {venv_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to create virtual environment: {e}")
        return False

def upgrade_pip():
    """Upgrade pip to latest version"""
    python_exe = get_python_executable()
    print("Upgrading pip...")
    try:
        # Use python -m pip instead of pip directly to avoid permission issues
        subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print("Pip upgraded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not upgrade pip: {e}")
        print("Continuing with existing pip version...")
        return True  # Continue anyway since pip is already installed

def install_requirements():
    """Install all requirements"""
    python_exe = get_python_executable()
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("requirements.txt not found")
        return False
    
    print("Installing requirements...")
    print("Note: This may take several minutes, especially for torch and transformers")
    
    try:
        # Use python -m pip for consistency and to avoid permission issues
        subprocess.run([str(python_exe), "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("All requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requirements: {e}")
        print("\nTroubleshooting tips:")
        print("1. If torch fails, you may need to install it separately:")
        print(f"   {python_exe} -m pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("2. For CUDA support, visit: https://pytorch.org/get-started/locally/")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if env_file.exists():
        print(".env file already exists")
        return True
    
    # Create example content
    example_content = """# Environment Configuration for Dual-Space Memory System

# Set offline mode to avoid HuggingFace rate limits
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1

# Optional: Set custom ChromaDB path
# CHROMADB_PATH=./state/chromadb

# Optional: Set custom benchmark datasets path
# BENCHMARK_DATASETS_PATH=./benchmark_datasets

# Optional: Enable debug mode
# DEBUG=1

# Optional: Set Flask configuration
# FLASK_ENV=development
# FLASK_DEBUG=0
# FLASK_PORT=5000
"""
    
    try:
        # Write example file
        with open(env_example, 'w') as f:
            f.write(example_content)
        print(f"Created {env_example}")
        
        # Copy to .env
        with open(env_file, 'w') as f:
            f.write(example_content)
        print(f"Created {env_file} (customize as needed)")
        
        return True
    except Exception as e:
        print(f"Failed to create .env file: {e}")
        return False

def print_activation_instructions():
    """Print instructions for activating the virtual environment"""
    venv_path = get_venv_path()
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    
    print("\nTo activate the virtual environment, run:")
    
    if platform.system() == "Windows":
        print(f"  .venv\\Scripts\\activate")
        print("\nOr in PowerShell:")
        print(f"  .venv\\Scripts\\Activate.ps1")
    else:
        print(f"  source .venv/bin/activate")
    
    print("\nTo deactivate, run:")
    print("  deactivate")
    
    print("\nNext steps:")
    print("1. Activate the virtual environment")
    print("2. Run the web interface: python run_web.py")
    print("3. Access at http://localhost:5000")
    
    print("\nFor more information, see README.md")

def main():
    """Main setup function"""
    print("="*60)
    print("Dual-Space Memory System - Virtual Environment Setup")
    print("="*60)
    print()
    
    # Check Python version
    if sys.version_info < (3, 10):
        print(f"Python 3.10+ required (current: {sys.version})")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.system()}")
    print()
    
    # Create virtual environment
    if not create_venv():
        sys.exit(1)
    
    # Upgrade pip
    if not upgrade_pip():
        print("Warning: pip upgrade failed, continuing anyway...")
    
    # Install requirements
    if not install_requirements():
        print("\nPartial installation completed.")
        print("Please install missing requirements manually.")
    
    # Create .env file
    create_env_file()
    
    # Print activation instructions
    print_activation_instructions()

if __name__ == "__main__":
    main()
