"""
Automatic LoRA Adapter Creator for Tool Calling
Generates, tests, and fine-tunes LoRA adapters for Qwen3-4B-Instruct-2507 (July 2025)
using llama.cpp server for tool calling capabilities.
"""

import json
import requests
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
try:
    from peft import prepare_model_for_kbit_training
except ImportError:
    # Fallback for older PEFT versions
    def prepare_model_for_kbit_training(model):
        return model
from datasets import Dataset as HFDataset
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, asdict, field
import random
import time
from pathlib import Path
import yaml
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import re
from tqdm import tqdm
import hashlib
from datetime import datetime
import subprocess
import os
import tempfile
import shutil
import platform
import sys
import importlib
import psutil
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DependencyChecker:
    """Check all required dependencies and system requirements before training"""
    
    @staticmethod
    def check_python_version():
        """Check Python version"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            return False, f"Python 3.8+ required, found {version.major}.{version.minor}.{version.micro}"
        return True, f"Python {version.major}.{version.minor}.{version.micro} ✓"
    
    @staticmethod
    def check_required_modules():
        """Check all required Python modules"""
        required_modules = {
            'torch': 'torch>=2.0.0',
            'transformers': 'transformers>=4.30.0',
            'peft': 'peft>=0.4.0',
            'datasets': 'datasets',
            'numpy': 'numpy',
            'aiohttp': 'aiohttp',
            'tqdm': 'tqdm',
            'psutil': 'psutil',
            'yaml': 'pyyaml',
            'requests': 'requests'
        }
        
        missing_modules = []
        warnings_list = []
        
        for module_name, pip_name in required_modules.items():
            try:
                importlib.import_module(module_name)
            except ImportError:
                missing_modules.append(pip_name)
        
        # Check optional modules
        optional_modules = {
            'bitsandbytes': 'bitsandbytes (for 4-bit quantization)',
            'accelerate': 'accelerate (for distributed training)'
        }
        
        for module_name, description in optional_modules.items():
            try:
                importlib.import_module(module_name)
            except ImportError:
                warnings_list.append(f"Optional: {description} not installed")
        
        if missing_modules:
            return False, f"Missing required modules: {', '.join(missing_modules)}\nInstall with: pip install {' '.join(missing_modules)}"
        
        message = "All required modules installed ✓"
        if warnings_list:
            message += f"\n  Warnings: {'; '.join(warnings_list)}"
        return True, message
    
    @staticmethod
    def check_cuda_availability():
        """Check CUDA availability and GPU memory"""
        if not torch.cuda.is_available():
            return False, "CUDA not available. Training will be slow on CPU"
        
        device_count = torch.cuda.device_count()
        total_memory = 0
        device_info = []
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            total_memory += memory_gb
            device_info.append(f"{props.name} ({memory_gb:.1f}GB)")
        
        if total_memory < 6:
            return False, f"Insufficient GPU memory: {total_memory:.1f}GB (minimum 6GB recommended)"
        
        return True, f"CUDA available: {', '.join(device_info)} ✓"
    
    @staticmethod
    def check_disk_space(required_gb=10):
        """Check available disk space"""
        try:
            disk_usage = psutil.disk_usage('.')
            available_gb = disk_usage.free / (1024**3)
            
            if available_gb < required_gb:
                return False, f"Insufficient disk space: {available_gb:.1f}GB available, {required_gb}GB required"
            
            return True, f"Disk space: {available_gb:.1f}GB available ✓"
        except Exception as e:
            return None, f"Could not check disk space: {e}"
    
    @staticmethod
    def check_memory(required_gb=8):
        """Check available RAM"""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            
            if available_gb < required_gb:
                return False, f"Insufficient RAM: {available_gb:.1f}GB available, {required_gb}GB required"
            
            return True, f"RAM: {available_gb:.1f}GB/{total_gb:.1f}GB available ✓"
        except Exception as e:
            return None, f"Could not check memory: {e}"
    
    @staticmethod
    def check_llama_cpp_server(base_url="http://localhost:8000"):
        """Check if llama.cpp server is accessible"""
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code == 200:
                return True, f"llama.cpp server accessible at {base_url} ✓"
        except:
            pass
        
        return None, f"llama.cpp server not running at {base_url} (will start automatically)"
    
    @staticmethod
    def check_model_files(model_path: str, model_name: str):
        """Check if model files exist"""
        checks = []
        
        # Check GGUF model file
        if model_path and Path(model_path).exists():
            size_gb = Path(model_path).stat().st_size / (1024**3)
            checks.append((True, f"GGUF model found: {Path(model_path).name} ({size_gb:.1f}GB) ✓"))
        elif model_path:
            checks.append((False, f"GGUF model not found at: {model_path}"))
        
        # Check if HuggingFace model can be accessed
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_cache_pattern = model_name.replace("/", "--")
        cached_models = list(cache_dir.glob(f"*{model_cache_pattern}*")) if cache_dir.exists() else []
        
        if cached_models:
            checks.append((True, f"HuggingFace model cached: {model_name} ✓"))
        else:
            checks.append((None, f"HuggingFace model not cached: {model_name} (will download on first use)"))
        
        return checks
    
    @staticmethod
    def check_llama_cpp_tools():
        """Check for llama.cpp conversion tools"""
        # Check for convert script
        llama_cpp_path = os.environ.get('LLAMA_CPP_PATH')
        
        possible_paths = []
        if llama_cpp_path:
            possible_paths.extend([
                os.path.join(llama_cpp_path, "convert_lora_to_gguf.py"),
                os.path.join(llama_cpp_path, "convert-lora-to-gguf.py")
            ])
        
        possible_paths.extend([
            "convert-lora-to-gguf.py",
            "convert_lora_to_gguf.py",
            "./llama.cpp/convert-lora-to-gguf.py",
            "./llama.cpp/convert_lora_to_gguf.py"
        ])
        
        for path in possible_paths:
            if Path(path).exists():
                return True, f"LoRA conversion script found: {path} ✓"
        
        message = "LoRA conversion script not found (GGUF export will be skipped)"
        if not llama_cpp_path:
            message += "\n  Tip: Set LLAMA_CPP_PATH environment variable to your llama.cpp directory"
        
        return None, message
    
    @staticmethod
    def check_llama_server_binary():
        """Check if llama-server binary exists"""
        server_cmd = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
        
        if shutil.which(server_cmd):
            return True, f"llama-server binary found in PATH ✓"
        
        # Check LLAMA_CPP_PATH
        llama_cpp_path = os.environ.get('LLAMA_CPP_PATH')
        if llama_cpp_path:
            server_paths = [
                os.path.join(llama_cpp_path, "build", "bin", "release", server_cmd),
                os.path.join(llama_cpp_path, "build", "bin", server_cmd),
                os.path.join(llama_cpp_path, server_cmd)
            ]
            
            for path in server_paths:
                if Path(path).exists():
                    return True, f"llama-server binary found at: {path} ✓"
        
        # If server is already running, that's good enough
        try:
            response = requests.get("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                return True, f"llama-server already running (binary location unknown) ✓"
        except:
            pass
        
        return False, f"{server_cmd} not found. Please install llama.cpp and add to PATH or set LLAMA_CPP_PATH"
    
    @staticmethod
    def run_all_checks(config: 'TrainingConfig') -> Tuple[bool, List[str]]:
        """Run all dependency checks and return results"""
        results = []
        has_errors = False
        has_warnings = False
        
        print("\n" + "="*60)
        print(" DEPENDENCY AND SYSTEM CHECK")
        print("="*60 + "\n")
        
        # Python version
        success, message = DependencyChecker.check_python_version()
        results.append((success, message))
        if not success:
            has_errors = True
        
        # Required modules
        success, message = DependencyChecker.check_required_modules()
        results.append((success, message))
        if not success:
            has_errors = True
        
        # CUDA
        success, message = DependencyChecker.check_cuda_availability()
        results.append((success, message))
        if success is False:
            has_warnings = True
        
        # System resources
        success, message = DependencyChecker.check_memory()
        results.append((success, message))
        if success is False:
            has_errors = True
        
        success, message = DependencyChecker.check_disk_space()
        results.append((success, message))
        if success is False:
            has_errors = True
        
        # llama.cpp components
        success, message = DependencyChecker.check_llama_server_binary()
        results.append((success, message))
        if success is False:
            has_errors = True
        
        success, message = DependencyChecker.check_llama_cpp_server(config.llama_cpp_url)
        results.append((success, message))
        
        success, message = DependencyChecker.check_llama_cpp_tools()
        results.append((success, message))
        if success is None:
            has_warnings = True
        
        # Model files
        model_checks = DependencyChecker.check_model_files(config.llama_cpp_model_path, config.model_name)
        for success, message in model_checks:
            results.append((success, message))
            if success is False:
                has_errors = True
            elif success is None:
                has_warnings = True
        
        # Print results
        for success, message in results:
            if success is True:
                print(f"Success: {message}")
            elif success is False:
                print(f"Fail: {message}")
            else:  # None = warning
                print(f"Warning:  {message}")
        
        print("\n" + "="*60)
        
        if has_errors:
            print(" ERRORS FOUND: Please fix the issues above before proceeding")
            print("="*60 + "\n")
            return False, results
        elif has_warnings:
            print("  WARNINGS: Training can proceed, but some features may be limited")
            print("="*60 + "\n")
            return True, results
        else:
            print(" ALL CHECKS PASSED: Ready to start training!")
            print("="*60 + "\n")
            return True, results


@dataclass
class ToolDefinition:
    """Definition of a tool for function calling"""
    name: str
    description: str
    parameters: Dict[str, Any]
    returns: Dict[str, Any]
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "returns": self.returns
            }
        }

@dataclass
class TrainingConfig:
    """Configuration for training LoRA adapters"""
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    llama_cpp_url: str = "http://localhost:8000"
    llama_cpp_model_path: str = r"C:\models\Qwen3-4B-Instruct-2507\Qwen3-4B-Instruct-2507-F16.gguf"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    num_epochs: int = 3
    max_seq_length: int = 2048
    warmup_steps: int = 100
    save_steps: int = 200
    eval_steps: int = 100
    output_dir: str = "./lora_adapters"
    cache_dir: str = "./cache"
    num_synthetic_examples: int = 500
    test_split_ratio: float = 0.2
    temperature: float = 0.7
    top_p: float = 0.9
    max_retries: int = 3
    quantization_bits: int = 0  # Set to 0 to disable quantization, 4 for 4-bit


@dataclass
class ToolCallExample:
    """Example of a tool call"""
    user_prompt: str
    tool_calls: List[Dict[str, Any]]
    tool_responses: List[Dict[str, Any]]
    assistant_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LlamaCppClient:
    """Client for interacting with llama.cpp server"""
    
    def __init__(self, base_url: str, model_path: str = None):
        self.base_url = base_url.rstrip('/')
        # Extract port from URL if provided
        from urllib.parse import urlparse
        parsed = urlparse(self.base_url)
        self.port = parsed.port if parsed.port else 8000
        self.model_path = model_path
        self.session = None
        self._server_process = None
        
    async def start_server(self):
        """Start llama.cpp server if not running"""
        try:
            # Check if server is already running
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as resp:
                    if resp.status == 200:
                        logger.info("Llama.cpp server already running")
                        return
        except:
            pass
        
        if not self.model_path:
            raise ValueError("Model path not provided")
        if not Path(self.model_path).exists():
            raise ValueError(f"Model file not found at {self.model_path}. Please download and convert the model first.")
        
        # Start llama.cpp server with platform-specific command
        server_cmd = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
        
        # Check if server command exists in PATH
        if not shutil.which(server_cmd):
            raise RuntimeError(f"{server_cmd} not found in PATH. Please install llama.cpp first.")
        
        cmd = [
            server_cmd,
            "-m", self.model_path,
            "-c", "4096",
            "-n", "2048",
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "-ngl", "999",  # Use ALL GPU layers
            "--n-batch", "512",  # Larger batch size for faster processing
            "--n-ubatch", "512",  # Larger micro-batch
            "--chat-template", "chatml",  # More compatible template
            "--flash-attn"  # Enable flash attention if available
        ]
        
        logger.info(f"Starting llama.cpp server with command: {' '.join(cmd)}")
        self._server_process = subprocess.Popen(cmd)
        
        # Wait for server to start
        for _ in range(30):
            await asyncio.sleep(1)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/health") as resp:
                        if resp.status == 200:
                            logger.info("Llama.cpp server started successfully")
                            return
            except:
                continue
        
        raise RuntimeError("Failed to start llama.cpp server")
    
    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7,
                      top_p: float = 0.9, stop: List[str] = None) -> str:
        """Generate text using llama.cpp server"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop or [],
            "stream": False
        }
        
        try:
            async with self.session.post(f"{self.base_url}/completion", json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("content", "")
                else:
                    error = await resp.text()
                    raise RuntimeError(f"Generation failed: {error}")
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error during generation: {e}")
            if self.session:
                await self.session.close()
                self.session = None
            raise
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    async def generate_with_tools(self, messages: List[Dict[str, str]], 
                                 tools: List[ToolDefinition]) -> Dict[str, Any]:
        """Generate with tool calling support"""
        # Format prompt for tool calling
        system_prompt = self._format_tools_prompt(tools)
        full_prompt = self._format_messages(messages, system_prompt)
        
        response = await self.generate(
            full_prompt,
            max_tokens=1024,
            temperature=0.1,  # Lower temperature for tool calling
            stop=["<|im_end|>", "\n\nHuman:", "\n\nAssistant:"]
        )
        
        # Parse tool calls from response
        tool_calls = self._parse_tool_calls(response)
        
        return {
            "content": response,
            "tool_calls": tool_calls
        }
    
    def _format_tools_prompt(self, tools: List[ToolDefinition]) -> str:
        """Format tools into system prompt"""
        tools_json = [tool.to_json_schema() for tool in tools]
        return f"""You are a helpful assistant with access to the following tools:

{json.dumps(tools_json, indent=2)}

When you need to use a tool, respond with a JSON object in the following format:
{{
    "tool_call": {{
        "name": "tool_name",
        "arguments": {{...}}
    }}
}}

After receiving the tool response, provide your final answer to the user."""
    
    def _format_messages(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        """Format messages for Qwen2.5 chat template"""
        formatted = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        formatted += "<|im_start|>assistant\n"
        return formatted
    
    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from response"""
        tool_calls = []
        
        # Look for JSON tool call patterns with better nested handling
        json_pattern = r'\{(?:[^{}]|\{[^{}]*\})*"tool_call"(?:[^{}]|\{[^{}]*\})*\}'
        matches = re.findall(json_pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                if "tool_call" in parsed:
                    tool_calls.append(parsed["tool_call"])
            except json.JSONDecodeError:
                continue
        
        return tool_calls
    
    async def close(self):
        """Close the client and stop server if started by us"""
        if self.session:
            await self.session.close()
            self.session = None
        
        if self._server_process:
            self._server_process.terminate()
            try:
                # Use asyncio-compatible wait
                await asyncio.wait_for(
                    asyncio.create_task(asyncio.to_thread(self._server_process.wait)), 
                    timeout=10
                )
            except asyncio.TimeoutError:
                self._server_process.kill()
            logger.info("Llama.cpp server stopped")


class SyntheticDataGenerator:
    """Generate synthetic training data for tool calling"""
    
    def __init__(self, llm_client: LlamaCppClient, tools: List[ToolDefinition]):
        self.llm_client = llm_client
        self.tools = tools
    
    async def generate_examples(self, num_examples: int) -> List[ToolCallExample]:
        """Generate synthetic examples for tool calling"""
        examples = []
        
        for i in tqdm(range(num_examples), desc="Generating synthetic examples"):
            example = await self._generate_single_example()
            if example:
                examples.append(example)
                
        return examples
    
    async def _generate_single_example(self) -> Optional[ToolCallExample]:
        """Generate a single synthetic example"""
        # Select random tool(s)
        num_tools = random.randint(1, min(3, len(self.tools)))
        selected_tools = random.sample(self.tools, num_tools)
        
        # Generate user prompt that would require these tools
        prompt = await self._generate_user_prompt(selected_tools)
        
        # Generate tool calls
        tool_calls = await self._generate_tool_calls(prompt, selected_tools)
        
        # Simulate tool responses
        tool_responses = self._simulate_tool_responses(tool_calls)
        
        # Generate final assistant response
        assistant_response = await self._generate_assistant_response(
            prompt, tool_calls, tool_responses
        )
        
        return ToolCallExample(
            user_prompt=prompt,
            tool_calls=tool_calls,
            tool_responses=tool_responses,
            assistant_response=assistant_response,
            metadata={
                "tools_used": [t.name for t in selected_tools],
                "generation_timestamp": datetime.now().isoformat()
            }
        )
    
    async def _generate_user_prompt(self, tools: List[ToolDefinition]) -> str:
        """Generate a user prompt that would require the given tools"""
        tools_desc = ", ".join([f"{t.name} ({t.description})" for t in tools])
        
        generation_prompt = f"""Generate a realistic user question that would require using these tools: {tools_desc}

The question should be natural and specific. Output only the user question, nothing else.

User question:"""
        
        response = await self.llm_client.generate(
            generation_prompt,
            max_tokens=100,  # Shorter for faster generation
            temperature=0.5  # Lower temperature for more consistent training data
        )
        
        return response.strip()
    
    async def _generate_tool_calls(self, user_prompt: str, 
                                  tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Generate appropriate tool calls for the user prompt"""
        messages = [{"role": "user", "content": user_prompt}]
        
        response = await self.llm_client.generate_with_tools(messages, tools)
        return response.get("tool_calls", [])
    
    def _simulate_tool_responses(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simulate responses from tool calls"""
        responses = []
        
        for call in tool_calls:
            # Generate mock response based on tool type
            response = {
                "tool_name": call.get("name"),
                "result": self._generate_mock_result(call)
            }
            responses.append(response)
        
        return responses
    
    def _generate_mock_result(self, tool_call: Dict[str, Any]) -> Any:
        """Generate mock result for a tool call"""
        tool_name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})
        
        # Generate more realistic mock results based on tool type and arguments
        if "search" in tool_name.lower():
            query = args.get("query", "general topic")
            return {
                "results": [
                    {
                        "title": f"Understanding {query}: A Comprehensive Guide",
                        "snippet": f"This article provides detailed information about {query}, including recent developments and expert insights.",
                        "url": f"https://example.com/article-{i+1}"
                    }
                    for i in range(3)
                ]
            }
        elif "calculate" in tool_name.lower() or "math" in tool_name.lower():
            expr = args.get("expression", "")
            # Try to actually evaluate simple expressions safely
            try:
                if all(c in "0123456789+-*/() ." for c in expr):
                    result = eval(expr)
                else:
                    result = random.uniform(1, 100)
            except:
                result = random.uniform(1, 100)
            return {"result": round(result, 2), "expression": expr}
        elif "weather" in tool_name.lower():
            location = args.get("location", "Unknown City")
            return {
                "location": location,
                "temperature": random.randint(15, 30),
                "condition": random.choice(["sunny", "partly cloudy", "cloudy", "light rain"]),
                "humidity": random.randint(40, 80),
                "wind_speed": random.randint(5, 25)
            }
        elif "reminder" in tool_name.lower():
            return {
                "success": True,
                "reminder_id": f"rem_{random.randint(1000, 9999)}",
                "message": args.get("message", "Reminder set"),
                "time": args.get("time", "")
            }
        else:
            return {"status": "success", "data": f"Processed request for {tool_name}", "args_received": args}
    
    async def _generate_assistant_response(self, user_prompt: str,
                                          tool_calls: List[Dict[str, Any]],
                                          tool_responses: List[Dict[str, Any]]) -> str:
        """Generate final assistant response after tool calls"""
        context = f"""User asked: {user_prompt}

Tool calls made: {json.dumps(tool_calls, indent=2)}

Tool responses: {json.dumps(tool_responses, indent=2)}

Based on the tool responses, provide a helpful answer to the user's question. Be specific and use the information from the tools.

Assistant response:"""
        
        response = await self.llm_client.generate(
            context,
            max_tokens=200,  # Shorter for faster generation
            temperature=0.6
        )
        
        return response.strip()


class ToolCallingDataset(Dataset):
    """PyTorch dataset for tool calling examples"""
    
    def __init__(self, examples: List[ToolCallExample], tokenizer, max_length: int = 2048):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format the conversation for training
        conversation = self._format_conversation(example)
        
        # Tokenize
        encoding = self.tokenizer(
            conversation,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()  # For causal LM
        }
    
    def _format_conversation(self, example: ToolCallExample) -> str:
        """Format example into training conversation"""
        conversation = f"<|im_start|>user\n{example.user_prompt}<|im_end|>\n"
        
        # Add tool calls
        if example.tool_calls:
            tool_calls_str = json.dumps(example.tool_calls, indent=2)
            conversation += f"<|im_start|>assistant\nI'll help you with that. Let me use the following tools:\n```json\n{tool_calls_str}\n```<|im_end|>\n"
        
        # Add tool responses
        if example.tool_responses:
            for response in example.tool_responses:
                conversation += f"<|im_start|>tool\n{json.dumps(response)}<|im_end|>\n"
        
        # Add final response
        conversation += f"<|im_start|>assistant\n{example.assistant_response}<|im_end|>"
        
        return conversation


class LoRATrainer:
    """Trainer for LoRA adapters"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def cleanup_model(self):
        """Release model from memory"""
        if self.peft_model:
            del self.peft_model
            self.peft_model = None
        if self.model:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def setup_model(self):
        """Setup model and tokenizer with LoRA"""
        logger.info(f"Loading model {self.config.model_name}")
        
        # Configure quantization for memory efficiency if available
        quantization_config = None
        if BITSANDBYTES_AVAILABLE and self.config.quantization_bits == 4:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                logger.info("Using 4-bit quantization")
            except Exception as e:
                logger.warning(f"Could not setup quantization: {e}")
                logger.info("Falling back to full precision")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )
        
        # Set pad token properly for Qwen models
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = "<|endoftext|>"
            self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if self.tokenizer.pad_token_id == self.tokenizer.unk_token_id:
                # Fallback if token doesn't exist
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Load model
        model_kwargs = {
            "cache_dir": self.config.cache_dir,
            "trust_remote_code": True,
            "torch_dtype": torch.float16  # Use FP16 for efficiency
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            # Without quantization, load to GPU directly
            model_kwargs["device_map"] = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Prepare model for training
        if quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            # Enable gradient checkpointing for memory efficiency if available
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            else:
                logger.info("Gradient checkpointing not available for this model")
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
    def train(self, train_dataset: ToolCallingDataset, eval_dataset: ToolCallingDataset):
        """Train the LoRA adapter"""
        if not self.peft_model:
            self.setup_model()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            logging_steps=10,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy="steps",  # Changed from evaluation_strategy
            save_strategy="steps",
            load_best_model_at_end=True,
            learning_rate=self.config.learning_rate,
            fp16=torch.cuda.is_available(),  # Only use FP16 if CUDA available
            report_to=[],  # Disable reporting for now
            logging_dir=f"{self.config.output_dir}/logs",
            push_to_hub=False,
            remove_unused_columns=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        logger.info("Starting training...")
        trainer.train()
        
        # Save the final model
        self.save_adapter(f"{self.config.output_dir}/final_adapter")
        
        return trainer.state.log_history
    
    def save_adapter(self, path: str):
        """Save LoRA adapter"""
        if self.peft_model:
            self.peft_model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info(f"Adapter saved to {path}")
    
    def export_to_gguf(self, adapter_path: str, output_path: str):
        """Export LoRA adapter to GGUF format for llama.cpp"""
        logger.info(f"Exporting adapter to GGUF format: {output_path}")
        
        # Look for convert script in common locations
        convert_script = None
        possible_paths = []
        
        # Check environment variable first
        llama_cpp_path = os.environ.get('LLAMA_CPP_PATH')
        if llama_cpp_path:
            possible_paths.append(os.path.join(llama_cpp_path, "convert_lora_to_gguf.py"))
            possible_paths.append(os.path.join(llama_cpp_path, "convert-lora-to-gguf.py"))
        
        # Add common locations
        possible_paths.extend([
            "convert-lora-to-gguf.py",
            "convert_lora_to_gguf.py",
            "./llama.cpp/convert-lora-to-gguf.py",
            "./llama.cpp/convert_lora_to_gguf.py",
            os.path.join(os.path.dirname(sys.executable), "Scripts", "convert-lora-to-gguf.py")
        ])
        
        for path in possible_paths:
            if Path(path).exists():
                convert_script = path
                break
        
        if not convert_script:
            logger.warning("convert-lora-to-gguf.py not found. Skipping GGUF export.")
            logger.info("Please set LLAMA_CPP_PATH environment variable to your llama.cpp directory,")
            logger.info("or ensure convert_lora_to_gguf.py is in your PATH.")
            logger.info(f"Searched in: {possible_paths[:3]}")  # Show first 3 paths tried
            return
        
        cmd = [
            sys.executable, convert_script,
            adapter_path,
            "--outfile", output_path,
            "--base-model", self.config.llama_cpp_model_path
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Successfully exported to {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to export to GGUF: {e.stderr}")
            logger.info("You can manually convert the adapter later using llama.cpp tools.")


class ToolCallEvaluator:
    """Evaluate tool calling performance"""
    
    def __init__(self, llm_client: LlamaCppClient, tools: List[ToolDefinition]):
        self.llm_client = llm_client
        self.tools = tools
        self.metrics = {}
    
    async def evaluate(self, test_examples: List[ToolCallExample], 
                      use_adapter: bool = False, adapter_path: str = None) -> Dict[str, float]:
        """Evaluate model performance on test examples"""
        logger.info(f"Evaluating {len(test_examples)} examples...")
        
        correct_tool_calls = 0
        total_tool_calls = 0
        response_quality_scores = []
        
        for example in tqdm(test_examples, desc="Evaluating"):
            # Get model's tool calls
            messages = [{"role": "user", "content": example.user_prompt}]
            response = await self.llm_client.generate_with_tools(messages, self.tools)
            
            predicted_calls = response.get("tool_calls", [])
            expected_calls = example.tool_calls
            
            # Evaluate tool call accuracy
            tool_accuracy = self._compute_tool_accuracy(predicted_calls, expected_calls)
            correct_tool_calls += tool_accuracy["correct"]
            total_tool_calls += tool_accuracy["total"]
            
            # Evaluate response quality
            quality_score = await self._evaluate_response_quality(
                response.get("content", ""),
                example.assistant_response
            )
            response_quality_scores.append(quality_score)
        
        # Compute metrics
        metrics = {
            "tool_call_accuracy": correct_tool_calls / total_tool_calls if total_tool_calls > 0 else 0.0,
            "avg_response_quality": np.mean(response_quality_scores) if response_quality_scores else 0.0,
            "total_examples": len(test_examples),
            "adapter_used": use_adapter
        }
        
        return metrics
    
    def _compute_tool_accuracy(self, predicted: List[Dict], expected: List[Dict]) -> Dict[str, int]:
        """Compute accuracy of tool calls"""
        correct = 0
        total = max(len(predicted), len(expected))
        
        for i in range(min(len(predicted), len(expected))):
            pred_call = predicted[i]
            exp_call = expected[i]
            
            # Check if tool name matches
            if pred_call.get("name") == exp_call.get("name"):
                # Check if arguments are similar (simplified comparison)
                pred_args = json.dumps(pred_call.get("arguments", {}), sort_keys=True)
                exp_args = json.dumps(exp_call.get("arguments", {}), sort_keys=True)
                
                if pred_args == exp_args:
                    correct += 1
        
        return {"correct": correct, "total": total}
    
    async def _evaluate_response_quality(self, predicted: str, expected: str) -> float:
        """Evaluate quality of response using LLM as judge"""
        evaluation_prompt = f"""Compare these two responses and rate the quality of Response A compared to Response B on a scale of 0-1.

Response A (Predicted):
{predicted}

Response B (Expected):
{expected}

Consider factual accuracy, completeness, and clarity. Output only a number between 0 and 1.

Score:"""
        
        try:
            score_str = await self.llm_client.generate(
                evaluation_prompt,
                max_tokens=10,
                temperature=0.1
            )
            
            # Extract number from response
            match = re.search(r'0?\.\d+|1\.0|1|0', score_str)
            if match:
                return float(match.group())
        except:
            pass
        
        return 0.5  # Default score if evaluation fails


class AutoLoRACreator:
    """Main orchestrator for automatic LoRA creation"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.llm_client = LlamaCppClient(config.llama_cpp_url, config.llama_cpp_model_path)
        self.tools = []
        self.trainer = None
        self.evaluator = None
        
    def add_tool(self, tool: ToolDefinition):
        """Add a tool definition"""
        self.tools.append(tool)
        logger.info(f"Added tool: {tool.name}")
    
    async def create_adapter(self, num_iterations: int = 3, skip_checks: bool = False):
        """Main pipeline to create and optimize LoRA adapter"""
        logger.info("Starting automatic LoRA adapter creation...")
        
        # Run dependency checks unless explicitly skipped
        if not skip_checks:
            checks_passed, check_results = DependencyChecker.run_all_checks(self.config)
            if not checks_passed:
                logger.error("Dependency checks failed. Please fix the issues and try again.")
                logger.info("To skip checks (not recommended), use skip_checks=True")
                return None, {"error": "Dependency checks failed"}
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Start llama.cpp server
        await self.llm_client.start_server()
        
        best_metrics = {"tool_call_accuracy": 0}
        best_adapter_path = None
        
        for iteration in range(num_iterations):
            logger.info(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
            
            # Generate synthetic data
            generator = SyntheticDataGenerator(self.llm_client, self.tools)
            examples = await generator.generate_examples(self.config.num_synthetic_examples)
            
            # Split into train/test
            split_idx = int(len(examples) * (1 - self.config.test_split_ratio))
            train_examples = examples[:split_idx]
            test_examples = examples[split_idx:]
            
            # Create datasets
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True
            )
            
            train_dataset = ToolCallingDataset(train_examples, tokenizer, self.config.max_seq_length)
            eval_dataset = ToolCallingDataset(test_examples, tokenizer, self.config.max_seq_length)
            
            # Train adapter
            self.trainer = LoRATrainer(self.config)
            self.trainer.setup_model()
            
            training_history = self.trainer.train(train_dataset, eval_dataset)
            
            # Save adapter for this iteration
            adapter_path = f"{self.config.output_dir}/iteration_{iteration + 1}"
            Path(adapter_path).mkdir(parents=True, exist_ok=True)
            self.trainer.save_adapter(adapter_path)
            
            # Clean up model memory before evaluation
            self.trainer.cleanup_model()
            
            # Export to GGUF
            gguf_path = f"{adapter_path}/adapter.gguf"
            self.trainer.export_to_gguf(adapter_path, gguf_path)
            
            # Evaluate
            self.evaluator = ToolCallEvaluator(self.llm_client, self.tools)
            metrics = await self.evaluator.evaluate(test_examples, use_adapter=True, adapter_path=gguf_path)
            
            logger.info(f"Iteration {iteration + 1} metrics: {metrics}")
            
            # Track best adapter
            if metrics["tool_call_accuracy"] > best_metrics["tool_call_accuracy"]:
                best_metrics = metrics
                best_adapter_path = adapter_path
                logger.info(f"New best adapter found with accuracy: {metrics['tool_call_accuracy']:.3f}")
            
            # Save metrics
            with open(f"{adapter_path}/metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Analyze errors and adjust for next iteration
            if iteration < num_iterations - 1:
                await self._analyze_and_adjust(test_examples, metrics)
        
        # Final report
        logger.info(f"\n=== Training Complete ===")
        logger.info(f"Best adapter: {best_adapter_path}")
        logger.info(f"Best metrics: {best_metrics}")
        
        # Clean up
        await self.llm_client.close()
        
        return best_adapter_path, best_metrics
    
    async def _analyze_and_adjust(self, test_examples: List[ToolCallExample], metrics: Dict[str, float]):
        """Analyze errors and adjust training strategy"""
        logger.info("Analyzing errors and adjusting strategy...")
        
        # Identify common failure patterns
        error_patterns = []
        
        for example in test_examples[:10]:  # Sample analysis
            messages = [{"role": "user", "content": example.user_prompt}]
            response = await self.llm_client.generate_with_tools(messages, self.tools)
            
            predicted_calls = response.get("tool_calls", [])
            expected_calls = example.tool_calls
            
            if predicted_calls != expected_calls:
                error_patterns.append({
                    "prompt": example.user_prompt,
                    "expected": expected_calls,
                    "predicted": predicted_calls
                })
        
        # Adjust based on patterns
        if metrics["tool_call_accuracy"] < 0.5:
            # Need more diverse examples
            self.config.num_synthetic_examples = int(self.config.num_synthetic_examples * 1.5)
            logger.info(f"Increasing synthetic examples to {self.config.num_synthetic_examples}")
        
        if metrics["avg_response_quality"] < 0.6:
            # Adjust generation parameters
            self.config.temperature = max(0.3, self.config.temperature - 0.1)
            logger.info(f"Reducing temperature to {self.config.temperature}")
        
        # Save error analysis
        with open(f"{self.config.output_dir}/error_analysis.json", "w") as f:
            json.dump(error_patterns, f, indent=2)


def run_checks_only():
    """Run dependency checks without training"""
    print("\n" + "="*60)
    print(" Running LoRA Training Dependency Checks")
    print("="*60 + "\n")
    
    # Create a sample config for checking
    config = TrainingConfig(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        llama_cpp_url="http://localhost:8000",
        llama_cpp_model_path=r"C:\models\Qwen3-4B-Instruct-2507\Qwen3-4B-Instruct-2507-F16.gguf",
        output_dir="./lora_adapters"
    )
    
    checks_passed, results = DependencyChecker.run_all_checks(config)
    
    if checks_passed:
        print("\n System is ready for LoRA training!")
        print("\nTo start training, run:")
        print("  python auto_lora.py --train")
    else:
        print("\n Please fix the issues above before training")
        print("\nFor help:")
        print("  - Install missing Python packages: pip install -r requirements.txt")
        print("  - Set LLAMA_CPP_PATH: set LLAMA_CPP_PATH=C:\\path\\to\\llama.cpp")
        print("  - Download model: huggingface-cli download Qwen/Qwen3-4B-Instruct-2507")
    
    return checks_passed


async def main():
    """Main entry point"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Automatic LoRA Adapter Creator")
    parser.add_argument("--check", action="store_true", help="Run dependency checks only")
    parser.add_argument("--train", action="store_true", help="Start training")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency checks (not recommended)")
    parser.add_argument("--model-path", type=str, help="Path to GGUF model file")
    parser.add_argument("--examples", type=int, default=100, help="Number of synthetic examples to generate")
    parser.add_argument("--iterations", type=int, default=3, help="Number of training iterations")
    args = parser.parse_args()
    
    # If only checking dependencies
    if args.check or (not args.train and not args.check):
        checks_passed = run_checks_only()
        return
    
    # Configuration
    config = TrainingConfig(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        llama_cpp_url="http://localhost:8000",
        llama_cpp_model_path=args.model_path or r"C:\models\Qwen3-4B-Instruct-2507\Qwen3-4B-Instruct-2507-F16.gguf",
        output_dir="./lora_adapters",
        num_synthetic_examples=args.examples,
        num_epochs=3,
        batch_size=4,
        learning_rate=3e-4
    )
    
    # Create auto-creator
    creator = AutoLoRACreator(config)
    
    # Define tools for the model to learn
    tools = [
        ToolDefinition(
            name="web_search",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            },
            returns={"type": "array", "items": {"type": "object"}}
        ),
        ToolDefinition(
            name="calculator",
            description="Perform mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression"}
                },
                "required": ["expression"]
            },
            returns={"type": "number"}
        ),
        ToolDefinition(
            name="get_weather",
            description="Get current weather information",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
                },
                "required": ["location"]
            },
            returns={"type": "object"}
        ),
        ToolDefinition(
            name="create_reminder",
            description="Create a reminder for a specific time",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Reminder message"},
                    "time": {"type": "string", "description": "Time in ISO format"}
                },
                "required": ["message", "time"]
            },
            returns={"type": "object", "properties": {"success": {"type": "boolean"}}}
        )
    ]
    
    # Add tools
    for tool in tools:
        creator.add_tool(tool)
    
    # Create and optimize adapter
    best_adapter, metrics = await creator.create_adapter(
        num_iterations=args.iterations,
        skip_checks=args.skip_checks
    )
    
    # Check if training was successful
    if best_adapter is None:
        print("\n Training failed. Please check the logs above for errors.")
        return
    
    print(f"\n LoRA adapter creation complete!")
    print(f" Best adapter saved to: {best_adapter}")
    print(f" Performance metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    # Instructions for using the adapter with Qwen3
    print("\n To use the adapter with llama.cpp:")
    print(f"   llama-cli -m {config.llama_cpp_model_path} \\")
    print(f"            --lora {best_adapter}/adapter.gguf \\")
    print(f"            --chat-template qwen3 \\")
    print(f"            -p \"Your prompt here\"")
    
    print("\n Prerequisites for Qwen3-4B-Instruct-2507:")
    print("   # Download and convert the model:")
    print("   huggingface-cli download Qwen/Qwen3-4B-Instruct-2507")
    print("   python convert-hf-to-gguf.py Qwen/Qwen3-4B-Instruct-2507 \\")
    print("       --outfile ./models/Qwen3-4B-Instruct-2507-F16.gguf \\")
    print("       --outtype F16")

if __name__ == "__main__":
    asyncio.run(main())
