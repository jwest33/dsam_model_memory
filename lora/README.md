# LoRA Adapter Auto-Creator Module

## Overview

The `auto_lora.py` module provides an automated pipeline for creating, training, and optimizing LoRA (Low-Rank Adaptation) adapters for Large Language Models, specifically designed for tool-calling capabilities with Qwen3-4B-Instruct models and llama.cpp.

## What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that:
- Freezes the original model weights
- Adds small trainable rank-decomposition matrices to model layers
- Reduces training memory requirements by 90%+ compared to full fine-tuning
- Produces small adapter files (typically <100MB) that can be swapped at runtime

## Quick Start

```bash
# Check dependencies
python auto_lora.py --check

# Start training with default settings
python auto_lora.py --train

# Custom training configuration
python auto_lora.py --train --examples 500 --iterations 5 --model-path /path/to/model.gguf
```

## Environment Setup

### Required Environment Variables

```bash
# Set path to your llama.cpp installation
export LLAMA_CPP_PATH=/path/to/llama.cpp

# Windows
set LLAMA_CPP_PATH=C:\Users\YourName\llama.cpp
```

### Dependencies

```bash
pip install torch transformers peft datasets numpy aiohttp tqdm psutil pyyaml

# Optional for optimization
pip install bitsandbytes accelerate
```

## How It Works

### 1. Dependency Checking Phase

The `DependencyChecker` class validates your environment before training:

```python
- Python version (3.8+ required)
- Required Python packages
- CUDA availability and GPU memory
- System RAM and disk space
- llama.cpp server and tools
- Model file availability
```

### 2. Synthetic Data Generation

The `SyntheticDataGenerator` creates training examples:

```python
1. Randomly selects tool combinations
2. Generates realistic user prompts requiring those tools
3. Creates appropriate tool calls with arguments
4. Simulates tool response data
5. Generates final assistant responses
```

Example flow:
```
User: "What's the weather in Paris and calculate 15% tip on $85"
→ Tool calls: [get_weather("Paris"), calculator("85 * 0.15")]
→ Tool responses: [{temp: 18°C}, {result: 12.75}]
→ Assistant: "The weather in Paris is 18°C. A 15% tip on $85 would be $12.75"
```

### 3. LoRA Configuration

The trainer applies LoRA to specific model layers:

```python
target_modules = [
    "q_proj",     # Query projection
    "k_proj",     # Key projection  
    "v_proj",     # Value projection
    "o_proj",     # Output projection
    "gate_proj",  # MLP gate
    "up_proj",    # MLP up projection
    "down_proj"   # MLP down projection
]
```

Key parameters:
- `lora_r`: Rank of adaptation (default: 16)
- `lora_alpha`: Scaling factor (default: 32)
- `lora_dropout`: Dropout for regularization (default: 0.1)

### 4. Training Pipeline

The `LoRATrainer` manages the training process:

```python
for iteration in range(num_iterations):
    1. Generate synthetic training data
    2. Create train/test splits
    3. Setup LoRA model with PEFT
    4. Train with gradient accumulation
    5. Evaluate on test set
    6. Export to GGUF format
    7. Analyze errors and adjust strategy
```

### 5. Iterative Optimization

The system automatically improves across iterations:

```python
- Iteration 1: Baseline training
- Iteration 2: Adjust based on error patterns
  - Low accuracy → Increase training examples
  - Poor quality → Reduce temperature
- Iteration 3: Fine-tune with best parameters
```

### 6. GGUF Export

Converts PyTorch LoRA weights to llama.cpp format:

```bash
python convert_lora_to_gguf.py adapter_path --outfile adapter.gguf
```

## Architecture Components

### Core Classes

#### `AutoLoRACreator`
Main orchestrator that coordinates the entire pipeline:
- Manages configuration
- Controls training iterations
- Tracks best performing adapters
- Handles cleanup and resource management

#### `LlamaCppClient`
Interfaces with llama.cpp server:
- Auto-starts server if not running
- Handles text generation requests
- Parses tool calling responses
- Manages server lifecycle

#### `ToolCallingDataset`
PyTorch dataset for training:
- Formats conversations for Qwen3 chat template
- Tokenizes with proper padding
- Creates attention masks
- Handles sequence truncation

#### `ToolCallEvaluator`
Measures adapter performance:
- Tool call accuracy (name and arguments)
- Response quality scoring
- Comparative evaluation using LLM-as-judge

## Tool Definition Format

Tools are defined with JSON schema:

```python
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
)
```

## Training Configuration

Key configuration options in `TrainingConfig`:

```python
# Model settings
model_name: HuggingFace model ID
llama_cpp_model_path: Path to GGUF file

# LoRA parameters
lora_r: Rank (4-64, higher = more capacity)
lora_alpha: Scaling (typically 2x rank)
lora_dropout: Regularization (0.05-0.2)

# Training settings
batch_size: Samples per batch (2-8)
learning_rate: Step size (1e-4 to 5e-4)
num_epochs: Training rounds (2-5)
gradient_accumulation_steps: Effective batch multiplier

# Generation settings
num_synthetic_examples: Training data size
temperature: Creativity (0.3-0.9)
top_p: Nucleus sampling (0.8-0.95)
```

## Memory Optimization

The module includes several memory-saving techniques:

1. **4-bit Quantization** (if bitsandbytes available):
   - Reduces model memory by 75%
   - Minimal accuracy loss

2. **Gradient Checkpointing**:
   - Trades computation for memory
   - Enables larger batch sizes

3. **Mixed Precision (FP16)**:
   - Halves memory usage
   - Faster training on modern GPUs

4. **Model Cleanup**:
   - Releases GPU memory between iterations
   - Prevents memory leaks

## Usage Examples

### Basic Training

```python
import asyncio
from auto_lora import AutoLoRACreator, TrainingConfig, ToolDefinition

config = TrainingConfig(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    num_synthetic_examples=200,
    num_epochs=3
)

creator = AutoLoRACreator(config)

# Define custom tools
creator.add_tool(ToolDefinition(
    name="database_query",
    description="Query SQL database",
    parameters={...},
    returns={...}
))

# Train adapter
best_adapter, metrics = await creator.create_adapter(num_iterations=3)
```

### Using Trained Adapter

```bash
# With llama.cpp CLI
llama-cli -m model.gguf \
    --lora adapter.gguf \
    --chat-template qwen3 \
    -p "Search for Python tutorials"

# With llama-server
llama-server -m model.gguf \
    --lora adapter.gguf \
    --port 8000
```

## Troubleshooting

### Common Issues

1. **"convert_lora_to_gguf.py not found"**
   - Set `LLAMA_CPP_PATH` environment variable
   - Ensure llama.cpp is properly installed

2. **"CUDA out of memory"**
   - Reduce `batch_size` in config
   - Enable 4-bit quantization
   - Use gradient accumulation

3. **"Model not found"**
   - Download with: `huggingface-cli download Qwen/Qwen3-4B-Instruct-2507`
   - Convert to GGUF format
   - Update `llama_cpp_model_path` in config

4. **Poor tool calling accuracy**
   - Increase `num_synthetic_examples`
   - Add more diverse tool examples
   - Increase training iterations

## Performance Tips

1. **GPU Optimization**:
   - Use CUDA 12.1+ for best performance
   - Enable Flash Attention if available
   - Monitor GPU memory with `nvidia-smi`

2. **Data Quality**:
   - More examples > longer training
   - Diverse tool combinations improve generalization
   - Balance simple and complex examples

3. **Hyperparameter Tuning**:
   - Start with default settings
   - Adjust based on evaluation metrics
   - Use lower learning rates for stability

## Advanced Features

### Custom Tool Examples

Provide specific examples for better training:

```python
tool.examples = [
    {
        "input": "Find recent AI papers",
        "arguments": {"query": "artificial intelligence papers 2024"},
        "output": [{"title": "GPT-4 Analysis", ...}]
    }
]
```

### Multi-Tool Chains

Train complex tool sequences:

```python
# Search → Calculate → Format
"Find Bitcoin price and calculate ROI for $1000 investment"
```

### Error Analysis

Review training errors to improve:

```python
# Check error_analysis.json after training
{
    "prompt": "User question",
    "expected": ["correct_tool_calls"],
    "predicted": ["actual_tool_calls"]
}
```

## File Structure

```
lora/
├── auto_lora.py           # Main module
├── README.md              # This documentation
└── lora_adapters/         # Output directory
    ├── iteration_1/       # First training iteration
    │   ├── adapter_model.bin
    │   ├── adapter_config.json
    │   ├── adapter.gguf   # llama.cpp format
    │   └── metrics.json   # Performance metrics
    ├── iteration_2/
    └── final_adapter/     # Best performing adapter
```

## Contributing

To extend the module:

1. Add new tool types in `ToolDefinition`
2. Implement custom evaluation metrics in `ToolCallEvaluator`
3. Extend `SyntheticDataGenerator` for domain-specific data
4. Add new target modules for different model architectures

## License

This module is part of the JAM (Journalistic Agent Memory) project and follows the same licensing terms.

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [llama.cpp LoRA Guide](https://github.com/ggerganov/llama.cpp/blob/master/examples/finetune/README.md)
- [Qwen Model Documentation](https://huggingface.co/Qwen)