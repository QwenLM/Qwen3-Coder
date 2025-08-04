# Qwen3-Coder Developer Guide

This guide provides comprehensive information for developers who want to contribute to or work with Qwen3-Coder.

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Development Environment Setup](#development-environment-setup)
- [Working with Qwen3-Coder Models](#working-with-qwen3-coder-models)
- [Development Workflow](#development-workflow)
- [Testing and Evaluation](#testing-and-evaluation)
- [Performance Optimization](#performance-optimization)
- [Common Issues and Solutions](#common-issues-and-solutions)
- [Advanced Topics](#advanced-topics)

## Hardware Requirements

### Minimum Requirements

#### For Inference
- **GPU Memory**: 
  - Qwen3-Coder-30B-A3B-Instruct: 24GB VRAM minimum
  - Qwen3-Coder-480B-A35B-Instruct: Multi-GPU setup with 4×80GB or 8×40GB
- **System RAM**: 64GB minimum, 128GB recommended
- **Storage**: 1TB SSD (for model weights and workspace)

#### For Development/Fine-tuning
- **GPU**: Multiple high-end GPUs (A100 80GB, H100, or similar)
- **System RAM**: 256GB minimum
- **Storage**: 2TB+ NVMe SSD

### Recommended Setup
```
# For Qwen3-Coder-30B-A3B-Instruct
- 1× NVIDIA RTX 4090 (24GB) or better
- 128GB DDR5 RAM
- 1TB NVMe SSD

# For Qwen3-Coder-480B-A35B-Instruct
- 4× NVIDIA A100 80GB or 8× A100 40GB
- 512GB RAM
- 2TB NVMe SSD
```

## Development Environment Setup

### 1. System Prerequisites

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y build-essential git curl wget python3-dev

# Install CUDA (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-12-2
```

### 2. Python Environment

```bash
# Install Miniconda (recommended)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create conda environment
conda create -n qwen3-coder python=3.10
conda activate qwen3-coder

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 3. Repository Setup

```bash
# Clone the repository
git clone https://github.com/QwenLM/Qwen3-Coder.git
cd Qwen3-Coder

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install black ruff pytest pytest-cov pre-commit

# Set up pre-commit hooks
pre-commit install
```

### 4. Model Access

```python
# For Hugging Face
pip install huggingface-hub
huggingface-cli login  # Enter your token

# For ModelScope
pip install modelscope
```

## Working with Qwen3-Coder Models

### Loading Models Efficiently

#### Single GPU (smaller models)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

#### Multi-GPU Setup (large models)
```python
# For Qwen3-Coder-480B-A35B-Instruct
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen3-Coder-480B-A35B-Instruct"

# Define custom device map for optimal distribution
device_map = {
    "transformer.wte": 0,
    "transformer.wpe": 0,
    "transformer.ln_f": 0,
    "lm_head": 0,
}

# Distribute layers across GPUs
num_gpus = torch.cuda.device_count()
num_layers = 60  # Adjust based on model
layers_per_gpu = num_layers // num_gpus

for i in range(num_layers):
    device_map[f"transformer.h.{i}"] = i // layers_per_gpu

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map=device_map,
    trust_remote_code=True
)
```

#### Quantized Models (FP8)
```python
# For memory-efficient inference
model_name = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float8_e4m3fn,  # FP8 format
    device_map="auto",
    trust_remote_code=True
)
```

### Memory Management

```python
import torch
import gc

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    
def get_memory_usage():
    """Get current GPU memory usage"""
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
```

## Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes
# ... edit files ...

# Run formatter
black . --line-length 120
ruff check --fix

# Run tests
pytest tests/

# Commit changes
git add .
git commit -m "feat: add your feature description"
```

### 2. Adding New Evaluation Benchmarks

```python
# Create new benchmark in qwencoder-eval/
class YourBenchmark:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def evaluate(self, test_data):
        results = []
        for item in test_data:
            # Implement evaluation logic
            pass
        return results
```

### 3. Implementing Model Improvements

```python
# Example: Adding a new decoding strategy
def custom_decode(model, input_ids, **kwargs):
    """Custom decoding strategy for Qwen3-Coder"""
    # Implementation
    pass
```

## Testing and Evaluation

### Unit Testing

```python
# tests/test_your_feature.py
import pytest
from your_module import your_function

def test_your_function():
    """Test your new function"""
    result = your_function(input_data)
    assert result == expected_output

@pytest.mark.parametrize("input,expected", [
    ("test1", "output1"),
    ("test2", "output2"),
])
def test_multiple_cases(input, expected):
    assert your_function(input) == expected
```

### Integration Testing

```bash
# Run specific evaluation
cd qwencoder-eval/instruct
python eval_plus/generate.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct

# Run comprehensive evaluation
bash evaluate.sh
```

### Performance Testing

```python
import time
import torch

def benchmark_inference(model, tokenizer, prompts, num_runs=10):
    """Benchmark model inference speed"""
    times = []
    
    for _ in range(num_runs):
        start = time.time()
        
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100)
        
        times.append(time.time() - start)
    
    print(f"Average time: {sum(times)/len(times):.2f}s")
    print(f"Throughput: {len(prompts)*num_runs/sum(times):.2f} prompts/s")
```

## Performance Optimization

### 1. Inference Optimization

```python
# Use torch.compile for faster inference
import torch

model = torch.compile(model, mode="reduce-overhead")

# Enable flash attention
model.config.use_flash_attention = True

# Use better batch processing
def batch_inference(model, tokenizer, prompts, batch_size=8):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs)
        results.extend(tokenizer.batch_decode(outputs))
    return results
```

### 2. Memory Optimization

```python
# Gradient checkpointing for training
model.gradient_checkpointing_enable()

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for batch in dataloader:
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Common Issues and Solutions

### Issue 1: Out of Memory (OOM)

**Symptoms**: `CUDA out of memory` error

**Solutions**:
```python
# 1. Reduce batch size
batch_size = 1

# 2. Use gradient accumulation
accumulation_steps = 8

# 3. Enable CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    offload_folder="offload",
    offload_state_dict=True
)

# 4. Use quantization
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)
```

### Issue 2: Slow Inference

**Solutions**:
```python
# 1. Use torch.compile
model = torch.compile(model)

# 2. Enable KV cache
model.config.use_cache = True

# 3. Use vLLM for serving
# pip install vllm
from vllm import LLM, SamplingParams

llm = LLM(model=model_name, tensor_parallel_size=4)
```

### Issue 3: Model Loading Errors

**Solutions**:
```bash
# 1. Clear cache
rm -rf ~/.cache/huggingface

# 2. Use specific revision
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    revision="main",  # or specific commit
    force_download=True
)

# 3. Check disk space
df -h
```

## Advanced Topics

### 1. Fine-tuning Qwen3-Coder

```python
# Using LoRA for efficient fine-tuning
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

model = get_peft_model(model, lora_config)
```

### 2. Custom Tool Integration

```python
# Implementing function calling
from qwen3coder_tool_parser import parse_tools, execute_tool

tools = [
    {
        "name": "execute_code",
        "description": "Execute Python code",
        "parameters": {
            "code": {"type": "string", "description": "Python code to execute"}
        }
    }
]

# Parse model output for tool calls
tool_calls = parse_tools(model_output)
results = [execute_tool(call) for call in tool_calls]
```

### 3. Multi-Modal Extensions

```python
# Integrating with vision models
class MultiModalQwen3Coder:
    def __init__(self, text_model, vision_model):
        self.text_model = text_model
        self.vision_model = vision_model
    
    def process(self, text, image=None):
        if image:
            vision_features = self.vision_model(image)
            # Combine features
        return self.text_model.generate(text)
```

## Resources

- **Documentation**: https://qwen.readthedocs.io/
- **Model Hub**: https://huggingface.co/Qwen
- **Discord Community**: https://discord.gg/CV4E9rpNSD
- **Paper**: https://arxiv.org/abs/2505.09388

## Getting Help

If you encounter issues:

1. Check this guide and the main README
2. Search existing GitHub issues
3. Ask in the Discord community
4. Create a new issue with detailed information

Remember to include:
- Your environment details
- Minimal reproducible example
- Error messages and logs
- What you've already tried

Happy coding with Qwen3-Coder! 🚀