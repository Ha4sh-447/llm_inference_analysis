# Google Colab Setup Guide for vLLM and BitsAndBytes

When running vLLM with BitsAndBytes on Google Colab, you may encounter missing dependency errors (such as `libnvJitLink.so.13`) due to CUDA runtime path mismatches.

For convenience, a ready-to-run Jupyter Notebook is available in the root of the repository: [`vllm_bnb_colab.ipynb`](../vllm_bnb_colab.ipynb).

To resolve these errors manually, run the following setup commands in Google Colab to clean your environment and install PyTorch and vLLM targeting CUDA 12.1 directly:

```python
# 1. Clean up existing conflicting package versions
!pip uninstall -y torch torchvision torchaudio vllm autoawq bitsandbytes

# 2. Install PyTorch targeting CUDA 12.1
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install vLLM from the official CUDA 12.1 wheels index
!pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# 4. Install matching CUDA 12.1 AutoAWQ and standard BitsAndBytes
!pip install autoawq --extra-index-url https://download.pytorch.org/whl/cu121
!pip install bitsandbytes>=0.46.1
```

Once installed, verify the setup by running the following verification block:

```python
import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version (PyTorch): {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
```
