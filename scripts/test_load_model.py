import os
from pathlib import Path

# Set cache directories
HF_ROOT = "/ocean/projects/cis250206p/aanugu"
os.environ["HF_HOME"] = f"{HF_ROOT}/hf-cache"
os.environ["HF_HUB_CACHE"] = f"{HF_ROOT}/hf-cache/hub"
os.environ["TRANSFORMERS_CACHE"] = f"{HF_ROOT}/tf-cache"
os.environ["DIFFUSERS_CACHE"] = f"{HF_ROOT}/diff-cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

import diffusers
print(f"Diffusers: {diffusers.__version__}")

from diffusers import QwenImageTransformer2DModel, QwenImageEditPlusPipeline

dtype = torch.bfloat16

print("\n--- Loading Transformer ---")
transformer = QwenImageTransformer2DModel.from_pretrained(
    "linoyts/Qwen-Image-Edit-Rapid-AIO",
    subfolder="transformer",
    torch_dtype=dtype,
)
print("Transformer loaded!")

print("\n--- Loading Pipeline ---")
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    transformer=transformer,
    torch_dtype=dtype,
)
print("Pipeline loaded!")

print("\n--- Moving to GPU ---")
pipe = pipe.to("cuda")
print("Done!")
