import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Completely hide GPU

from pathlib import Path
HF_ROOT = "/ocean/projects/cis250206p/aanugu"
os.environ["HF_HOME"] = f"{HF_ROOT}/hf-cache"
os.environ["HF_HUB_CACHE"] = f"{HF_ROOT}/hf-cache/hub"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
print(f"CUDA available: {torch.cuda.is_available()}")  # Should be False

from diffusers import QwenImageTransformer2DModel

print("\n--- Loading Transformer (CPU only) ---")
transformer = QwenImageTransformer2DModel.from_pretrained(
    "linoyts/Qwen-Image-Edit-Rapid-AIO",
    subfolder="transformer",
    torch_dtype=torch.float32,
)
print("SUCCESS: Transformer loaded on CPU!")
print(f"Model device: {next(transformer.parameters()).device}")
