import importlib
import sys

import torch

# import numpy as np
# import PIL
# import torchvision

for name in ["numpy", "torch", "torchvision", "PIL", "matplotlib", "tqdm"]:
    try:
        module = importlib.import_module(name)
        print(f"{name}: {module.__version__}")
    except ImportError:
        print(f"{name}: 未安装")

# print("NumPy version:", np.__version__)
# print("PyTorch version:", torch.__version__)
# print("Torchvision version:", torchvision.__version__)
# print("Pillow version:", PIL.__version__)
print("Torch CUDA available:", torch.cuda.is_available())
print("python version:", sys.version)
