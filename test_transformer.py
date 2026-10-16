import torch
import transformers

print("PyTorch version:", torch.__version__)
print("Transformers version:", transformers.__version__)

print("PyTorch working:", torch.tensor([1, 2, 3]))