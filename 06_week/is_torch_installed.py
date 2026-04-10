import torch
print(torch.__version__)
# 2.6.0+cu121
print(torch.cuda.is_available())
# True ← GPU 정상!
print(torch.cuda.get_device_name(0))
# NVIDIA GeForce RTX 3060 등