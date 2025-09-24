## torch
import torch

# 检查 CUDA 是否可用
print(f"CUDA available: {torch.cuda.is_available()}")

# 检查可用的 GPU 数量
print(f"Number of GPUs: {torch.cuda.device_count()}")

# 获取当前 GPU 设备索引
if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    # 创建一个在 GPU 上的张量来测试
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.mm(x, y)
    print("GPU computation test passed!")
