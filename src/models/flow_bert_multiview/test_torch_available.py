import torch


if __name__ == '__main__':  # ✅ 正确写法
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version:    {torch.version.cuda}")
        print(f"Device Name:     {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ 警告: CUDA 不可用，PyTorch 正在使用 CPU！")