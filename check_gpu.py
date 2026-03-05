import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if hasattr(torch.version, 'cuda') else "N/A")

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())
else:
    print("\n⚠️ GPU NOT DETECTED")
    print("\nTo enable GPU support:")
    print("1. Check your NVIDIA GPU driver is installed")
    print("2. Install PyTorch with CUDA support:")
    print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("   (Replace cu118 with your CUDA version: cu117, cu118, cu121, etc.)")
