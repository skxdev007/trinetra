"""
Script to install PyTorch with CUDA support for GPU acceleration.

This script will:
1. Uninstall CPU-only PyTorch
2. Install PyTorch with CUDA 12.1 support (compatible with CUDA 13.0)
3. Verify GPU is detected
"""

import subprocess
import sys

def run_command(cmd):
    """Run command and print output."""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    return result.returncode == 0

def main():
    print("🚀 Installing PyTorch with CUDA Support")
    print("="*60)
    
    # Step 1: Uninstall CPU-only PyTorch
    print("\n📦 Step 1: Uninstalling CPU-only PyTorch...")
    if not run_command(f"{sys.executable} -m pip uninstall -y torch torchvision torchaudio"):
        print("⚠️  Warning: Failed to uninstall old PyTorch (may not be installed)")
    
    # Step 2: Install PyTorch with CUDA 12.1 (compatible with CUDA 13.0)
    print("\n📦 Step 2: Installing PyTorch with CUDA 12.1 support...")
    cuda_url = "https://download.pytorch.org/whl/cu121"
    install_cmd = f"{sys.executable} -m pip install torch torchvision torchaudio --index-url {cuda_url}"
    
    if not run_command(install_cmd):
        print("❌ Failed to install PyTorch with CUDA support")
        return False
    
    # Step 3: Verify GPU detection
    print("\n🔍 Step 3: Verifying GPU detection...")
    verify_cmd = f"{sys.executable} check_gpu.py"
    
    if not run_command(verify_cmd):
        print("⚠️  Warning: GPU verification script failed")
        return False
    
    print("\n" + "="*60)
    print("✅ PyTorch with CUDA support installed successfully!")
    print("="*60)
    print("\n💡 Next steps:")
    print("1. Run 'python check_gpu.py' to verify GPU is detected")
    print("2. Test with: python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\"")
    print("3. Your models will now automatically use GPU when device='auto'")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
