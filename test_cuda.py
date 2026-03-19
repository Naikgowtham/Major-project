#!/usr/bin/env python3
"""Test CUDA availability and fix environment issues"""
import os
import sys

# Remove problematic environment variables before importing torch
env_vars_to_remove = ['CUDA_VISIBLE_DEVICES', 'CUDA_ROOT', 'CUDA_HOME', 'XLA_FLAGS']
for var in env_vars_to_remove:
    if var in os.environ:
        print(f"Removing {var}={os.environ[var]}")
        del os.environ[var]

# Set correct library path
os.environ['LD_LIBRARY_PATH'] = '/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu'

print("=" * 60)
print("CUDA Environment Test")
print("=" * 60)

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA compiled version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")

print("\n" + "=" * 60)
print("Testing CUDA Availability...")
print("=" * 60)

try:
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"✓ Device count: {torch.cuda.device_count()}")
        print(f"✓ Device name: {torch.cuda.get_device_name(0)}")
        print(f"✓ Device capability: {torch.cuda.get_device_capability(0)}")
        
        # Test tensor creation
        print("\nTesting GPU tensor operations...")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"✓ Matrix multiplication successful on {z.device}")
        print(f"✓ GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print("\n" + "="* 60)
        print("SUCCESS: GPU is working!")
        print("=" * 60)
    else:
        print("\n✗ CUDA not available - GPU cannot be used")
        print("=" * 60)
        sys.exit(1)
        
except Exception as e:
    print(f"\n✗ Error during CUDA test: {e}")
    import traceback
    traceback.print_exc()
    print("=" * 60)
    sys.exit(1)
