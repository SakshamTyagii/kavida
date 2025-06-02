#!/usr/bin/env python3
"""
Test script to verify the LLM quantization environment is properly set up.
"""

import torch
import transformers
import accelerate
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
import sys

def test_environment():
    """Test the LLM quantization environment setup."""
    
    print("=" * 60)
    print("LLM QUANTIZATION ENVIRONMENT TEST")
    print("=" * 60)
    
    # Test Python version
    print(f"Python version: {sys.version}")
    print()
    
    # Test PyTorch and CUDA
    print("PYTORCH & CUDA:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        
        # Test GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU memory: {gpu_memory:.1f} GB")
    else:
        print("  WARNING: CUDA not available!")
    print()
    
    # Test quantization libraries
    print("QUANTIZATION LIBRARIES:")
    print(f"  Transformers version: {transformers.__version__}")
    print(f"  Accelerate version: {accelerate.__version__}")
    print(f"  BitsAndBytes version: {bnb.__version__}")
    print()
    
    # Test BitsAndBytes configuration
    print("BITSANDBYTES CONFIG TEST:")
    try:
        # Test 8-bit config
        bnb_config_8bit = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )
        print("  ✓ 8-bit quantization config created successfully")
        
        # Test 4-bit config
        bnb_config_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        print("  ✓ 4-bit quantization config created successfully")
        
    except Exception as e:
        print(f"  ✗ BitsAndBytes config error: {e}")
    print()
    
    # Test a simple tensor operation on GPU
    print("GPU TENSOR TEST:")
    try:
        if torch.cuda.is_available():
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = torch.matmul(x, y)
            print("  ✓ GPU tensor operations working")
        else:
            print("  ⚠ Skipping GPU tensor test (CUDA not available)")
    except Exception as e:
        print(f"  ✗ GPU tensor operation error: {e}")
    print()
    
    print("=" * 60)
    print("ENVIRONMENT TEST COMPLETE")
    print("=" * 60)
    
    # Summary
    cuda_ok = torch.cuda.is_available()
    libs_ok = True  # We got this far, so imports worked
    
    if cuda_ok and libs_ok:
        print("🎉 SUCCESS: Environment is ready for LLM quantization!")
        print("\nNext steps:")
        print("1. Run quantization examples with small models")
        print("2. Try loading larger models with quantization")
        print("3. Experiment with 4-bit and 8-bit quantization")
    else:
        print("⚠️  WARNING: Some components may not be working properly")
        if not cuda_ok:
            print("- CUDA is not available")
        if not libs_ok:
            print("- Some libraries failed to import")

if __name__ == "__main__":
    test_environment()
