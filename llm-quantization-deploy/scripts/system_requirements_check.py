#!/usr/bin/env python3
"""
System Requirements Check for LLM Quantization and Deployment
Checks GPU, CUDA, Python version, pip, virtualenv, system memory, and disk space.
Compatible with Windows, macOS, and Linux.
"""

import os
import platform
import subprocess
import sys
import shutil
import psutil
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'-'*40}")
    print(f"  {title}")
    print(f"{'-'*40}")

def run_command(command, shell=True, capture_output=True):
    """Run a command and return the result."""
    try:
        if capture_output:
            result = subprocess.run(command, shell=shell, capture_output=True, text=True)
            return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
        else:
            result = subprocess.run(command, shell=shell)
            return result.returncode == 0, "", ""
    except Exception as e:
        return False, "", str(e)

def check_gpu():
    """Check GPU information and VRAM."""
    print_section("GPU Information")
    
    # Try NVIDIA GPU first
    success, output, error = run_command("nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader,nounits")
    
    if success and output:
        print("✅ NVIDIA GPU(s) detected:")
        lines = output.strip().split('\n')
        for i, line in enumerate(lines):
            parts = line.split(', ')
            if len(parts) >= 4:
                name, total_mem, free_mem, used_mem = parts[:4]
                print(f"  GPU {i}: {name}")
                print(f"    Total VRAM: {total_mem} MB ({float(total_mem)/1024:.1f} GB)")
                print(f"    Free VRAM:  {free_mem} MB ({float(free_mem)/1024:.1f} GB)")
                print(f"    Used VRAM:  {used_mem} MB ({float(used_mem)/1024:.1f} GB)")
        return True
    else:
        print("❌ NVIDIA GPU not detected or nvidia-smi not available")
        
        # Try to check for other GPUs (AMD, Intel)
        if platform.system() == "Windows":
            success, output, error = run_command('wmic path win32_VideoController get name')
            if success and output:
                print("Other GPU(s) detected:")
                lines = output.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    line = line.strip()
                    if line:
                        print(f"  - {line}")
        
        print("⚠️  For LLM quantization, an NVIDIA GPU with CUDA support is highly recommended")
        return False

def check_cuda():
    """Check CUDA installation and version."""
    print_section("CUDA Installation")
    
    # Check nvcc
    success, output, error = run_command("nvcc --version")
    if success:
        print("✅ CUDA Compiler (nvcc) is installed:")
        # Extract version from output
        for line in output.split('\n'):
            if 'release' in line.lower():
                print(f"  {line.strip()}")
        cuda_available = True
    else:
        print("❌ CUDA Compiler (nvcc) not found")
        cuda_available = False
    
    # Check if PyTorch can detect CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ PyTorch can access CUDA:")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("❌ PyTorch cannot access CUDA")
    except ImportError:
        print("ℹ️  PyTorch not installed (will be installed later)")
    
    return cuda_available

def check_python_version():
    """Check Python version."""
    print_section("Python Version")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version >= (3, 9):
        print(f"✅ Python version is sufficient: {version_str}")
        print(f"  Full version: {sys.version}")
        return True
    else:
        print(f"❌ Python version is insufficient: {version_str}")
        print("  Required: Python 3.9 or higher")
        print("  Please upgrade your Python installation")
        return False

def check_pip_and_tools():
    """Check pip, virtualenv, and other tools."""
    print_section("Python Package Management Tools")
    
    tools_status = {}
    
    # Check pip
    success, output, error = run_command("pip --version")
    if success:
        print(f"✅ pip is installed: {output}")
        tools_status['pip'] = True
    else:
        print("❌ pip is not installed")
        tools_status['pip'] = False
    
    # Check virtualenv
    success, output, error = run_command("virtualenv --version")
    if success:
        print(f"✅ virtualenv is installed: {output}")
        tools_status['virtualenv'] = True
    else:
        print("❌ virtualenv is not installed")
        tools_status['virtualenv'] = False
    
    # Check if venv module is available (built-in)
    try:
        import venv
        print("✅ venv module (built-in) is available")
        tools_status['venv'] = True
    except ImportError:
        print("❌ venv module is not available")
        tools_status['venv'] = False
    
    # Check git
    success, output, error = run_command("git --version")
    if success:
        print(f"✅ git is installed: {output}")
        tools_status['git'] = True
    else:
        print("❌ git is not installed (recommended for downloading models)")
        tools_status['git'] = False
    
    return tools_status

def check_system_resources():
    """Check system memory and disk space."""
    print_section("System Resources")
    
    # Memory information
    memory = psutil.virtual_memory()
    print("Memory Information:")
    print(f"  Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"  Available: {memory.available / (1024**3):.1f} GB")
    print(f"  Used: {memory.used / (1024**3):.1f} GB ({memory.percent:.1f}%)")
    
    # Disk space for current drive
    current_path = Path.cwd()
    disk_usage = shutil.disk_usage(current_path)
    
    print(f"\nDisk Space (Current Drive: {current_path.anchor}):")
    print(f"  Total: {disk_usage.total / (1024**3):.1f} GB")
    print(f"  Free: {disk_usage.free / (1024**3):.1f} GB")
    print(f"  Used: {(disk_usage.total - disk_usage.free) / (1024**3):.1f} GB")
    
    # Requirements assessment
    print(f"\nResource Assessment:")
    if memory.total >= 16 * (1024**3):  # 16 GB
        print("✅ RAM is sufficient for LLM quantization (16+ GB recommended)")
    elif memory.total >= 8 * (1024**3):  # 8 GB
        print("⚠️  RAM is marginal for LLM quantization (8-16 GB)")
    else:
        print("❌ RAM may be insufficient for LLM quantization (<8 GB)")
    
    if disk_usage.free >= 50 * (1024**3):  # 50 GB
        print("✅ Disk space is sufficient (50+ GB free recommended)")
    elif disk_usage.free >= 20 * (1024**3):  # 20 GB
        print("⚠️  Disk space is marginal (20-50 GB free)")
    else:
        print("❌ Disk space may be insufficient (<20 GB free)")

def check_system_info():
    """Display general system information."""
    print_section("System Information")
    
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")

def generate_recommendations():
    """Generate recommendations based on system check results."""
    print_section("Recommendations")
    
    print("Based on the system check results:")
    print("\n1. For optimal LLM quantization and deployment:")
    print("   - NVIDIA GPU with 8+ GB VRAM (RTX 3070/4060 or better)")
    print("   - 16+ GB system RAM")
    print("   - 50+ GB free disk space")
    print("   - CUDA 11.8+ or 12.x")
    print("   - Python 3.9+ with pip and virtualenv")
    
    print("\n2. Next steps:")
    print("   - Install missing dependencies if any")
    print("   - Create a virtual environment for the project")
    print("   - Install PyTorch with CUDA support")
    print("   - Install quantization libraries (bitsandbytes, auto-gptq)")
    print("   - Download and test with a small model first")

def main():
    """Main function to run all checks."""
    print_header("LLM Quantization System Requirements Check")
    print(f"Running on: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    try:
        # Run all checks
        check_system_info()
        gpu_ok = check_gpu()
        cuda_ok = check_cuda()
        python_ok = check_python_version()
        tools_status = check_pip_and_tools()
        check_system_resources()
        
        # Summary
        print_section("Summary")
        
        requirements_met = []
        requirements_failed = []
        
        if gpu_ok:
            requirements_met.append("✅ NVIDIA GPU detected")
        else:
            requirements_failed.append("❌ NVIDIA GPU not detected")
            
        if cuda_ok:
            requirements_met.append("✅ CUDA available")
        else:
            requirements_failed.append("❌ CUDA not available")
            
        if python_ok:
            requirements_met.append("✅ Python 3.9+")
        else:
            requirements_failed.append("❌ Python version insufficient")
            
        if tools_status.get('pip', False):
            requirements_met.append("✅ pip available")
        else:
            requirements_failed.append("❌ pip not available")
        
        print("Requirements Status:")
        for req in requirements_met:
            print(f"  {req}")
        for req in requirements_failed:
            print(f"  {req}")
            
        if len(requirements_failed) == 0:
            print(f"\n🎉 All major requirements are met! You're ready to proceed.")
        else:
            print(f"\n⚠️  {len(requirements_failed)} requirement(s) need attention before proceeding.")
        
        generate_recommendations()
        
    except Exception as e:
        print(f"\n❌ Error during system check: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)