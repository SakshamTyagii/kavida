import os
import sys
import subprocess
import platform

def check_gpu():
    try:
        gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True).decode('utf-8').strip()
        if gpu_info:
            print("GPU Model and VRAM:")
            print(gpu_info)
        else:
            print("No GPU found.")
    except Exception as e:
        print("Error checking GPU:", e)

def check_cuda():
    cuda_installed = subprocess.call("nvcc --version", shell=True) == 0
    if cuda_installed:
        print("CUDA is installed and accessible.")
    else:
        print("CUDA is not installed or not accessible.")

def check_python_version():
    if sys.version_info >= (3, 9):
        print("Python version is acceptable:", sys.version)
    else:
        print("Python version is not supported. Please use Python 3.9 or higher.")

def check_pip_virtualenv():
    pip_installed = subprocess.call("pip --version", shell=True) == 0
    virtualenv_installed = subprocess.call("virtualenv --version", shell=True) == 0
    if pip_installed:
        print("pip is installed.")
    else:
        print("pip is not installed.")
    
    if virtualenv_installed:
        print("virtualenv is installed.")
    else:
        print("virtualenv is not installed.")

def check_system_memory_and_disk():
    mem_info = subprocess.check_output("free -h", shell=True).decode('utf-8').strip()
    print("Current System Memory:")
    print(mem_info)
    
    disk_info = subprocess.check_output("df -h", shell=True).decode('utf-8').strip()
    print("Available Disk Space:")
    print(disk_info)

if __name__ == "__main__":
    check_gpu()
    check_cuda()
    check_python_version()
    check_pip_virtualenv()
    check_system_memory_and_disk()