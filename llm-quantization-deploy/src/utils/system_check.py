def check_gpu():
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # Convert bytes to MB
        print(f"GPU Model: {gpu_name}")
        print(f"Available VRAM: {vram:.2f} MB")
    else:
        print("No GPU found.")

def check_cuda():
    import subprocess
    try:
        result = subprocess.run(['nvcc', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("CUDA is installed and accessible.")
        else:
            print("CUDA is not installed or not accessible.")
    except FileNotFoundError:
        print("CUDA is not installed or not accessible.")

def check_python_version():
    import sys
    if sys.version_info >= (3, 9):
        print("Python version is 3.9 or higher.")
    else:
        print("Python version is lower than 3.9.")

def check_pip_virtualenv():
    import subprocess
    try:
        subprocess.run(['pip', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("pip is installed.")
    except FileNotFoundError:
        print("pip is not installed.")

    try:
        subprocess.run(['virtualenv', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("virtualenv is installed.")
    except FileNotFoundError:
        print("virtualenv is not installed.")

def check_system_memory_and_disk():
    import psutil
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    print(f"Current System Memory: {memory.total / (1024 ** 3):.2f} GB")
    print(f"Available Disk Space: {disk.free / (1024 ** 3):.2f} GB")

if __name__ == "__main__":
    check_gpu()
    check_cuda()
    check_python_version()
    check_pip_virtualenv()
    check_system_memory_and_disk()