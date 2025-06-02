import psutil

def get_free_ram():
    """
    Retrieves the amount of free RAM in bytes.
    """
    memory = psutil.virtual_memory()
    return memory.available

if __name__ == "__main__":
    free_ram = get_free_ram()
    print(f"Free RAM: {free_ram / (1024 * 1024):.2f} MB")