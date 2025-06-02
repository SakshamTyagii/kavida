
# Create: src/clear_cache.py
#!/usr/bin/env python3
"""
Clear HuggingFace Cache to Free Up Space
"""

import os
import shutil
from pathlib import Path

def clear_huggingface_cache():
    """Clear all HuggingFace cache directories."""
    
    cache_dirs = [
        "E:/huggingface_cache",
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.cache/torch"),
    ]
    
    total_freed = 0
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            # Calculate size before deletion
            size_before = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(cache_dir)
                for filename in filenames
            ) / 1024**3
            
            print(f"🗑️ Clearing {cache_dir}: {size_before:.2f} GB")
            
            try:
                shutil.rmtree(cache_dir)
                os.makedirs(cache_dir, exist_ok=True)
                total_freed += size_before
                print(f"✅ Cleared successfully!")
            except Exception as e:
                print(f"❌ Error clearing {cache_dir}: {e}")
        else:
            print(f"📁 {cache_dir} doesn't exist")
    
    print(f"\n🎉 Total space freed: {total_freed:.2f} GB")
    return total_freed

if __name__ == "__main__":
    print("=" * 60)
    print("🧹 CLEARING HUGGINGFACE CACHE")
    print("=" * 60)
    clear_huggingface_cache()