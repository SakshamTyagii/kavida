#!/usr/bin/env python3
"""
Diagnostic Loader to Identify GPU/Memory Issues
Helps diagnose why 3.8B parameter models fail to load
"""

import os
import torch
import psutil
import platform
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import time
import gc
from typing import Dict, Any

# Set cache directories to E: drive
os.environ["HF_HOME"] = "E:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "E:/huggingface_cache"
os.makedirs("E:/huggingface_cache", exist_ok=True)

class SystemDiagnostic:
    """Comprehensive system diagnostic for model loading issues."""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        
    def run_full_diagnostic(self) -> Dict[str, Any]:
        """Run complete system diagnostic."""
        
        print("=" * 80)
        print("🔍 SYSTEM DIAGNOSTIC FOR MODEL LOADING ISSUES")
        print("=" * 80)
        
        results = {}
        
        # 1. System Information
        print("\n1️⃣ SYSTEM INFORMATION:")
        print("-" * 40)
        results['system'] = self.check_system_info()
        
        # 2. GPU Information
        print("\n2️⃣ GPU INFORMATION:")
        print("-" * 40)
        results['gpu'] = self.check_gpu_info()
        
        # 3. Memory Information
        print("\n3️⃣ MEMORY INFORMATION:")
        print("-" * 40)
        results['memory'] = self.check_memory_info()
        
        # 4. PyTorch/Transformers Environment
        print("\n4️⃣ PYTORCH/TRANSFORMERS ENVIRONMENT:")
        print("-" * 40)
        results['environment'] = self.check_environment()
        
        # 5. Storage Information
        print("\n5️⃣ STORAGE INFORMATION:")
        print("-" * 40)
        results['storage'] = self.check_storage_info()
        
        # 6. Model Loading Test
        print("\n6️⃣ MODEL LOADING TEST:")
        print("-" * 40)
        results['model_test'] = self.test_model_loading()
        
        # 7. Recommendations
        print("\n7️⃣ RECOMMENDATIONS:")
        print("-" * 40)
        self.provide_recommendations(results)
        
        return results
    
    def check_system_info(self) -> Dict[str, Any]:
        """Check basic system information."""
        
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture()[0]
        }
        
        print(f"   Platform: {info['platform']}")
        print(f"   Processor: {info['processor']}")
        print(f"   Python: {info['python_version']} ({info['architecture']})")
        
        return info
    
    def check_gpu_info(self) -> Dict[str, Any]:
        """Check GPU information and capabilities."""
        
        info = {}
        
        if not self.gpu_available:
            print("   ❌ CUDA not available")
            info['available'] = False
            return info
        
        info['available'] = True
        info['device_count'] = torch.cuda.device_count()
        
        for i in range(info['device_count']):
            props = torch.cuda.get_device_properties(i)
            info[f'device_{i}'] = {
                'name': props.name,
                'total_memory_gb': props.total_memory / 1024**3,
                'multi_processor_count': props.multi_processor_count,
                'major': props.major,
                'minor': props.minor
            }
            
            print(f"   GPU {i}: {props.name}")
            print(f"   Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            
            # Check current memory usage
            try:
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                free = (props.total_memory / 1024**3) - reserved
                
                print(f"   Current Usage: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
                print(f"   Available: {free:.2f} GB")
                
                info[f'device_{i}']['current_allocated'] = allocated
                info[f'device_{i}']['current_reserved'] = reserved
                info[f'device_{i}']['available'] = free
                
            except Exception as e:
                print(f"   ⚠️ Error checking GPU memory: {e}")
        
        return info
    
    def check_memory_info(self) -> Dict[str, Any]:
        """Check system memory information."""
        
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        info = {
            'total_ram_gb': memory.total / 1024**3,
            'available_ram_gb': memory.available / 1024**3,
            'used_ram_gb': memory.used / 1024**3,
            'ram_percent': memory.percent,
            'total_swap_gb': swap.total / 1024**3,
            'used_swap_gb': swap.used / 1024**3,
            'swap_percent': swap.percent
        }
        
        print(f"   Total RAM: {info['total_ram_gb']:.2f} GB")
        print(f"   Available RAM: {info['available_ram_gb']:.2f} GB")
        print(f"   Used RAM: {info['used_ram_gb']:.2f} GB ({info['ram_percent']:.1f}%)")
        print(f"   Total Swap: {info['total_swap_gb']:.2f} GB")
        print(f"   Used Swap: {info['used_swap_gb']:.2f} GB ({info['swap_percent']:.1f}%)")
        
        # Check if RAM is sufficient
        if info['available_ram_gb'] < 8:
            print("   ⚠️ WARNING: Low available RAM (need 8GB+ for 3B models)")
        if info['total_swap_gb'] < 16:
            print("   ⚠️ WARNING: Low swap space (recommend 16GB+ virtual memory)")
        
        return info
    
    def check_environment(self) -> Dict[str, Any]:
        """Check PyTorch and Transformers environment."""
        
        info = {}
        
        try:
            info['torch_version'] = torch.__version__
            print(f"   PyTorch: {info['torch_version']}")
        except:
            print("   ❌ PyTorch not found")
        
        try:
            import transformers
            info['transformers_version'] = transformers.__version__
            print(f"   Transformers: {info['transformers_version']}")
        except:
            print("   ❌ Transformers not found")
        
        try:
            import bitsandbytes
            info['bitsandbytes_version'] = bitsandbytes.__version__
            print(f"   BitsAndBytes: {info['bitsandbytes_version']}")
        except:
            print("   ❌ BitsAndBytes not found")
        
        if self.gpu_available:
            info['cuda_version'] = torch.version.cuda
            print(f"   CUDA Version: {info['cuda_version']}")
            
            # Test CUDA operations
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                test_result = torch.matmul(test_tensor, test_tensor)
                del test_tensor, test_result
                torch.cuda.empty_cache()
                print("   ✅ CUDA operations working")
                info['cuda_operations'] = True
            except Exception as e:
                print(f"   ❌ CUDA operations failed: {e}")
                info['cuda_operations'] = False
        
        return info
    
    def check_storage_info(self) -> Dict[str, Any]:
        """Check storage space on relevant drives."""
        
        info = {}
        
        # Check C: drive
        try:
            c_usage = psutil.disk_usage('C:')
            info['c_drive'] = {
                'total_gb': c_usage.total / 1024**3,
                'free_gb': c_usage.free / 1024**3,
                'used_gb': c_usage.used / 1024**3,
                'percent': (c_usage.used / c_usage.total) * 100
            }
            print(f"   C: Drive - Free: {info['c_drive']['free_gb']:.1f} GB / {info['c_drive']['total_gb']:.1f} GB")
        except:
            print("   ⚠️ Cannot check C: drive")
        
        # Check E: drive
        try:
            e_usage = psutil.disk_usage('E:')
            info['e_drive'] = {
                'total_gb': e_usage.total / 1024**3,
                'free_gb': e_usage.free / 1024**3,
                'used_gb': e_usage.used / 1024**3,
                'percent': (e_usage.used / e_usage.total) * 100
            }
            print(f"   E: Drive - Free: {info['e_drive']['free_gb']:.1f} GB / {info['e_drive']['total_gb']:.1f} GB")
        except:
            print("   ⚠️ Cannot check E: drive")
        
        # Check cache directory
        cache_dir = "E:/huggingface_cache"
        if os.path.exists(cache_dir):
            cache_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                           for dirpath, dirnames, filenames in os.walk(cache_dir)
                           for filename in filenames) / 1024**3
            info['cache_size_gb'] = cache_size
            print(f"   HuggingFace Cache: {cache_size:.2f} GB")
        
        return info
    
    def test_model_loading(self) -> Dict[str, Any]:
        """Test loading progressively larger models to find the limit."""
        
        test_results = {}
        
        # Test models in order of increasing size
        test_models = [
            {"name": "gpt2", "size": "124M", "description": "GPT-2 Small"},
            {"name": "distilgpt2", "size": "82M", "description": "DistilGPT-2"},
            {"name": "microsoft/DialoGPT-small", "size": "117M", "description": "DialoGPT Small"},
            {"name": "microsoft/DialoGPT-medium", "size": "345M", "description": "DialoGPT Medium"},
            {"name": "microsoft/Phi-3-mini-4k-instruct", "size": "3.8B", "description": "Phi-3 Mini (Target)"}
        ]
        
        for model_info in test_models:
            print(f"\n   Testing: {model_info['description']} ({model_info['size']})")
            
            try:
                # Clear memory first
                if self.gpu_available:
                    torch.cuda.empty_cache()
                gc.collect()
                
                start_time = time.time()
                
                # Try loading with minimal config
                tokenizer = AutoTokenizer.from_pretrained(
                    model_info["name"],
                    cache_dir="E:/huggingface_cache"
                )
                
                if model_info["size"] == "3.8B":
                    # Use quantization for large model
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_info["name"],
                        quantization_config=bnb_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        max_memory={0: "2.0GB", "cpu": "6GB"},
                        cache_dir="E:/huggingface_cache"
                    )
                else:
                    # Load normally for small models
                    model = AutoModelForCausalLM.from_pretrained(
                        model_info["name"],
                        cache_dir="E:/huggingface_cache"
                    )
                
                load_time = time.time() - start_time
                
                # Check memory usage
                if self.gpu_available:
                    gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
                else:
                    gpu_memory = 0
                
                # Test generation
                inputs = tokenizer("Hello", return_tensors="pt")
                if self.gpu_available and next(model.parameters()).device.type == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=5)
                
                test_results[model_info["name"]] = {
                    "status": "SUCCESS",
                    "load_time": load_time,
                    "gpu_memory_gb": gpu_memory,
                    "device": str(next(model.parameters()).device)
                }
                
                print(f"   ✅ SUCCESS - Loaded in {load_time:.1f}s, GPU: {gpu_memory:.2f}GB")
                
                # Clean up
                del model, tokenizer, inputs, outputs
                if self.gpu_available:
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                error_msg = str(e)
                test_results[model_info["name"]] = {
                    "status": "FAILED",
                    "error": error_msg[:100]
                }
                
                if "out of memory" in error_msg.lower():
                    print(f"   ❌ FAILED - Out of memory")
                elif "checkpoint" in error_msg.lower():
                    print(f"   ❌ FAILED - Checkpoint loading issue")
                else:
                    print(f"   ❌ FAILED - {error_msg[:50]}...")
                
                # Clean up on failure
                if self.gpu_available:
                    torch.cuda.empty_cache()
                gc.collect()
        
        return test_results
    
    def provide_recommendations(self, results: Dict[str, Any]) -> None:
        """Provide specific recommendations based on diagnostic results."""
        
        recommendations = []
        
        # Check GPU memory
        if results.get('gpu', {}).get('available', False):
            gpu_info = results['gpu']['device_0']
            if gpu_info['total_memory_gb'] <= 4.5:
                recommendations.append("🎮 GPU has limited memory - use aggressive quantization and CPU offloading")
        
        # Check system RAM
        memory_info = results.get('memory', {})
        if memory_info.get('available_ram_gb', 0) < 8:
            recommendations.append("🧠 Increase system RAM or close other applications")
        
        if memory_info.get('total_swap_gb', 0) < 16:
            recommendations.append("💾 Increase Windows virtual memory to 16-32GB")
        
        # Check model loading results
        model_test = results.get('model_test', {})
        phi3_result = model_test.get('microsoft/Phi-3-mini-4k-instruct', {})
        
        if phi3_result.get('status') == 'FAILED':
            error = phi3_result.get('error', '').lower()
            if 'memory' in error:
                recommendations.append("🔧 Use more aggressive CPU offloading for Phi-3 Mini")
                recommendations.append("⚙️ Try loading with max_memory={0: '1.5GB', 'cpu': '8GB'}")
            elif 'checkpoint' in error:
                recommendations.append("📦 Try alternative model loading approach (load_in_8bit first)")
            else:
                recommendations.append("🔄 Try smaller model like TinyLlama-1.1B as alternative")
        
        # Environment issues
        env_info = results.get('environment', {})
        if not env_info.get('cuda_operations', True):
            recommendations.append("🔧 CUDA issues detected - reinstall PyTorch with CUDA support")
        
        # Print recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print("   ✅ System appears well-configured for model loading")

# Run diagnostic
if __name__ == "__main__":
    diagnostic = SystemDiagnostic()
    results = diagnostic.run_full_diagnostic()
    
    print(f"\n{'='*80}")
    print("🏁 DIAGNOSTIC COMPLETE")
    print(f"{'='*80}")
    print("📋 Check the recommendations above to resolve model loading issues.")
    print("💡 Share these results to get targeted help for your specific system.")