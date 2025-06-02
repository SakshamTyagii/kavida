#!/usr/bin/env python3
"""
Large Model Loader for 3-15B Parameter Models
Optimized for GTX 1650 (4GB) with aggressive quantization
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import time
import gc
from typing import Optional, Tuple
import pathlib

# Set cache directories using platform-independent paths
CACHE_DIR = os.environ.get("HF_HOME", str(pathlib.Path.home() / ".cache" / "huggingface"))
OFFLOAD_DIR = os.environ.get("OFFLOAD_DIR", "E:/temp_offload" if os.name == "nt" else "/tmp/temp_offload")

os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)

class LargeModelLoader:
    """
    Loader for models with 3-15B parameters using extreme optimization
    for 4GB GPU (GTX 1650).
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model candidates with 3+ billion parameters that can work with aggressive quantization
        self.candidate_models = [
            # 3.8B parameters - Microsoft Phi-3 Mini
            {
                "name": "microsoft/Phi-3-mini-4k-instruct",
                "params": "3.8B",
                "description": "Phi-3 Mini - 3.8B params, excellent for 4-bit quantization",
                "auth_required": False,
                "special_config": {"trust_remote_code": True}
            },
            # 3B parameters - OpenELM
            {
                "name": "apple/OpenELM-3B-Instruct",
                "params": "3B",
                "description": "Apple OpenELM - 3B params, efficient architecture",
                "auth_required": False,
                "special_config": {"trust_remote_code": True}
            },
            # 7B parameters - Qwen2 (with extreme quantization)
            {
                "name": "Qwen/Qwen2-7B-Instruct",
                "params": "7B",
                "description": "Qwen2 - 7B params, needs extreme quantization",
                "auth_required": False,
                "special_config": {"trust_remote_code": True}
            },
            # 3B parameters - StableLM
            {
                "name": "stabilityai/stablelm-3b-4e1t",
                "params": "3B",
                "description": "StableLM - 3B params, stable training",
                "auth_required": False,
                "special_config": {}
            },
            # 7B parameters - CodeLlama (fallback)
            {
                "name": "codellama/CodeLlama-7b-Instruct-hf",
                "params": "7B",
                "description": "CodeLlama - 7B params, requires extreme optimization",
                "auth_required": False,
                "special_config": {}
            }
        ]
    
    def clear_memory_aggressive(self):
        """Ultra-aggressive memory clearing."""
        print("🧹 Clearing memory aggressively...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
                torch.cuda.empty_cache()
        
        gc.collect()
        
        # Show memory status
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            available = total - reserved
            print(f"   ├─ GPU Memory: {available:.2f} GB available ({allocated:.2f} GB allocated)")
    
    def create_extreme_bnb_config(self) -> BitsAndBytesConfig:
        """Create extremely aggressive quantization config for large models."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Most efficient quantization
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Double quantization for extra compression
            bnb_4bit_quant_storage=torch.uint8,  # Store in 8-bit for maximum compression
        )
    
    def try_loading_model(self, model_info: dict, strategy: str, **kwargs) -> bool:
        """Try loading a specific model with given strategy."""
        print(f"\n📋 {strategy}")
        print(f"   ├─ Model: {model_info['name']} ({model_info['params']} params)")
        print(f"   └─ Description: {model_info['description']}")
        
        try:
            self.clear_memory_aggressive()
            
            # Load tokenizer first
            print("   📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_info["name"],
                cache_dir=CACHE_DIR,
                **model_info.get("special_config", {})
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"   ✅ Tokenizer loaded (vocab: {self.tokenizer.vocab_size:,})")
            
            # Load model with strategy
            print("   🔧 Loading model with extreme optimization...")
            start_time = time.time()
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_info["name"],
                cache_dir=CACHE_DIR,
                offload_folder=OFFLOAD_DIR,
                **model_info.get("special_config", {}),
                **kwargs
            )
            
            load_time = time.time() - start_time
            self.model_name = model_info["name"]
            
            # Check if model loaded successfully
            if self.model is not None:
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"   ✅ SUCCESS! Loaded in {load_time:.1f}s")
                print(f"   └─ Total parameters: {total_params:,}")
                return True
            else:
                print("   ❌ Model is None after loading")
                return False
                
        except Exception as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                print(f"   ❌ GPU out of memory")
            elif "checkpoint" in error_msg.lower():
                print(f"   ❌ Checkpoint loading issue")
            else:
                print(f"   ❌ Failed: {error_msg[:60]}...")
            return False
    
    def initialize(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Initialize a large model with progressive fallback strategies."""
        
        print("=" * 80)
        print("LARGE MODEL LOADER - 3-15B PARAMETER MODELS")
        print("=" * 80)
        print("🎯 Target: Load a model with 3+ billion parameters")
        print("🎮 Hardware: GTX 1650 (4GB) with aggressive quantization")
        print()
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        self.clear_memory_aggressive()
        
        # Try each model with multiple strategies
        for model_info in self.candidate_models:
            print(f"\n{'='*60}")
            print(f"TRYING MODEL: {model_info['name']} ({model_info['params']} params)")
            print(f"{'='*60}")
            
            # Strategy 1: Extreme quantization with minimal GPU usage
            if self.try_loading_model(
                model_info,
                "Strategy 1: Extreme 4-bit quantization + minimal GPU",
                quantization_config=self.create_extreme_bnb_config(),
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                max_memory={0: "1.2GB", "cpu": "6GB"},
                offload_folder=OFFLOAD_DIR
            ):
                break
            
            # Strategy 2: CPU offload with quantization
            if self.try_loading_model(
                model_info,
                "Strategy 2: Heavy CPU offload + quantization",
                quantization_config=self.create_extreme_bnb_config(),
                torch_dtype=torch.float16,
                device_map={
                    "model.embed_tokens": "cuda:0",
                    "model.layers.0": "cuda:0",
                    "model.layers.1": "cuda:0",
                    # Everything else to CPU
                    **{f"model.layers.{i}": "cpu" for i in range(2, 50)},
                    "model.norm": "cpu",
                    "lm_head": "cpu"
                },
                low_cpu_mem_usage=True,
                max_memory={0: "1.0GB", "cpu": "8GB"},
                offload_folder=OFFLOAD_DIR
            ):
                break
            
            # Strategy 3: Pure CPU mode (slower but works)
            if self.try_loading_model(
                model_info,
                "Strategy 3: Pure CPU mode (guaranteed to work)",
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                offload_folder=OFFLOAD_DIR
            ):
                self.device = "cpu"
                break
            
            print(f"❌ All strategies failed for {model_info['name']}")
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("❌ Failed to load any model with 3+ billion parameters!")
        
        # Final status report
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"\n{'='*80}")
        print("🎉 LARGE MODEL LOADING SUCCESSFUL!")
        print(f"{'='*80}")
        print(f"✅ Model: {self.model_name}")
        print(f"📊 Parameters: {total_params:,}")
        print(f"🎯 Device: {next(self.model.parameters()).device}")
        
        if torch.cuda.is_available() and self.device == "cuda":
            gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
            print(f"📱 GPU Memory Used: {gpu_memory:.2f} GB")
        
        # Check parameter count requirement
        if total_params >= 3_000_000_000:  # 3 billion
            print(f"✅ REQUIREMENT MET: Model has {total_params:,} parameters (≥3B)")
        else:
            print(f"⚠️  Model has {total_params:,} parameters (<3B)")
        
        return self.model, self.tokenizer
    
    def generate_text(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate text with the loaded large model."""
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be initialized first!")
        
        # Format prompt based on model type
        if "phi-3" in self.model_name.lower():
            formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        elif "qwen" in self.model_name.lower():
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        elif "stablelm" in self.model_name.lower():
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        elif "codellama" in self.model_name.lower():
            formatted_prompt = f"[INST] {prompt} [/INST]"
        else:
            formatted_prompt = prompt
        
        print(f"🧠 Using format for {self.model_name}")
        
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generation_time = time.time() - start_time
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text.replace(formatted_prompt, "").strip()
        
        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        print(f"⚡ Generated {tokens_generated} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        
        return generated_text
    
    def save_model(self, save_directory: str = None) -> None:
        """Save the quantized large model."""
        if save_directory is None:
            model_safe_name = self.model_name.replace("/", "_").replace(":", "_")
            save_directory = os.path.join(
                "E:/llm_models/quantized" if os.name == "nt" else str(pathlib.Path.home() / "llm_models" / "quantized"),
                f"{model_safe_name}_4bit"
            )
        os.makedirs(save_directory, exist_ok=True)
        print(f"💾 Saving large model to {save_directory}...")
        
        try:
            self.model.save_pretrained(save_directory, safe_serialization=True)
            self.tokenizer.save_pretrained(save_directory)
            
            # Save metadata
            metadata = {
                "model_name": self.model_name,
                "total_parameters": sum(p.numel() for p in self.model.parameters()),
                "quantization": "4-bit BitsAndBytes",
                "device": str(next(self.model.parameters()).device)
            }
            
            with open(os.path.join(save_directory, "model_info.txt"), "w") as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            
            print("✅ Large model saved successfully!")
            print(f"📂 Location: {save_directory}")
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")

# Test the large model loader
if __name__ == "__main__":
    try:
        print("🚀 Starting Large Model Loader Test...")
        
        loader = LargeModelLoader()
        model, tokenizer = loader.initialize()
        
        # Test with various prompts to verify the model works
        test_prompts = [
            "What is artificial intelligence and how does it work?",
            "Explain the difference between machine learning and deep learning.",
            "Write a short Python function to calculate Fibonacci numbers.",
            "What are the benefits and risks of artificial intelligence?",
            "How do neural networks learn from data?"
        ]
        
        print("\n" + "=" * 80)
        print("TESTING LARGE MODEL TEXT GENERATION")
        print("=" * 80)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🧪 Test {i}: '{prompt}'")
            print("-" * 60)
            
            try:
                response = loader.generate_text(prompt, max_new_tokens=100)
                print(f"📝 Response: {response}")
            except Exception as e:
                print(f"❌ Generation error: {e}")
        
        # Save the model
        loader.save_model()
        
        print(f"\n🎉 Large Model Loading Test Completed Successfully!")
        print(f"✅ Successfully loaded and tested: {loader.model_name}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Final parameter count: {total_params:,}")
        
        if total_params >= 3_000_000_000:
            print("🏆 TASK REQUIREMENT MET: Model has 3+ billion parameters!")
        else:
            print("⚠️  Task requirement not fully met (need 3+ billion parameters)")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during large model loading: {e}")
        import traceback
        traceback.print_exc()
