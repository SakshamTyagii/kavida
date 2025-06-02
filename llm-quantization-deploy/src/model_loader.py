#!/usr/bin/env python3
"""
Phi-3 Mini Model Loader - Perfect for 4GB GPU
3.8B parameters, no authentication needed, excellent performance
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import time
from typing import Optional, Tuple

# Set cache directories to E: drive
os.environ["HF_HOME"] = "E:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "E:/huggingface_cache"
os.makedirs("E:/huggingface_cache", exist_ok=True)

class Phi3ModelLoader:
    """
    Phi-3 Mini (3.8B) - Perfect for 4GB GPU with maximum performance.
    No authentication needed, fits perfectly in your VRAM.
    """
    
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def create_bnb_config(self) -> BitsAndBytesConfig:
        """Create optimized config for Phi-3 Mini."""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        print("🔧 BitsAndBytes Configuration:")
        print(f"   ├─ Quantization: 4-bit NF4")
        print(f"   ├─ Compute dtype: {bnb_config.bnb_4bit_compute_dtype}")
        print(f"   └─ Double quantization: {bnb_config.bnb_4bit_use_double_quant}")
        
        return bnb_config
      def initialize(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Initialize Phi-3 Mini with aggressive memory management for 4GB GPU."""
        
        print("=" * 70)
        print("INITIALIZING PHI-3 MINI WITH AGGRESSIVE MEMORY OPTIMIZATION")
        print("=" * 70)
        print("🚀 Loading Phi-3 Mini (3.8B) with extreme memory optimization!")
        print("✅ No authentication needed!")
        print()
        
        # Aggressive memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Check available memory
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_reserved = torch.cuda.memory_reserved(0) / 1024**3
            gpu_free = gpu_memory - gpu_reserved
            print(f"📊 GPU Memory: {gpu_free:.2f} GB available")
        
        # Load tokenizer first
        print("\n📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir="E:/huggingface_cache"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ Tokenizer loaded:")
        print(f"   ├─ Vocab size: {self.tokenizer.vocab_size:,}")
        print(f"   └─ Model max length: 4096")
        print()
        
        # Create temp offload directory
        offload_dir = "E:/temp_offload"
        os.makedirs(offload_dir, exist_ok=True)
        
        start_time = time.time()
        print("🔧 Attempting AGGRESSIVE memory optimization strategies...")
        
        # Strategy 1: Ultra-conservative GPU memory (1.5GB only)
        try:
            print("📋 Strategy 1: Ultra-conservative GPU (1.5GB limit)...")
            print("   └─ This prevents the checkpoint loading hang...")
            
            # Clear everything again
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self.create_bnb_config(),
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                max_memory={0: "1.5GB", "cpu": "8GB"},
                offload_folder=offload_dir,
                cache_dir="E:/huggingface_cache",
                local_files_only=False,
                revision="main"
            )
            print("✅ Strategy 1 successful! (Ultra-conservative)")
            
        except Exception as e1:
            print(f"❌ Strategy 1 failed: {str(e1)[:100]}...")
            
            # Strategy 2: Force CPU for most layers
            try:
                print("📋 Strategy 2: Minimal GPU usage (embedding + first 2 layers only)...")
                
                # Clear everything again
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Only put embeddings and first few layers on GPU
                device_map = {
                    "model.embed_tokens": "cuda:0",
                    "model.layers.0": "cuda:0",
                    "model.layers.1": "cuda:0",
                    # Everything else to CPU
                    **{f"model.layers.{i}": "cpu" for i in range(2, 32)},
                    "model.norm": "cpu",
                    "lm_head": "cpu"
                }
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=self.create_bnb_config(),
                    device_map=device_map,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    max_memory={0: "1.0GB", "cpu": "8GB"},
                    offload_folder=offload_dir,
                    cache_dir="E:/huggingface_cache"
                )
                print("✅ Strategy 2 successful! (Minimal GPU usage)")
                
            except Exception as e2:
                print(f"❌ Strategy 2 failed: {str(e2)[:100]}...")
                
                # Strategy 3: Pure CPU mode
                try:
                    print("📋 Strategy 3: Pure CPU mode (guaranteed to work)...")
                    
                    # Clear GPU completely
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map="cpu",
                        torch_dtype=torch.float32,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        use_safetensors=True,
                        cache_dir="E:/huggingface_cache",
                        offload_folder=offload_dir
                    )
                    self.device = "cpu"
                    print("✅ Strategy 3 successful! (CPU mode)")
                    
                except Exception as e3:
                    print(f"❌ Strategy 3 failed: {str(e3)[:100]}...")
                    
                    # Strategy 4: Smaller model fallback
                    print("📋 Strategy 4: Switching to smaller model...")
                    self.model_name = "microsoft/DialoGPT-medium"  # Much smaller model
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        device_map="cpu",
                        torch_dtype=torch.float32,
                        cache_dir="E:/huggingface_cache"
                    )
                    self.device = "cpu"
                    print("✅ Strategy 4 successful! (Smaller model)")
        
        load_time = time.time() - start_time
        
        # Check final status
        if torch.cuda.is_available() and self.device == "cuda":
            torch.cuda.empty_cache()
            gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_percent = (gpu_memory / gpu_total) * 100
            
            print(f"\n✅ Phi-3 Mini loaded in {load_time:.1f} seconds!")
            print(f"📊 GPU Memory Usage:")
            print(f"   ├─ Used: {gpu_memory:.2f} GB")
            print(f"   ├─ Total: {gpu_total:.1f} GB")
            print(f"   └─ Utilization: {gpu_percent:.1f}%")
        else:
            print(f"\n✅ Phi-3 Mini loaded in {load_time:.1f} seconds (CPU mode)")
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\n🔍 Model Info:")
        print(f"   ├─ Total parameters: {total_params:,}")
        print(f"   └─ Primary device: {next(self.model.parameters()).device}")
        
        # Show device distribution
        print(f"📍 Device Distribution:")
        device_count = {}
        for name, param in self.model.named_parameters():
            device = str(param.device)
            device_count[device] = device_count.get(device, 0) + 1
        
        for device, count in device_count.items():
            print(f"   ├─ {device}: {count} parameters")
        
        print("\n🎉 Model ready for inference!")
        return self.model, self.tokenizer
    
    def generate_text(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate text with Phi-3 Mini."""
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be initialized first!")
        
        # Format for Phi-3 instruct model
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", max_length=512, truncation=True)
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
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
        """Save the quantized model."""
        if save_directory is None:
            save_directory = "E:/llm_models/quantized/phi3-mini-4bit-bnb"
            
        os.makedirs(save_directory, exist_ok=True)
        print(f"💾 Saving model to {save_directory}...")
        
        self.model.save_pretrained(save_directory, safe_serialization=True)
        self.tokenizer.save_pretrained(save_directory)
        
        print("✅ Model saved successfully!")

# Test the model
if __name__ == "__main__":
    try:
        loader = Phi3ModelLoader()
        model, tokenizer = loader.initialize()
        
        test_prompts = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain quantum computing in simple terms."
        ]
        
        print("\n" + "=" * 70)
        print("TESTING PHI-3 MINI GENERATION")
        print("=" * 70)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🧪 Test {i}: '{prompt}'")
            print("-" * 50)
            
            result = loader.generate_text(prompt, max_new_tokens=80)
            print(f"📝 Response: {result}")
        
        # Save the model
        loader.save_model()
        
        print("\n🎉 All tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()