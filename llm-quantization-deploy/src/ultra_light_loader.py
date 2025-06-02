# Create: src/ultra_light_loader.py
#!/usr/bin/env python3
"""
Ultra-Light Model Loader for Systems with Limited RAM
Uses TinyLlama-1.1B which works with minimal resources
"""

import os
import torch
import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import time

# Set cache to E: drive
os.environ["HF_HOME"] = "E:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "E:/huggingface_cache"

class UltraLightLoader:
    """Load smallest possible 1B+ parameter model for low-RAM systems."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        # Use TinyLlama 1.1B - smallest decent model above 1B params
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
    def aggressive_memory_cleanup(self):
        """Extreme memory cleanup."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
    def create_minimal_bnb_config(self) -> BitsAndBytesConfig:
        """Most aggressive quantization possible."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    def load_model_ultra_conservative(self):
        """Load with maximum memory conservation."""
        
        print("=" * 70)
        print("🔬 ULTRA-LIGHT MODEL LOADING (1.1B PARAMETERS)")
        print("=" * 70)
        print("⚠️ Using TinyLlama due to severe RAM constraints")
        print("📊 Your system: 0.93GB available RAM")
        print()
        
        # Extreme cleanup first
        self.aggressive_memory_cleanup()
        
        # Load tokenizer with minimal memory
        print("📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir="E:/huggingface_cache",
            local_files_only=False,
            low_cpu_mem_usage=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ Tokenizer loaded: {self.tokenizer.vocab_size:,} tokens")
        
        # Load model with extreme constraints
        print("\n🔧 Loading TinyLlama with aggressive optimization...")
        start_time = time.time()
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=self.create_minimal_bnb_config(),
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                
                # EXTREME memory limits for your 0.93GB available RAM
                max_memory={0: "1.5GB", "cpu": "2GB"},
                offload_folder="E:/temp_offload",
                cache_dir="E:/huggingface_cache",
            )
            
            load_time = time.time() - start_time
            
            # Check memory usage
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
                print(f"✅ TinyLlama loaded in {load_time:.1f}s")
                print(f"📊 GPU memory: {gpu_memory:.2f} GB")
            else:
                print(f"✅ TinyLlama loaded in {load_time:.1f}s (CPU)")
            
            # Model info
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"🔍 Parameters: {total_params:,} (~1.1B)")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load TinyLlama: {e}")
            return False
    
    def generate_text(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate text with memory conservation."""
        
        if self.model is None:
            return "Model not loaded"
        
        # Format prompt for TinyLlama
        formatted_prompt = f"<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
        
        # Tokenize with truncation for memory
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            max_length=256,  # Short to save memory
            truncation=True
        )
        
        # Move to GPU if available
        if torch.cuda.is_available() and next(self.model.parameters()).device.type == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate with memory optimization
        start_time = time.time()
        self.aggressive_memory_cleanup()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_response.replace(formatted_prompt, "").strip()
        
        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        print(f"⚡ Generated {tokens_generated} tokens in {generation_time:.2f}s")
        
        # Cleanup after generation
        del inputs, outputs
        self.aggressive_memory_cleanup()
        
        return response

# Test the ultra-light loader
if __name__ == "__main__":
    print("🚨 SYSTEM RAM CRITICAL: 0.93GB available")
    print("💡 Using smallest viable model for your constraints")
    print()
    
    loader = UltraLightLoader()
    
    if loader.load_model_ultra_conservative():
        # Test with short prompts to save memory
        test_prompts = [
            "What is AI?",
            "How does ML work?",
            "Explain Python."
        ]
        
        print("\n" + "=" * 70)
        print("🧪 TESTING ULTRA-LIGHT GENERATION")
        print("=" * 70)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🧪 Test {i}: '{prompt}'")
            print("-" * 40)
            
            result = loader.generate_text(prompt, max_new_tokens=30)
            print(f"📝 Response: {result}")
        
        print("\n🎉 Ultra-light testing complete!")
    else:
        print("❌ Even ultra-light model failed - need to increase virtual memory first")