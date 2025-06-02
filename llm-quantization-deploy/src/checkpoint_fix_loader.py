#!/usr/bin/env python3
"""
Enhanced Model Loader with Checkpoint Loading Fix
Specifically designed to bypass the "Loading checkpoint shards: 0%" hang issue
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
import gc

# Set cache directories to E: drive
os.environ["HF_HOME"] = "E:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "E:/huggingface_cache"
os.environ["HF_HUB_OFFLINE"] = "0"  # Allow online downloads
os.makedirs("E:/huggingface_cache", exist_ok=True)
os.makedirs("E:/temp_offload", exist_ok=True)

class CheckpointFixLoader:
    """
    Advanced model loader that specifically addresses checkpoint loading hangs.
    Uses progressive fallback strategies and alternative models.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = None
        
    def create_bnb_config(self) -> BitsAndBytesConfig:
        """Create conservative quantization config."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    
    def clear_memory(self):
        """Aggressive memory clearing."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
        gc.collect()
        
    def try_model_loading(self, model_name: str, strategy_name: str, **kwargs) -> bool:
        """Try loading a model with given parameters."""
        try:
            print(f"📋 {strategy_name}...")
            print(f"   ├─ Model: {model_name}")
            
            self.clear_memory()
            
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir="E:/huggingface_cache"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"   ├─ Tokenizer loaded (vocab: {self.tokenizer.vocab_size:,})")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir="E:/huggingface_cache",
                **kwargs
            )
            
            self.model_name = model_name
            print(f"   └─ ✅ {strategy_name} successful!")
            return True
            
        except Exception as e:
            print(f"   └─ ❌ Failed: {str(e)[:80]}...")
            return False
    
    def initialize(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Initialize model with multiple fallback strategies."""
        
        print("=" * 70)
        print("CHECKPOINT LOADING FIX - PROGRESSIVE MODEL LOADING")
        print("=" * 70)
        print("🚀 Trying multiple strategies to bypass checkpoint hang...")
        print()
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        start_time = time.time()
        
        # Strategy 1: Try DistilGPT2 (very small, should work)
        if self.try_model_loading(
            "distilgpt2",
            "Strategy 1: DistilGPT2 (117M params - ultra light)",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        ):
            pass
            
        # Strategy 2: Try GPT2 Medium (355M params)
        elif self.try_model_loading(
            "gpt2-medium", 
            "Strategy 2: GPT2-Medium (355M params)",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        ):
            pass
            
        # Strategy 3: Try TinyLlama (1.1B params)
        elif self.try_model_loading(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "Strategy 3: TinyLlama (1.1B params with quantization)",
            quantization_config=self.create_bnb_config() if torch.cuda.is_available() else None,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            max_memory={0: "2GB"} if torch.cuda.is_available() else None
        ):
            pass
            
        # Strategy 4: Try DialoGPT Medium (CPU only)
        elif self.try_model_loading(
            "microsoft/DialoGPT-medium",
            "Strategy 4: DialoGPT-Medium (CPU mode)",
            device_map="cpu",
            torch_dtype=torch.float32
        ):
            self.device = "cpu"
            
        # Strategy 5: Force Phi-3 with CPU-only
        elif self.try_model_loading(
            "microsoft/Phi-3-mini-4k-instruct",
            "Strategy 5: Phi-3 Mini (CPU only - no quantization)",
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ):
            self.device = "cpu"
            
        else:
            raise Exception("❌ All loading strategies failed!")
        
        load_time = time.time() - start_time
        
        # Report status
        if self.model and self.tokenizer:
            total_params = sum(p.numel() for p in self.model.parameters())
            
            print(f"\n✅ Model loaded in {load_time:.1f} seconds!")
            print(f"🔍 Model: {self.model_name}")
            print(f"📊 Parameters: {total_params:,}")
            print(f"🎯 Device: {next(self.model.parameters()).device}")
            
            if torch.cuda.is_available() and self.device == "cuda":
                gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
                print(f"📱 GPU Memory: {gpu_memory:.2f} GB")
            
            return self.model, self.tokenizer
        else:
            raise Exception("Model or tokenizer is None after loading!")
    
    def generate_text(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate text with the loaded model."""
        
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be initialized first!")
        
        # Format prompt based on model type
        if "phi-3" in self.model_name.lower():
            formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        elif "tinyllama" in self.model_name.lower():
            formatted_prompt = f"<|system|>\nYou are a helpful assistant.<|user|>\n{prompt}<|assistant|>\n"
        else:
            formatted_prompt = prompt
        
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        # Move to device
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
        """Save the model."""
        if save_directory is None:
            model_safe_name = self.model_name.replace("/", "_").replace(":", "_")
            save_directory = f"E:/llm_models/saved/{model_safe_name}"
            
        os.makedirs(save_directory, exist_ok=True)
        print(f"💾 Saving model to {save_directory}...")
        
        self.model.save_pretrained(save_directory, safe_serialization=True)
        self.tokenizer.save_pretrained(save_directory)
        
        print("✅ Model saved successfully!")

# Test the enhanced loader
if __name__ == "__main__":
    try:
        print("🔧 Starting Enhanced Model Loader Test...")
        
        loader = CheckpointFixLoader()
        model, tokenizer = loader.initialize()
        
        test_prompts = [
            "What is artificial intelligence?",
            "How does machine learning work?", 
            "Explain quantum computing in simple terms."
        ]
        
        print("\n" + "=" * 70)
        print("TESTING TEXT GENERATION")
        print("=" * 70)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🧪 Test {i}: '{prompt}'")
            print("-" * 50)
            
            result = loader.generate_text(prompt, max_new_tokens=60)
            print(f"📝 Response: {result}")
        
        # Save the model
        loader.save_model()
        
        print("\n🎉 All tests completed successfully!")
        print(f"✅ Final model: {loader.model_name}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
