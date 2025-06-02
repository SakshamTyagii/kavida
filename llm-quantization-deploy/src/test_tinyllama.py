#!/usr/bin/env python3
"""
TinyLlama Chat Model Test - Better for conversations
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import time

# Set cache directories to E: drive
os.environ["HF_HOME"] = "E:/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "E:/huggingface_cache"

def test_tinyllama():
    """Test TinyLlama which is much better for chat."""
    
    print("=" * 70)
    print("TESTING TINYLLAMA CHAT MODEL (1.1B PARAMETERS)")
    print("=" * 70)
    print("🚀 TinyLlama is specifically designed for chat and instruction following!")
    print()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🎮 GPU: {gpu_name}")
    
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Load tokenizer
    print("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="E:/huggingface_cache"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Tokenizer loaded (vocab: {tokenizer.vocab_size:,})")
    
    # Create quantization config for better GPU efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print("🔧 Loading TinyLlama with 4-bit quantization...")
    start_time = time.time()
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        cache_dir="E:/huggingface_cache"
    )
    
    load_time = time.time() - start_time
    
    # Check GPU usage
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated(0) / 1024**3
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        print(f"📱 GPU Memory: {gpu_memory:.2f} GB")
    
    # Test generation function
    def generate_response(prompt: str, max_new_tokens: int = 100) -> str:
        """Generate response with TinyLlama chat format."""
        
        # TinyLlama chat format
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        print(f"🧠 Formatted prompt:\n{formatted_prompt[:200]}...")
        
        # Tokenize
        inputs = tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|assistant|>" in generated_text:
            response = generated_text.split("<|assistant|>")[-1].strip()
        else:
            response = generated_text.replace(formatted_prompt, "").strip()
        
        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
        
        print(f"⚡ Generated {tokens_generated} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tok/s)")
        
        return response
    
    # Test with various prompts
    test_prompts = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Write a short poem about technology.",
        "Explain the difference between AI and machine learning.",
        "What are the benefits of using Python for data science?"
    ]
    
    print("\n" + "=" * 70)
    print("TESTING TINYLLAMA CHAT RESPONSES")
    print("=" * 70)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🧪 Test {i}: '{prompt}'")
        print("-" * 50)
        
        try:
            response = generate_response(prompt, max_new_tokens=80)
            print(f"📝 TinyLlama: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Save the model
    save_dir = "E:/llm_models/saved/tinyllama_chat_quantized"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n💾 Saving quantized TinyLlama to {save_dir}...")
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    print("✅ Model saved successfully!")
    
    print("\n🎉 TinyLlama test completed!")
    print(f"✅ Model works great for chat and uses minimal GPU memory!")
    
    return model, tokenizer

if __name__ == "__main__":
    try:
        model, tokenizer = test_tinyllama()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
