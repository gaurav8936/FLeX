import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import argparse

def merge_lora_into_base_model():
    # Base model path
    base_model_path = "codellama/CodeLlama-7b-hf"
    
    # LoRA adapter path
    lora_path = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_03_alpha16_dropout0.05_r8_lr0.0002_epochs3"
    
    # Output merged model path
    merged_model_name = os.path.basename(lora_path) + "_merged"
    merged_model_path = os.path.join(os.path.dirname(lora_path), merged_model_name)
    
    print(f"Loading base model from {base_model_path}...")
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    print(f"Loading LoRA adapter from {lora_path}...")
    # Load LoRA model
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("Merging LoRA weights into base model...")
    # Merge LoRA weights with base model
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {merged_model_path}...")
    # Save the merged model
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(os.path.join(merged_model_path, "tokenizer"))
    
    print("Model merging complete!")
    return merged_model_path

if __name__ == "__main__":
    merged_path = merge_lora_into_base_model()
    print(f"Merged model saved to: {merged_path}")