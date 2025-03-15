import argparse
import json
import numpy as np
import os
import time
import torch
import random
from tqdm import tqdm
from typing import Dict, List, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed,
)
from peft import PeftModel, PeftConfig
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom stopping criteria for code generation."""
    
    def __init__(self, start_length: int, eos_tokens: List[str], tokenizer):
        self.start_length = start_length
        self.eos_tokens = eos_tokens
        self.tokenizer = tokenizer
        
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        decoded = self.tokenizer.decode(input_ids[0][self.start_length:])
        # Stop when we detect a function ending or specific end marker
        if any(token in decoded for token in self.eos_tokens):
            return True
        return False


def complete_code(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95,
    n_samples: int = 1,
) -> List[str]:
    """Generate code completions for a given prompt."""
    # Prepare model inputs
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    # Define stopping criteria (end of function or specific tokens)
    eos_tokens = ["\ndef", "\nclass", "\nif __name__", "\n#", "\nprint", "\nassert"]
    stopping_criteria = StoppingCriteriaList([
        EndOfFunctionCriteria(input_length, eos_tokens, tokenizer)
    ])
    
    # Store all generated samples
    all_completions = []
    
    for _ in range(n_samples):
        # Generate with specified parameters
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=n_samples > 1,  # Use sampling for multiple samples
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
            )
        
        # Decode the completion
        completion = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        all_completions.append(completion)
    
    return all_completions


def evaluate_on_humaneval(
    model_path: str,
    output_dir: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    n_samples: int = 1,
    max_new_tokens: int = 512,
    pass_at_k: List[int] = [1],
    device: str = "cuda",
    load_in_8bit: bool = False,
) -> Dict:
    """Evaluate model on HumanEval benchmark."""
    # Load model and tokenizer
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # Load as PEFT model if adapter_config.json exists
        peft_config = PeftConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        
        # Load model with appropriate quantization
        if load_in_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                load_in_8bit=True,
                device_map={"": device},
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                torch_dtype=torch.float16,
                device_map={"": device},
            )
        
        # Load PEFT adapter
        model = PeftModel.from_pretrained(model, model_path)
    else:
        # Load as a regular model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if load_in_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=True,
                device_map={"": device},
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map={"": device},
            )
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set model to evaluation mode
    model.eval()
    
    # Load HumanEval problems
    problems = read_problems()
    
    # Generate completions for each problem
    samples = []
    
    for task_id, problem in tqdm(problems.items(), desc="Generating solutions"):
        prompt = problem["prompt"]
        
        # Generate n_samples completions
        completions = complete_code(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            n_samples=n_samples,
        )
        
        # Format the samples for evaluation
        for completion in completions:
            samples.append({
                "task_id": task_id,
                "completion": completion,
            })
    
    # Save the generated samples
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    samples_file = os.path.join(output_dir, f"humaneval_samples_{timestamp}.jsonl")
    write_jsonl(samples_file, samples)
    
    # Evaluate the samples
    results = evaluate_functional_correctness(samples, k=max(pass_at_k))
    
    # Format results
    formatted_results = {}
    for k in pass_at_k:
        if k <= n_samples:
            formatted_results[f"pass@{k}"] = results[f"pass@{k}"] * 100
    
    # Save results to file
    results_file = os.path.join(output_dir, f"humaneval_results_{timestamp}.json")
    with open(results_file, "w") as f:
        json.dump(formatted_results, f, indent=2)
    
    print("\n" + "=" * 50)
    print(f"Evaluation Results:")
    for k, score in formatted_results.items():
        print(f"{k}: {score:.2f}%")
    print("=" * 50)
    
    return formatted_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a code model on HumanEval")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or adapter")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling probability")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit quantization")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Evaluate model on HumanEval
    evaluate_on_humaneval(
        model_path=args.model_path,
        output_dir=args.output_dir,
        temperature=args.temperature,
        top_p=args.top_p,
        n_samples=args.n_samples,
        max_new_tokens=args.max_new_tokens,
        pass_at_k=[1, min(args.n_samples, 10), min(args.n_samples, 100)],  # Calculate pass@k based on samples
        device=args.device,
        load_in_8bit=args.load_in_8bit,
    )


if __name__ == "__main__":
    main()