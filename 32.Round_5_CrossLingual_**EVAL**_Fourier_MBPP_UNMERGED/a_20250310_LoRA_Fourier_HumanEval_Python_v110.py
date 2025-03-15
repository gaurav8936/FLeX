#!/usr/bin/env python3
# Purpose: Evaluate Code Llama 7B with LoRA models on HumanEval using different configurations

##################### Cheatsheet -- run time ################################

# # Run first two models with first three configurations
# python a_20250309_LoRA_Fourier_HumanEval_Python_v120.py --model_range "0:2" --config_range "0:3" --gpu_id 0

# # Run models 2-4 with configurations 1-3
# python a_20250309_LoRA_Fourier_HumanEval_Python_v120.py --model_range "2:5" --config_range "1:4" --gpu_id 1

# # Run specific model with all configurations
# python a_20250309_LoRA_Fourier_HumanEval_Python_v120.py --model_range "3:4" --gpu_id 2

####### Specific model and config ########

# python a_20250309_LoRA_Fourier_HumanEval_Python_v110.py --model_range "14:15" --config_range "6:7"

######### Specific Model -- all configs #########
# python a_20250309_LoRA_Fourier_HumanEval_Python_v110.py --model_range "2:3" --config_range "0:15" --gpu_id 0

############### Run all configs for index 8 & 9 #########################
# python a_20250310_LoRA_Fourier_HumanEval_Python_v110.py --model_range "8:9" --config_range "0:15"
# python a_20250310_LoRA_Fourier_HumanEval_Python_v110.py --model_range "9:10" --config_range "0:15"
# python a_20250310_LoRA_Fourier_HumanEval_Python_v110.py --model_range "7:8" --config_range "0:15"
# python a_20250310_LoRA_Fourier_HumanEval_Python_v110.py --model_range "3:4" --config_range "0:15
# python a_20250310_LoRA_Fourier_HumanEval_Python_v110.py --model_range "3:4" --config_range "8:15"

# CHEATSHEET #
# 20250310_bph_alpha16_dropout0.05_r8_lr0.0002_fou0.01_epochs3  --> INDEX 0
# 20250310_wov_alpha16_dropout0.05_r8_lr0.0002_fou0.05_epochs3  --> INDEX 1
# 20250310_rpl_alpha16_dropout0.05_r8_lr0.0002_fou0.001_epochs3  --> INDEX 2
# 20250310_kfg_alpha16_dropout0.05_r8_lr0.0002_fou0.01_epochs3  --> INDEX 3
# 20250310_cbg_alpha16_dropout0.05_r8_lr0.0002_fou0.01_epochs3  --> INDEX 4
# 20250310_kjg_alpha16_dropout0.05_r8_lr0.0002_fou0.01_epochs3  --> INDEX 5
# 20250310_bch_alpha16_dropout0.05_r16_lr0.0002_fou0.01_epochs3  --> INDEX 6
# 20250310_hiv_alpha16_dropout0.05_r16_lr0.0002_fou0.03_epochs3  --> INDEX 7
# 20250310_gkm_alpha16_dropout0.05_r8_lr0.0002_fou0.02_epochs3  --> INDEX 8
# 20250310_wdq_alpha16_dropout0.05_r8_lr0.0002_fou0.02_epochs3  --> INDEX 9

# "High beam + medium temp" (beams=8, temp=0.25, top_p=0.95) --> INDEX 0
# "High beam (12) + low temp (0.2)" (beams=12, temp=0.2, top_p=0.92) --> INDEX 1
# "Greedy with length penalty" (beams=10, do_sample=False, temp=1.0, length_penalty=1.05) --> INDEX 2
# "Very low temp + repetition penalty" (beams=8, temp=0.15, top_p=0.9, repetition_penalty=1.1) --> INDEX 3
# "High beam + low temp" (beams=10, temp=0.2, top_p=0.95) --> INDEX 4
# "Pure greedy with high beam" (beams=12, do_sample=False, temp=1.0) --> INDEX 5
# "Early stopping + optimal params" (beams=8, temp=0.18, top_p=0.92, early_stopping=True) --> INDEX 6
# "Diversity penalty" (beams=9, temp=0.15, diversity_penalty=0.2) --> INDEX 7
# "Balanced beam + low temp" (beams=10, temp=0.2, top_p=0.9) --> INDEX 8
# "Pure greedy" (beams=8, do_sample=False, temp=1.0) --> INDEX 9
# "High beam + repetition penalty" (beams=12, temp=0.15, top_p=0.95, repetition_penalty=1.15) --> INDEX 10
# "Balanced params" (beams=10, temp=0.22, top_p=0.92) --> INDEX 11
# "Greedy with length penalty (alt)" (beams=12, do_sample=False, temp=1.0, length_penalty=1.05) --> INDEX 12
# "Very low temp + repetition" (beams=8, temp=0.15, top_p=0.9, repetition_penalty=1.2) --> INDEX 13
# "Best tuned params" (beams=10, temp=0.18, top_p=0.92, repetition_penalty=1.05) --> INDEX 14


####################################################################################

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel  # Add this import
from human_eval.evaluation import evaluate_functional_correctness
from human_eval.data import read_problems, write_jsonl
import warnings
import os
import logging
from tqdm import tqdm
import time
from datetime import datetime
import csv
import shutil
import argparse
import gc

from b_20250310_LoRA_Fourier_HumanEval_Python_Config_v120 import get_evaluation_configurations

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress transformer warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

def setup_logging(log_dir):
    """Set up logging with both file and console handlers"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = os.path.join(log_dir, f"{timestamp}_lora_eval.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__), timestamp

def backup_tracking_file(tracking_file, backup_dir):
    """Backup tracking CSV file to backup directory with timestamp"""
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_file = os.path.join(backup_dir, f"{timestamp}_eval_tracking_backup.csv")
    
    if os.path.exists(tracking_file):
        shutil.copy2(tracking_file, backup_file)
        logging.info(f"Backed up tracking file to {backup_file}")

def update_tracking_file(tracking_file, backup_dir, config, eval_timestamp, duration_minutes, 
                         num_problems, pass_score, samples_file, summary_file):
    """Update tracking CSV file with details of the current evaluation run"""
    # Create backup of existing tracking file
    backup_tracking_file(tracking_file, backup_dir)
    
    # Create the tracking directory if it doesn't exist
    os.makedirs(os.path.dirname(tracking_file), exist_ok=True)
    
    # Extract key parameters to track
    base_model = config["base_model"]["name"]
    lora_model = config["lora_model"]["path"]
    metric = config["evaluation"]["metric"]
    num_samples = config["evaluation"]["num_samples_per_problem"]
    temperature = config["evaluation"]["generation"].get("temperature", "N/A")
    num_beams = config["evaluation"]["generation"].get("num_beams", "N/A")
    top_p = config["evaluation"]["generation"].get("top_p", "N/A")
    do_sample = config["evaluation"]["generation"].get("do_sample", "N/A")
    max_new_tokens = config["evaluation"]["generation"].get("max_new_tokens", "N/A")
    config_description = config["evaluation"].get("config_description", "baseline")
    
    # Define field names for the CSV file
    field_names = [
        "timestamp", "base_model", "lora_model", "metric", "num_samples", 
        "temperature", "num_beams", "top_p", "do_sample", "max_new_tokens",
        "num_problems", "score", "duration_minutes", "samples_file", "summary_file",
        "config_description"
    ]
    
    # Prepare data for the new row
    row_data = {
        "timestamp": eval_timestamp,
        "base_model": base_model,
        "lora_model": os.path.basename(lora_model),
        "metric": metric,
        "num_samples": num_samples,
        "temperature": temperature,
        "num_beams": num_beams,
        "top_p": top_p,
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "num_problems": num_problems,
        "score": f"{pass_score:.2f}",
        "duration_minutes": f"{duration_minutes:.2f}",
        "samples_file": os.path.basename(samples_file),
        "summary_file": os.path.basename(summary_file),
        "config_description": config_description
    }
    
    # Check if file exists to determine if header should be written
    file_exists = os.path.isfile(tracking_file)
    
    # Write to CSV
    with open(tracking_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write data row
        writer.writerow(row_data)
    
    logging.info(f"Updated tracking file at {tracking_file}")

def clean_completion(completion: str, prompt: str) -> str:
    """
    Clean and normalize the completion output with robust function boundary detection.
    
    Args:
        completion (str): Raw completion from the model
        prompt (str): Original prompt for reference
        
    Returns:
        str: Cleaned and properly formatted completion
    """
    # First apply the same initial cleaning as script 1
    completion = completion.split("\n\n")[0].rstrip()
    
    # Split into lines
    completion_lines = completion.split('\n')
    cleaned_lines = []
    
    # Track function body
    in_function = False
    function_indent = None
    
    for line in completion_lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append('')
            continue
            
        # Detect function start
        if stripped.startswith('def '):
            in_function = True
            function_indent = len(line) - len(stripped)
            cleaned_lines.append(line)
            continue
            
        # Add all non-empty lines
        if stripped:
            cleaned_lines.append(line)
    
    # Get base indentation from prompt
    base_indent = None
    for line in prompt.split('\n'):
        if line.strip():
            base_indent = len(line) - len(line.lstrip())
            break
    
    if base_indent is None:
        base_indent = 4

    # Process each line preserving the original indentation structure
    lines = []
    for line in cleaned_lines:
        if not line.strip():
            lines.append('')
            continue
            
        # Calculate the relative indentation level from the original line
        stripped = line.lstrip()
        indent_level = (len(line) - len(stripped)) // 4
        
        # Apply the base indentation plus relative indentation
        indented_line = ' ' * (base_indent + (indent_level * 4)) + stripped
        lines.append(indented_line)
    
    return '\n'.join(lines)

def generate_solution(model, tokenizer, prompt: str, config, sample_idx=0):
    """
    Generate a solution completion for a given prompt using the provided configuration.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer to use for encoding/decoding
        prompt (str): The input prompt containing the function signature and docstring
        config: Configuration dictionary with generation parameters
        sample_idx (int): Index to use for random seed
        
    Returns:
        str: The generated completion
    """
    # Set a different seed for each sample to ensure diversity
    torch.manual_seed(42 + sample_idx)
    
    # Extract generation parameters from config
    generation_params = config["evaluation"]["generation"]
    
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate completion
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=generation_params.get("max_new_tokens", 256),
            temperature=generation_params.get("temperature", 0.2),
            num_beams=generation_params.get("num_beams", 3),
            top_p=generation_params.get("top_p", 0.90),
            top_k=generation_params.get("top_k", 0),
            repetition_penalty=generation_params.get("repetition_penalty", 1.0),
            do_sample=generation_params.get("do_sample", True),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Get the full generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract just the completion part
    completion = generated_text[len(prompt):]
    
    # Use the clean_completion function
    return clean_completion(completion, prompt)

def run_evaluation(config, eval_number=1):
    """
    Run a model evaluation with the given configuration
    
    Args:
        config: Configuration dictionary
        eval_number: Sequential number for this evaluation
        
    Returns:
        tuple: (pass_score, duration_minutes, samples_file, summary_file)
    """
    # Setup logging
    logger, eval_timestamp = setup_logging(config["logging"]["log_dir"])
    
    logger.info(f"Starting HumanEval {config['evaluation']['metric']} evaluation #{eval_number}")
    logger.info(f"Base model: {config['base_model']['name']}")
    logger.info(f"LoRA adapter path: {config['lora_model']['path']}")
    logger.info(f"Generation parameters: {config['evaluation']['generation']}")
    logger.info(f"Device mapping: {config['base_model']['device_map']}")
    
    # Ensure output directory exists
    os.makedirs(config["output"]["results_dir"], exist_ok=True)
    
    try:
        # Set specific CUDA device if not "auto"
        if isinstance(config["base_model"]["device_map"], str) and config["base_model"]["device_map"] != "auto" and "cuda:" in config["base_model"]["device_map"]:
            gpu_id = int(config["base_model"]["device_map"].replace("cuda:", ""))
            torch.cuda.set_device(gpu_id)
            logger.info(f"Set active CUDA device to: {torch.cuda.current_device()}")
            
            # Check available GPU memory before loading
            free_memory = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory - torch.cuda.memory_allocated()
            logger.info(f"Available GPU memory before loading: {free_memory / 1024**3:.2f} GB")
        else:
            logger.info(f"Using automatic device mapping")
        
        # Clear all CUDA memory before starting
        torch.cuda.empty_cache()
        gc.collect()
        
        # Load tokenizer from the base model name
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"]["name"])
        
        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load the base model and LoRA adapter separately
        try:
            # First load the base model
            logger.info(f"Loading base model from {config['base_model']['name']}...")
            base_model = AutoModelForCausalLM.from_pretrained(
                config["base_model"]["name"],
                torch_dtype=getattr(torch, config["base_model"]["dtype"]),
                device_map=config["base_model"]["device_map"],
                max_position_embeddings=config["base_model"].get("max_position_embeddings", None)
            )
            
            # Then load the LoRA adapter
            logger.info(f"Loading LoRA adapter from {config['lora_model']['path']}...")
            model = PeftModel.from_pretrained(
                base_model,
                config["lora_model"]["path"],
                torch_dtype=getattr(torch, config["base_model"]["dtype"]),
                device_map=config["base_model"]["device_map"]
            )
            
            model.eval()  # Set model to inference mode
            logger.info("Model with LoRA adapter loaded successfully")
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory during model loading: {str(e)}")
            return None, None, None, None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None, None, None, None
        
        # Load HumanEval problems
        logger.info("Loading HumanEval problems")
        problems = read_problems()
        logger.info(f"Loaded {len(problems)} HumanEval problems")
        
        samples = []
        num_samples_per_problem = config["evaluation"]["num_samples_per_problem"]
        
        # Start the timer
        start = time.time()
        
        # Process each problem
        for task_id, problem_data in tqdm(problems.items(), desc="Processing problems", total=len(problems)):
            logger.info(f"Processing {task_id}")
            prompt = problem_data["prompt"]
            
            # Generate samples per problem
            for sample_idx in tqdm(range(num_samples_per_problem), desc=f"Generating samples for {task_id}", leave=False):
                try:
                    generated_code = generate_solution(model, tokenizer, prompt, config, sample_idx=sample_idx)
                    samples.append({"task_id": task_id, "completion": generated_code})
                except Exception as e:
                    logger.error(f"Error generating solution for {task_id}, sample {sample_idx}: {str(e)}")
                    continue
        
        # End the timer
        end = time.time()
        duration_minutes = (end - start) / 60
        logger.info(f"Generation completed in {duration_minutes:.2f} minutes")
        
        # Save the samples
        samples_file = os.path.join(config["output"]["results_dir"], f"{eval_timestamp}_generated_samples.jsonl")
        write_jsonl(samples_file, samples)
        logger.info(f"Wrote samples to {samples_file}")
        
        # Evaluate the results
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k_value = int(config["evaluation"]["metric"].split("@")[1])
            results = evaluate_functional_correctness(samples_file, [k_value], ignore_incomplete=True)
        
        pass_score = results[config["evaluation"]["metric"]] * 100
        logger.info(f"{config['evaluation']['metric']} Score: {pass_score:.2f}%")
        
        # Write the results to a summary file
        summary_file = os.path.join(config["output"]["results_dir"], f"{eval_timestamp}_results_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Base Model: {config['base_model']['name']}\n")
            f.write(f"LoRA Adapter Path: {config['lora_model']['path']}\n")
            f.write(f"LoRA Adapter Name: {os.path.basename(config['lora_model']['path'])}\n")
            f.write(f"Config Description: {config['evaluation'].get('config_description', 'baseline')}\n")
            f.write(f"Timestamp: {eval_timestamp}\n")
            f.write(f"Duration: {duration_minutes:.2f} minutes\n")
            f.write(f"Number of problems: {len(problems)}\n")
            f.write(f"Samples per problem: {num_samples_per_problem}\n")
            f.write(f"{config['evaluation']['metric']} Score: {pass_score:.2f}%\n")
            f.write(f"Generation parameters:\n")
            for key, value in config["evaluation"]["generation"].items():
                f.write(f"  - {key}: {value}\n")
        
        logger.info(f"Results summary written to {summary_file}")
        
        # Clean up model and cache to free memory
        del model
        del base_model
        torch.cuda.empty_cache()
        gc.collect()

        return pass_score, duration_minutes, samples_file, summary_file
        
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}", exc_info=True)
        # Clean up if there's an error
        if 'model' in locals():
            del model
        if 'base_model' in locals():
            del base_model
        torch.cuda.empty_cache()
        gc.collect()
        return None, None, None, None

def main():
    parser = argparse.ArgumentParser(description="Evaluate Code Llama with LoRA adapters on HumanEval.")
    parser.add_argument("--config_range", type=str, default=":1",
                        help="Specify the range of configurations to use, e.g., ':1', '1:2', '2:'")
    parser.add_argument("--model_range", type=str, default=":1",
                        help="Specify the range of models to evaluate, e.g., ':1', '1:2', '2:'")
    parser.add_argument("--gpu_id", type=str, default="auto",
                        help="Specify which GPU to use (0-7 or 'auto' for automatic assignment)")
    args = parser.parse_args()

    # Get evaluation configurations with the specified model and config ranges
    all_configs = get_evaluation_configurations(args.model_range, args.config_range)
    
    # Update device mapping for all configs
    for config in all_configs:
        if args.gpu_id.lower() == "auto":
            config["base_model"]["device_map"] = "auto"
        else:
            try:
                gpu_id = int(args.gpu_id)
                config["base_model"]["device_map"] = f"cuda:{gpu_id}"
            except ValueError:
                print(f"Invalid GPU ID: {args.gpu_id}. Using 'auto' instead.")
                config["base_model"]["device_map"] = "auto"

    logger, _ = setup_logging("logs")
    logger.info(f"Starting evaluation loop for {len(all_configs)} configurations")
    logger.info(f"Selected model range: {args.model_range}")
    logger.info(f"Selected config range: {args.config_range}")
    
    # Log which models will be evaluated
    model_names = [os.path.basename(config['lora_model']['path']) for config in all_configs]
    unique_models = list(dict.fromkeys(model_names))  # Remove duplicates while preserving order
    logger.info(f"Models to evaluate ({len(unique_models)}):")
    for i, model in enumerate(unique_models):
        logger.info(f"  Model {i}: {model}")
    
    # Log which configurations will be used
    config_descriptions = [config['evaluation'].get('config_description', 'baseline') for config in all_configs[:len(unique_models)]]
    logger.info(f"Configurations to use:")
    for i, desc in enumerate(set(config_descriptions)):
        logger.info(f"  Config {i}: {desc}")
    
    logger.info(f"Device mapping: {all_configs[0]['base_model']['device_map'] if all_configs else 'N/A'}")
    
    # Force no HuggingFace parallelism if using a specific GPU
    if args.gpu_id.lower() != "auto":
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            gpu_id = int(args.gpu_id)
            torch.cuda.set_device(gpu_id)
            logger.info(f"Set active CUDA device to: {torch.cuda.current_device()}")
            logger.info(f"Total CUDA memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.2f} GB")
        except ValueError:
            logger.info("Using automatic device mapping")
    else:
        logger.info("Using automatic device mapping")
    
    # Loop through each configuration
    for i, config in enumerate(all_configs, 1):
        # Aggressively clean memory before each run
        torch.cuda.empty_cache()
        gc.collect()
        
        # Small sleep to ensure resources are released
        time.sleep(5)
        
        model_name = os.path.basename(config["lora_model"]["path"])
        config_desc = config["evaluation"].get("config_description", "baseline config")
        
        # Get model and config indices from file names and descriptions
        try:
            model_index = next(idx for idx, name in enumerate(unique_models) if name == model_name)
        except StopIteration:
            model_index = "unknown"
            
        try:
            config_index = next(idx for idx, desc in enumerate(set(config_descriptions)) if desc == config_desc)
        except StopIteration:
            config_index = "unknown"
            
        logger.info(f"Running evaluation {i}/{len(all_configs)}: Model {model_index} ({model_name}), Config {config_index} ({config_desc})")
        
        try:
            # Log available memory before starting if using a specific GPU
            if args.gpu_id.lower() != "auto":
                try:
                    gpu_id = int(args.gpu_id)
                    free_memory = torch.cuda.get_device_properties(gpu_id).total_memory - torch.cuda.memory_allocated()
                    logger.info(f"Available GPU memory before starting: {free_memory / 1024**3:.2f} GB")
                except ValueError:
                    pass
            
            pass_score, duration_minutes, samples_file, summary_file = run_evaluation(config, eval_number=i)
            
            if pass_score is not None:
                logger.info(f"Successfully evaluated config {i}: Model {model_index} ({model_name}), Config {config_index} ({config_desc}), Score={pass_score:.2f}%, Time={duration_minutes:.2f} min")
                
                # Update tracking file
                update_tracking_file(
                    config["tracking"]["file"],
                    config["tracking"]["backup_dir"],
                    config,
                    datetime.now().strftime("%Y%m%d%H%M%S"),
                    duration_minutes,
                    len(read_problems()),
                    pass_score,
                    samples_file,
                    summary_file
                )
            else:
                logger.error(f"Evaluation failed for config {i}: Model {model_index} ({model_name}), Config {config_index} ({config_desc})")
                
        except Exception as e:
            logger.error(f"Error in evaluation {i}: {str(e)}", exc_info=True)
            # Clean up in case of exception
            torch.cuda.empty_cache()
            gc.collect()

    logger.info("All evaluations completed")

if __name__ == "__main__":
    main()