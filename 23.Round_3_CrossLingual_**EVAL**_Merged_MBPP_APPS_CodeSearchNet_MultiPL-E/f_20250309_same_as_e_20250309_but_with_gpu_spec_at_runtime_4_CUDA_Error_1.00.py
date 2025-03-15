#!/usr/bin/env python3
# Purpose: Automate MultiPL-E evaluation for LoRA merged models
# Usage examples:
# python f_20250309_same_as_e_20250309_but_with_gpu_spec_at_runtime_4_CUDA_Error_1.00.py --lang java
# python f_20250309_same_as_e_20250309_but_with_gpu_spec_at_runtime_4_CUDA_Error_1.00.py --lang java --index 1 2
# python f_20250309_same_as_e_20250309_but_with_gpu_spec_at_runtime_4_CUDA_Error_1.00.py --lang java --index 0
# python f_20250309_same_as_e_20250309_but_with_gpu_spec_at_runtime_4_CUDA_Error_1.00.py --lang java --gpu 2

# # Use GPU 1 (which is currently free)
# python f_20250309_same_as_e_20250309_but_with_gpu_spec_at_runtime_4_CUDA_Error_1.00.py --lang java --index 3 4 --gpu 1

# # Use GPU 2 (also free)
# python f_20250309_same_as_e_20250309_but_with_gpu_spec_at_runtime_4_CUDA_Error_1.00.py --lang java --index 5 6 --gpu 2

# # Use GPU 3 (also free)
# python f_20250309_same_as_e_20250309_but_with_gpu_spec_at_runtime_4_CUDA_Error_1.00.py --lang java --index 7 8 --gpu 3


import os
import subprocess
import time
import argparse
import csv
from pathlib import Path
from datetime import datetime
import shutil
import logging

# Configuration
OUTPUT_BASE_FOLDER = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/23.CrossLingual_**EVAL**MultiPL-E/03.Step3.PostFinetune_Eval_Java"
MULTIPL_E_FOLDER = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/09.Datasets/06.MultiPL-E_GitHub/MultiPL-E"
TRACKING_FILE = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/07.Tracking/23.Round_3_Merged_EVAL_MultiPL-E_Tracking.csv"
BACKUP_DIR = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/07.Tracking/23.Round_3_Merged_EVAL_MultiPL-E_Backup"
LOG_DIR = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/23.CrossLingual_**EVAL**MultiPL-E/logs"

# Hardcoded list of model paths
MODEL_PATHS = [
    "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/08.Models/22.CrossLingual_**TRAINING**_MBPP_Merged/20250309_nvf_alpha16_dropout0.05_r4_lr0.0002_epochs3_merged", #0
    "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/08.Models/22.CrossLingual_**TRAINING**_MBPP_Merged/20250309_nkr_alpha32_dropout0.05_r8_lr0.0002_epochs3_merged", #1
    "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/08.Models/22.CrossLingual_**TRAINING**_MBPP_Merged/20250309_nkr_alpha16_dropout0.05_r16_lr0.0002_epochs3_merged", #2
    "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/08.Models/22.CrossLingual_**TRAINING**_MBPP_Merged/20250309_nkr_alpha16_dropout0.05_r8_lr0.0005_epochs3_merged", #3
    "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/08.Models/22.CrossLingual_**TRAINING**_MBPP_Merged/20250309_nkr_alpha16_dropout0.05_r8_lr0.0002_epochs3_merged", #4
    "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/08.Models/22.CrossLingual_**TRAINING**_MBPP_Merged/20250309_nkr_alpha16_dropout0.05_r8_lr0.0001_epochs3_merged", #5
    "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/08.Models/22.CrossLingual_**TRAINING**_MBPP_Merged/20250309_nkr_alpha16_dropout0.1_r8_lr0.0002_epochs3_merged", #6
    "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/08.Models/22.CrossLingual_**TRAINING**_MBPP_Merged/20250309_nkr_alpha16_dropout0.0_r8_lr0.0002_epochs3_merged", #7
    "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/08.Models/22.CrossLingual_**TRAINING**_MBPP_Merged/20250309_nkr_alpha8_dropout0.05_r8_lr0.0002_epochs3_merged" #8
]


def setup_logging():
    """Set up logging with both file and console handlers"""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = os.path.join(LOG_DIR, f"{timestamp}_merged_eval.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__), timestamp

def backup_tracking_file():
    """Backup tracking CSV file to backup directory with timestamp"""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_file = os.path.join(BACKUP_DIR, f"{timestamp}_merged_tracking_backup.csv")
    
    if os.path.exists(TRACKING_FILE):
        shutil.copy2(TRACKING_FILE, backup_file)
        logging.info(f"Backed up tracking file to {backup_file}")

def update_tracking_file(base_model, dataset, metric, temperature, duration_minutes, 
                         num_problems, pass_score, samples_file, summary_file, model_desc):
    """Update tracking CSV file with details of the current evaluation run"""
    # Create backup of existing tracking file
    backup_tracking_file()
    
    # Create the tracking directory if it doesn't exist
    os.makedirs(os.path.dirname(TRACKING_FILE), exist_ok=True)
    
    # Extract timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Define field names for the CSV file
    field_names = [
        "timestamp", "base_model", "dataset", "metric", 
        "temperature", "num_beams", "top_p", "do_sample", "max_new_tokens",
        "num_problems", "score", "duration_minutes", "samples_file", "summary_file",
        "config_description"
    ]
    
    # Prepare data for the new row
    row_data = {
        "timestamp": timestamp,
        "base_model": base_model,
        "dataset": dataset,
        "metric": metric,
        "temperature": temperature,
        "num_beams": 8,
        "top_p": 0.95,
        "do_sample": True,
        "max_new_tokens": 512,
        "num_problems": num_problems,
        "score": f"{pass_score:.2f}",
        "duration_minutes": f"{duration_minutes:.2f}",
        "samples_file": os.path.basename(samples_file),
        "summary_file": os.path.basename(summary_file),
        "config_description": model_desc
    }
    
    # Check if file exists to determine if header should be written
    file_exists = os.path.isfile(TRACKING_FILE)
    
    # Write to CSV
    with open(TRACKING_FILE, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write data row
        writer.writerow(row_data)
    
    logging.info(f"Updated tracking file at {TRACKING_FILE}")

def run_command(cmd, cwd=None, env=None):
    """Run a command and print output in real-time."""
    logging.info(f"Running command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, 
                             stderr=subprocess.STDOUT, text=True, env=env)
    
    output = ""
    for line in process.stdout:
        print(line.strip())
        output += line
    
    process.wait()
    if process.returncode != 0:
        logging.error(f"Command failed with exit code {process.returncode}")
        return None
    return output

def run_evaluation(model_path, lang="java", gpu_index=1):
    """Run evaluation for a specific model and language"""
    if not os.path.exists(model_path):
        logging.error(f"Model path does not exist: {model_path}")
        return
    
    # Extract a short name for output folders from the model path
    model_folder = os.path.basename(model_path)
    model_short_name = "_".join(model_folder.split("_")[:3])
    
    logging.info(f"\n{'='*80}\nEvaluating model: {model_folder} on language: {lang} using GPU {gpu_index}\n{'='*80}")
    
    # Create output directory for this model
    output_dir = os.path.join(OUTPUT_BASE_FOLDER, f"{model_short_name}_{lang}_merged")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Created output directory: {output_dir}")
    
    # Set permissions on the directory
    os.system(f"chmod -R 777 {output_dir}")
    
    # Set the start time
    start_time = time.time()

    # Set environment variables for GPU selection
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    logging.info(f"Using GPU {gpu_index} for evaluation")

    # Step 1: Generate completions
    completion_cmd = [
        "python", "automodel_vllm_20250308_01.py",
        "--name", model_path,
        "--root-dataset", "humaneval",
        "--lang", lang,
        "--temperature", "0.2", 
        "--batch-size", "4",  # Very small batch size
        "--completion-limit", "10",  # Reduced from 20
        "--output-dir-prefix", output_dir,
        "--local-path",
        "--num-gpus", "1"  # Use just one GPU
    ]
    
    logging.info("\nGenerating completions...")
    output = run_command(completion_cmd, cwd=MULTIPL_E_FOLDER, env=env)
    if not output:
        logging.error(f"Skipping evaluation for {model_folder} due to generation failure")
        return
    
    # Step 2: Find the result directory
    result_dirs = list(Path(output_dir).glob("humaneval-*"))
    if not result_dirs:
        logging.error(f"No result directories found in {output_dir}")
        return
    
    result_dir = str(result_dirs[0])
    logging.info(f"\nFound result directory: {result_dir}")
    
    # Apply permissions to the results directory
    os.system(f"chmod -R 777 {result_dir}")
    
    # Step 3: Run evaluation with podman
    eval_cmd = [
        "podman", "run", "--rm", "--network", "none",
        "-v", f"{result_dir}:{result_dir}:rw",
        "multipl-e-eval",
        "--dir", result_dir,
        "--output-dir", result_dir,
        "--recursive"
    ]
    
    logging.info("\nRunning evaluation in container...")
    output = run_command(eval_cmd)
    if not output:
        logging.error(f"Evaluation failed for {model_folder}")
        return
    
    # Step 4: Calculate pass rates
    pass_k_cmd = ["python3", "pass_k.py", result_dir]
    logging.info("\nCalculating pass@k metrics...")
    output = run_command(pass_k_cmd, cwd=MULTIPL_E_FOLDER)
    if not output:
        logging.error(f"Error calculating pass rates for {model_folder}")
        return

    # Parse the pass rate from output
    pass_score = 0.0
    num_problems = 0
    try:
        lines = output.strip().split("\n")
        for line in lines:
            if line.startswith("Dataset,Pass@k,"):
                # Skip the header line
                continue
            if "," in line:  # This is a data line in CSV format
                parts = line.strip().split(',')
                if len(parts) >= 3 and parts[1] == "1":  # Check if this is the pass@1 result
                    pass_score = float(parts[2]) * 100  # Convert from decimal to percentage
                if len(parts) >= 4:
                    num_problems = int(parts[3])
    except Exception as e:
        logging.error(f"Error parsing pass rate: {str(e)}")
    
    # Calculate duration
    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    
    # Save results to a summary file
    summary_file = os.path.join(output_dir, f"{model_short_name}_{lang}_merged_results_summary.txt")
    samples_file = os.path.join(result_dir, "samples.jsonl") if os.path.exists(os.path.join(result_dir, "samples.jsonl")) else "unknown"
    
    with open(summary_file, 'w') as f:
        f.write(f"Model: {model_folder}\n")
        f.write(f"Language: {lang}\n")
        f.write(f"GPU Used: {gpu_index}\n")  # Added to track which GPU was used
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration_minutes:.2f} minutes\n")
        f.write(f"Number of problems: {num_problems}\n")
        f.write(f"Pass@1 Score: {pass_score:.2f}%\n")
        f.write(f"Raw output:\n{output}")
    
    # Set permissions on the summary file
    os.system(f"chmod 777 {summary_file}")
    
    # Update tracking file
    update_tracking_file(
        base_model=model_path,
        dataset=f"humaneval-{lang}",
        metric="pass@1",
        temperature=0.2,
        duration_minutes=duration_minutes,
        num_problems=num_problems,
        pass_score=pass_score,
        samples_file=samples_file,
        summary_file=summary_file,
        model_desc=f"LoRA merged model: {model_folder} (GPU {gpu_index})"
    )
    
    logging.info(f"\nCompleted evaluation for {model_folder} on {lang}")
    logging.info(f"Pass@1: {pass_score:.2f}%")
    logging.info(f"Duration: {duration_minutes:.2f} minutes")
    logging.info(f"Results saved to: {summary_file}")
    
    return pass_score, duration_minutes

def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA merged models on MultiPL-E HumanEval.")
    parser.add_argument("--lang", type=str, default="java",
                        help="Language to evaluate (java, py, cpp, cs, js)")
    parser.add_argument("--index", type=int, nargs="+", 
                        help="Specific indexes of models to process (0-based). If not provided, all models will be processed.")
    parser.add_argument("--gpu", type=int, default=1,
                        help="GPU index to use (0-3). Default is 1.")
    args = parser.parse_args()

    logger, _ = setup_logging()
    
    # Log the GPU being used
    logger.info(f"Using GPU {args.gpu} for evaluation")
    
    # Determine which models to process
    if args.index:
        # Validate indexes
        valid_indexes = []
        for idx in args.index:
            if idx < 0 or idx >= len(MODEL_PATHS):
                logger.warning(f"Index {idx} is out of range (0-{len(MODEL_PATHS)-1}). Skipping.")
            else:
                valid_indexes.append(idx)
        
        if not valid_indexes:
            logger.error("No valid indexes provided. Exiting.")
            return
        
        models_to_process = [MODEL_PATHS[idx] for idx in valid_indexes]
        logger.info(f"Processing {len(models_to_process)} models with indexes: {valid_indexes}")
    else:
        models_to_process = MODEL_PATHS
        logger.info(f"Processing all {len(models_to_process)} models")
    
    for i, model_path in enumerate(models_to_process):
        model_index = args.index[i] if args.index else i
        logger.info(f"\nProcessing model {i+1}/{len(models_to_process)} (index {model_index}): {os.path.basename(model_path)}")
        
        # Validate model path
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            continue
        
        try:
            pass_score, duration = run_evaluation(model_path, args.lang, args.gpu)
            if pass_score is not None:
                logger.info(f"Evaluation completed successfully with pass@1: {pass_score:.2f}%")
            else:
                logger.error(f"Evaluation failed for model: {os.path.basename(model_path)}")
        except Exception as e:
            logger.error(f"Error evaluating model {os.path.basename(model_path)}: {str(e)}", exc_info=True)
        
        # Add a delay between evaluations if there are more models to process
        if i < len(models_to_process) - 1:
            delay_time = 10  # seconds
            logger.info(f"Waiting {delay_time} seconds before next evaluation...")
            time.sleep(delay_time)
    
    logger.info(f"All evaluations completed for {len(models_to_process)} models")

if __name__ == "__main__":
    main()