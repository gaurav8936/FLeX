#!/usr/bin/env python3
# LoRA Training with Enhanced Metrics for APPS Dataset
# Usage: python a_20250303_LoRA_APPS_Training.py --optimizer adamw

import torch
import torch.cuda
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from peft import PeftModel, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from tqdm import tqdm
import os
import sys
import warnings
import time
import json
from datetime import datetime
import logging
import csv
import shutil
import pandas as pd
import numpy as np
import psutil
import argparse
import random
import string
import gc
from b_20250305_sophia_optimizer_v100 import SophiaG

# Custom APPS Dataset class
class APPSDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Prepare the full text (prompt + solution)
        full_text = f"### Problem:\n{example['prompt']}\n\n### Solution:\n{example['completion']}"
        
        # Tokenize
        tokenized = self.tokenizer(full_text, 
                                  return_tensors="pt", 
                                  truncation=True, 
                                  max_length=self.max_length,
                                  padding="max_length")
        
        # Create input_ids and labels
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        
        # For causal LM, labels are the same as input_ids
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def setup_logging(log_dir="01.Logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = os.path.join(log_dir, f"{timestamp}_apps_lora_training.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__), timestamp

def process_apps_dataset(dataset, split="train", difficulty=None, max_samples=None):
    """
    Process APPS dataset into prompt-completion format
    
    Args:
        dataset: The HuggingFace dataset containing APPS problems
        split: Dataset split to use ('train' or 'test')
        difficulty: Optional filter for difficulty level ('introductory', 'interview', or 'competition')
        max_samples: Optional limit on number of samples to process
    
    Returns:
        List of dictionaries with 'prompt' and 'completion' keys
    """
    processed_data = []
    
    # Get the dataset for the specified split
    split_data = dataset[split]
    
    # Filter by difficulty if specified
    if difficulty is not None:
        split_data = split_data.filter(lambda x: x["difficulty"] == difficulty)
        print(f"Filtered to {len(split_data)} examples with difficulty '{difficulty}'")
    
    # Limit to max_samples if specified
    if max_samples is not None:
        sample_count = min(max_samples, len(split_data))
        split_data = split_data.select(range(sample_count))
        print(f"Limited to first {sample_count} examples")
    
    for example in tqdm(split_data, desc=f"Processing {split} data"):
        # Get the problem and the first solution
        if example["solutions"] and len(example["solutions"]) > 0:
            prompt = example["question"]
            
            # Add starter code if it exists
            if example["starter_code"] and example["starter_code"] != "None":
                prompt += f"\n\nHere is some starter code:\n{example['starter_code']}"
            
            completion = example["solutions"][0]
            
            processed_data.append({
                "prompt": prompt,
                "completion": completion
            })
    
    print(f"Successfully processed {len(processed_data)} examples")
    return processed_data

def setup_metrics_tracking(log_dir, model_name, optimizer_name, model_dir_name, target_losses=[2.5, 2.3, 2.1]):
    """Setup files for tracking training metrics"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    metrics = {
        "model_name": model_name,
        "optimizer": optimizer_name,
        "timestamp": timestamp,
        "model_dir": model_dir_name,  # Added model directory
        "loss_log": [],
        "lr_log": [],
        "gradient_norm_log": [],
        "step_time_log": [],
        "memory_usage_log": [],
        "steps_to_target_loss": {str(loss): None for loss in target_losses},
        "target_losses": target_losses,
        "clipped_updates_log": [],  # For Sophia only
        "total_training_time": None,
        "peak_memory_usage": None,
        "avg_step_time": None,
        "best_validation_loss": None,
    }
    
    # Save initial metrics
    metrics_file = os.path.join(log_dir, f"{optimizer_name}_{timestamp}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Also create a CSV for easier analysis
    csv_file = os.path.join(log_dir, f"{optimizer_name}_{timestamp}_loss_log.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'loss', 'perplexity', 'loss_variance', 'lr', 'gradient_norm', 'step_time', 'memory_usage'])
    
    return metrics, metrics_file, csv_file

def update_metrics(metrics, metrics_file, csv_file, step, loss, lr, gradient_norm, step_time, memory_usage, 
                  clipped_updates_fraction=None):
    """Update metrics during training"""
    # Update loss log
    metrics["loss_log"].append({"step": step, "loss": loss})
    metrics["lr_log"].append({"step": step, "lr": lr})
    metrics["gradient_norm_log"].append({"step": step, "gradient_norm": gradient_norm})
    metrics["step_time_log"].append({"step": step, "time": step_time})
    metrics["memory_usage_log"].append({"step": step, "memory": memory_usage})
    
    # Only add clipping data if it's provided (for Sophia)
    if clipped_updates_fraction is not None:
        if "clipped_updates_log" not in metrics:
            metrics["clipped_updates_log"] = []
        metrics["clipped_updates_log"].append({"step": step, "fraction": clipped_updates_fraction})
    
    # Check if we've reached any target loss thresholds
    for target_loss in metrics["target_losses"]:
        if loss <= target_loss and metrics["steps_to_target_loss"][str(target_loss)] is None:
            metrics["steps_to_target_loss"][str(target_loss)] = step

    # Add sliding window loss variance (last 5 steps)
    if len(metrics["loss_log"]) >= 5:
        recent_losses = [entry["loss"] for entry in metrics["loss_log"][-5:]]
        loss_variance = np.var(recent_losses)
        if "loss_variance_log" not in metrics:
            metrics["loss_variance_log"] = []
        metrics["loss_variance_log"].append({"step": step, "variance": loss_variance})
    
    # Add perplexity (if loss is NLL/cross-entropy)
    perplexity = np.exp(loss)
    if "perplexity_log" not in metrics:
        metrics["perplexity_log"] = []
    metrics["perplexity_log"].append({"step": step, "perplexity": perplexity})
    
    # Update CSV with new metrics
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([step, loss, perplexity, loss_variance if "loss_variance_log" in metrics else "N/A", 
                        lr, gradient_norm, step_time, memory_usage])
    
    # Save updated metrics periodically
    if step % 50 == 0:
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return metrics

def finalize_metrics(metrics, metrics_file, total_time, peak_memory, avg_step_time=None, best_val_loss=None):
    """Finalize metrics at the end of training"""
    metrics["total_training_time"] = total_time
    metrics["peak_memory_usage"] = peak_memory
    
    if avg_step_time is not None:
        metrics["avg_step_time"] = avg_step_time
    else:
        # Calculate average step time from log
        step_times = [entry["time"] for entry in metrics["step_time_log"]]
        metrics["avg_step_time"] = sum(step_times) / len(step_times) if step_times else None
    
    if best_val_loss is not None:
        metrics["best_validation_loss"] = best_val_loss
    
    # Save final metrics
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def compute_gradient_norm(model):
    """Compute L2 norm of gradients"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def estimate_lora_hessian_from_ids(model, batch, lora_layers_only=True):
    """
    Modified version of estimate_lora_hessian that works with preprocessed input_ids
    
    Arguments:
        model: The model with LoRA layers
        batch: Batch with input_ids and attention_mask
        lora_layers_only: If True, only compute for LoRA parameters
        
    Returns:
        Dictionary of Hessian estimates for each parameter
    """
    # Make sure we're in training mode
    training = model.training
    model.train()  # Set to training mode to ensure requires_grad works
    
    # Forward pass to get logits
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits
        
        # Sample from the model's distribution
        probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        sampled_next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    # Zero gradients
    model.zero_grad()
    
    # Compute loss with sampled tokens - now OUTSIDE torch.no_grad()
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = torch.cat([batch['input_ids'][:, 1:], sampled_next_tokens.unsqueeze(-1)], dim=1)[:, :-1].contiguous()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    
    # Detach logits and then require gradients (this is key to fix the issue)
    shift_logits = shift_logits.detach().requires_grad_(True)
    
    # Use reshape instead of view for better compatibility
    loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
    
    # Compute gradients
    loss.backward()
    
    # Get Hessian estimates
    hessian_estimates = {}
    batch_size = batch['input_ids'].size(0)
    
    for name, param in model.named_parameters():
        # Only compute for LoRA parameters if specified
        if lora_layers_only and 'lora' not in name:
            continue
            
        if hasattr(param, '_optim_id') and param.grad is not None:
            # GNB Hessian estimate: B * (grad)^2
            hessian_estimates[param._optim_id] = batch_size * (param.grad.data ** 2)
        elif hasattr(param, '_optim_id'):
            # If parameter has no gradient, use a small positive value
            hessian_estimates[param._optim_id] = torch.ones_like(param.data) * 1e-4
    
    # Restore original training mode
    model.train(training)
    
    return hessian_estimates
    
def train_lora_on_apps(optimizer_type="adamw", base_model="codellama/CodeLlama-7b-hf", 
                       lora_r=8, lora_alpha=16, lora_dropout=0.05, 
                       learning_rate=2e-4, num_epochs=3, batch_size=4,
                       max_length=512, gradient_accumulation_steps=4,
                       # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                       target_modules=["q_proj", "v_proj", "down_proj", "up_proj"],
                       apps_data_path="/home/ubuntu/01.Stanford/01.CS224N/01.Project/09.Datasets/03.APPS/APPS_HF",
                       logging_steps=10):
    """
    Train a LoRA model on the APPS dataset with enhanced metrics tracking
    """
    # Generate unique model identifier
    model_id = ''.join(random.choices(string.ascii_lowercase, k=3))
    
    # Setup logging
    logger, timestamp = setup_logging()
    logger.info(f"Starting LoRA training on APPS with {optimizer_type} optimizer")
    logger.info(f"Model: {base_model}, LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    # Create model directory
    timestamp_prefix = datetime.now().strftime("%Y%m%d")
    model_dir_name = (
        f"{timestamp_prefix}_{model_id}_"
        f"MBPP2APPS_{optimizer_type}_"
        f"r{lora_r}_"
        f"lr{learning_rate}_"
        f"epochs{num_epochs}"
    )
    # model_save_path = os.path.join("/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/08.Models/11.Round_2_Sophia_Adam_**TRAINING**_APPS/", model_dir_name)
    # Modify the model_save_path to point to your destination folder
    model_save_path = os.path.join("/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/08.Models/11.Round_2_Adam_**TRAINING**_only_for_23CrossLingual_unmerged", model_dir_name)
    os.makedirs(model_save_path, exist_ok=True)
    
    # Setup metrics tracking
    metrics_dir = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/11.Round_2_Sophia_Adam_**TRAINING**_APPS/04.json_csv_logs_metrics_all_for_23CrossLingual"
    
    os.makedirs(metrics_dir, exist_ok=True)
    metrics, metrics_file, csv_file = setup_metrics_tracking(
        metrics_dir, 
        base_model, 
        optimizer_type,
        model_dir_name,  # Pass the model directory name
        target_losses=[3.0, 2.8, 2.6, 2.4, 2.2, 2.0]  # Custom loss thresholds for APPS
    )
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ########################## Use Base Model instead of Round 1 Model ###################
    # # Load base model (using the merged round 1 model instead of base + LoRA)
    # logger.info("Loading base model (merged from Round 1)...")
    # model = AutoModelForCausalLM.from_pretrained(
    #     "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/08.Models/04.Round_1_LoRA_MBPP_Model_**TRAINING**/20250301_03_alpha16_dropout0.05_r8_lr0.0002_epochs3_merged_round1_GOLD",
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )

    # Load base model (using base CodeLlama-7b-hf instead of merged Round 1 model)
    logger.info("Loading base model (CodeLlama-7b-hf)...")
    model = AutoModelForCausalLM.from_pretrained(
        "codellama/CodeLlama-7b-hf",  # Changed to base model
        torch_dtype=torch.float16,
        device_map="auto",
    )
    ###############################################################################################
    
    # Apply new LoRA for continued training
    logger.info("Configuring new LoRA layer for continued training...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)    
    
    logger.info("Trainable parameters:")
    model.print_trainable_parameters()
    
    # Load the APPS dataset
    logger.info("Loading APPS dataset...")
    try:
        dataset = load_dataset("parquet", data_files={
            "train": os.path.join(apps_data_path, "train.parquet"),
            "test": os.path.join(apps_data_path, "test.parquet"),
        })
        
        logger.info(f"Loaded {len(dataset['train'])} training and {len(dataset['test'])} testing examples")
    except Exception as e:
        logger.error(f"Failed to load APPS dataset: {str(e)}")
        raise
    
    # Process the dataset
    logger.info("Processing APPS dataset...")
    
    # # For 10 competition-level samples:
    # train_data = process_apps_dataset(dataset, "train", difficulty="competition", max_samples=10)
    
    # For 100 competition-level samples:
    # train_data = process_apps_dataset(dataset, "train", difficulty="competition", max_samples=100)

    # For all competition-level data
    train_data = process_apps_dataset(dataset, "train", difficulty="competition")

    # train_data = process_apps_dataset(dataset, "train", difficulty="interview")
    #              train  test  total
    # interview      2000  3000   5000
    # competition     361  1000   1361
    # introductory   2639  1000   3639
    # total          5000  5000  10000    
    
    logger.info(f"Processed {len(train_data)} training examples")
    
    # Create validation split
    if len(train_data) <= 20:  # For very small datasets
        # For tiny datasets, use 80/20 split but ensure at least 1 validation sample
        val_size = max(1, int(0.2 * len(train_data)))
        val_indices = random.sample(range(len(train_data)), val_size)
    else:  # For larger datasets
        # Use 10% for validation with a cap of 100 examples
        val_size = min(100, int(0.1 * len(train_data)))
        val_indices = random.sample(range(len(train_data)), val_size)
    
    # Log the split information
    logger.info(f"Using {val_size} examples for validation ({val_size/len(train_data):.1%} of data)")

    val_data = [train_data[i] for i in val_indices]
    # Remove validation examples from training
    train_data = [train_data[i] for i in range(len(train_data)) if i not in val_indices]
    
    logger.info(f"Split into {len(train_data)} training and {len(val_data)} validation examples")
    
    # Create datasets
    train_dataset = APPSDataset(train_data, tokenizer, max_length=max_length)
    val_dataset = APPSDataset(val_data, tokenizer, max_length=max_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    
    # Setup optimizer
    logger.info(f"Setting up {optimizer_type} optimizer...")
    
    if optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

    elif optimizer_type.lower() == "sophia":
        if SophiaG is None:
            logger.error("Sophia optimizer not found. Please ensure sophia_optimizer.py is installed.")
            raise ImportError("Could not import SophiaG.")

        optimizer = SophiaG(
            model.parameters(),
            lr=learning_rate * 0.8,
            betas=(0.965, 0.99),
            eps=1e-12,
            weight_decay=0.01,
            k=10,
            gamma=0.01,
        )
        optimizer.add_param_optimizer_id()

    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    # Setup LR scheduler
    num_training_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps,
        eta_min=1e-5,
    )
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    global_step = 0
    best_val_loss = float('inf')
    peak_memory = 0
    steps_since_last_log = 0
    total_loss = 0
    
    # Set model to training mode
    model.train()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            step_start_time = time.time()
            steps_since_last_log += 1
            
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Only perform optimization step and metrics logging every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
                # Calculate gradient norm before optimization step
                grad_norm = compute_gradient_norm(model)

                ######################### Replaced with this ################

                # For Sophia: prepare hessian estimates or fallback
                gnb_kwargs = None
                clipped_updates_fraction = None
                hessian_avg_magnitude = None
                
                if optimizer_type.lower() == "sophia" and (global_step % optimizer.defaults["k"] == 0 or global_step == 0):
                    try:
                        # Generate Hessian estimates for this batch
                        hessian_start_time = time.time()
                        
                        # Get a subset of the batch for Hessian estimation
                        hessian_batch = {k: v[:min(len(v), 32)] for k, v in batch.items()}
                        
                        # Use the adapted helper function for Hessian estimation
                        hessian_estimates = estimate_lora_hessian_from_ids(
                            model=model,
                            batch=hessian_batch,
                            lora_layers_only=True
                        )
                        
                        # Calculate average Hessian diagonal magnitude for tracking
                        if len(hessian_estimates) > 0:
                            hessian_sum = 0.0
                            hessian_count = 0
                            for h_id, h_val in hessian_estimates.items():
                                hessian_sum += torch.mean(h_val).item()
                                hessian_count += 1
                            
                            if hessian_count > 0:
                                hessian_avg_magnitude = hessian_sum / hessian_count
                        
                        # Recompute loss and backward for actual update
                        optimizer.zero_grad()
                        outputs = model(**batch)
                        loss = outputs.loss / gradient_accumulation_steps
                        loss.backward()
                        
                        # Create gnb_kwargs with the estimates
                        gnb_kwargs = {"hessian_estimates": hessian_estimates}
                        
                        hessian_time = time.time() - hessian_start_time
                        logger.info(f"Hessian estimation took {hessian_time:.2f}s")
                        
                    except Exception as e:
                        logger.warning(f"Hessian estimation failed: {str(e)}. Using fallback values.")
                        # Create fallback with constant values
                        gnb_kwargs = {"hessian_estimates": {}}
                        for name, param in model.named_parameters():
                            if hasattr(param, '_optim_id'):
                                gnb_kwargs["hessian_estimates"][param._optim_id] = torch.ones_like(param.data) * 1e-4
                
                # Optimizer step
                if optimizer_type.lower() == "sophia":
                    optimizer.step(gnb_kwargs=gnb_kwargs)
                    # Get clipping statistics after step
                    clipped_updates_fraction = optimizer.get_clipping_stats()
                else:
                    optimizer.step()
                
                # Add Hessian metrics to JSON if available
                if hessian_avg_magnitude is not None:
                    if "hessian_diag_log" not in metrics:
                        metrics["hessian_diag_log"] = []
                    metrics["hessian_diag_log"].append({"step": global_step, "hessian_diag": hessian_avg_magnitude})

                # Calculate the current learning rate
                current_lr = scheduler.get_last_lr()[0]
                
                # Get memory usage with proper GPU tracking
                if torch.cuda.is_available():
                    # Get the current and peak GPU memory usage in MB
                    memory_usage = torch.cuda.memory_allocated() / (1024 * 1024)
                    peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
                else:
                    # Fall back to system memory tracking if GPU isn't available
                    memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
                    peak_gpu_memory = memory_usage                
                
                # Update the peak memory with GPU peak memory
                peak_memory = max(peak_memory, peak_gpu_memory)
                
                # Calculate step time
                step_time = time.time() - step_start_time
                
                # Update metrics with the loss value directly
                metrics = update_metrics(
                    metrics, 
                    metrics_file, 
                    csv_file, 
                    global_step, 
                    loss.item() * gradient_accumulation_steps,  # Convert back to original scale
                    current_lr, 
                    grad_norm, 
                    step_time, 
                    memory_usage,
                    clipped_updates_fraction
                )                
                
                # Reset counters
                steps_since_last_log = 0
                total_loss = 0
                ####################### Added this line to increment to counter #############
                # Add this line to increment the step counter
                global_step += 1
                #####################################################################
                
        # End of epoch
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_batch in tqdm(val_dataloader, desc="Validating"):
                val_batch = {k: v.to(model.device) for k, v in val_batch.items()}
                val_outputs = model(**val_batch)
                val_loss += val_outputs.loss.item()
        
        val_loss /= len(val_dataloader)
        logger.info(f"Validation loss: {val_loss:.6f}")
        
        # Log validation loss
        with open(os.path.join(metrics_dir, f"{optimizer_type}_{timestamp}_val_loss.csv"), 'a', newline='') as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(['epoch', 'val_loss'])
            writer.writerow([epoch+1, val_loss])
        
        # Save checkpoint if it's the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"New best validation loss: {best_val_loss:.6f}")
            model.save_pretrained(os.path.join(model_save_path, "best_checkpoint"))
            tokenizer.save_pretrained(os.path.join(model_save_path, "best_checkpoint", "tokenizer"))
        
        # Save regular checkpoint
        model.save_pretrained(os.path.join(model_save_path, f"checkpoint-epoch-{epoch+1}"))
        
        # Back to training mode
        model.train()
    
    # End of training
    training_time = (time.time() - start_time) / 60  # in minutes
    logger.info(f"Training completed in {training_time:.2f} minutes")
    
    # Save final model
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(os.path.join(model_save_path, "tokenizer"))


    ############## Revised this to not merge but keep separate ##################    
    # # Merge LoRA weights with base model
    # logger.info("Merging LoRA weights into base model...")
    # merged_model = model.merge_and_unload()
    
    # # Create merged model directory
    # merged_model_path = os.path.join(
    #     os.path.dirname(model_save_path), 
    #     f"{model_dir_name}_merged_round2"
    # )
    # os.makedirs(merged_model_path, exist_ok=True)
    
    # # Save the merged model
    # logger.info(f"Saving merged model to {merged_model_path}...")
    # merged_model.save_pretrained(merged_model_path)
    # tokenizer.save_pretrained(os.path.join(merged_model_path, "tokenizer"))
    # logger.info(f"Merged model saved to {merged_model_path}")

    ############## Revised this to not merge but keep separate ##################
    # Just save the final LoRA weights
    logger.info(f"Saving final LoRA weights to {model_save_path}...")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(os.path.join(model_save_path, "tokenizer"))
    logger.info(f"LoRA weights saved to {model_save_path}")
    ####################################################################################
    
    # Finalize metrics
    finalize_metrics(metrics, metrics_file, training_time, peak_memory, best_val_loss=best_val_loss)
    
    # Save final config
    config = {
        "model": base_model,
        "optimizer": optimizer_type,
        "lora": {
            "r": lora_r,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": target_modules
        },
        "training": {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "max_length": max_length,
            "gradient_accumulation_steps": gradient_accumulation_steps
        },
        "metrics": {
            "file": metrics_file,
            "training_time_minutes": training_time,
            "peak_memory_mb": peak_memory,
            "best_val_loss": best_val_loss
        }
    }


    ############# Revised this to not merge but keep separate ##################

    # # Write config to the merged model directory instead
    # with open(os.path.join(merged_model_path, "training_config.json"), 'w') as f:
    #     json.dump(config, f, indent=2)

    # Write config to the model directory (not merged directory)
    with open(os.path.join(model_save_path, "training_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    ##############################################################################
    
    # Update tracking file
    tracking_file = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/07.Tracking/11.Round_2_Adam_Only_**TRAINING**_APPS_for_23CrossLingual_Tracking.csv"
    backup_dir = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/07.Tracking/11.Round_2_Adam_Only_**TRAINING**_APPS_for_23CrossLingual_Tracking_Backup"
    
    field_names = [
        "timestamp", "model_dir", "base_model", "lora_r", "lora_alpha", 
        "lora_dropout", "lora_bias", "target_modules", "num_epochs", "batch_size", 
        "max_length", "gradient_clip", "learning_rate", "optimizer", "weight_decay",
        "scheduler", "dataset", "training_time_minutes", "best_val_loss"
    ]
    
    row_data = {
        "timestamp": timestamp,
        "model_dir": model_dir_name,
        "base_model": base_model,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_bias": "none",
        "target_modules": ",".join(target_modules),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "gradient_clip": 1.0,
        "learning_rate": learning_rate,
        "optimizer": optimizer_type,
        "weight_decay": 0.01,
        "scheduler": "cosine",
        "dataset": "apps",
        "training_time_minutes": f"{training_time:.2f}",
        "best_val_loss": f"{best_val_loss:.6f}"
    }
    
    # Check if file exists
    file_exists = os.path.isfile(tracking_file)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(tracking_file), exist_ok=True)
    
    # Backup existing file if it exists
    if file_exists:
        backup_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_file = os.path.join(backup_dir, f"{backup_timestamp}_11.Round_2_Sophia_Adam_**TRAINING**_APPS_Tracking_backup.csv")
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(tracking_file, backup_file)
    
    # Write to CSV
    with open(tracking_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write data row
        writer.writerow(row_data)
    
    logger.info(f"Updated tracking file at {tracking_file}")
    logger.info(f"Training complete! Model saved to {model_save_path}")
    
    return model_save_path, metrics

def main():
    parser = argparse.ArgumentParser(description="Train LoRA on APPS dataset with enhanced metrics tracking")
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sophia"], default="adamw",
                      help="Optimizer to use (adamw or sophia)")
    parser.add_argument("--base_model", type=str, default="codellama/CodeLlama-7b-hf",
                      help="Base model to fine-tune")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                      help="Number of steps to accumulate gradients")
    parser.add_argument("--apps_data_path", type=str, 
                      default="/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/09.Datasets/03.APPS/APPS_HF",
                      help="Path to APPS dataset parquet files")
    parser.add_argument("--logging_steps", type=int, default=10, 
                      help="Number of steps between logging metrics")
    args = parser.parse_args()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Free memory
    gc.collect()
    
    # Train with the specified optimizer
    model_path, metrics = train_lora_on_apps(
        optimizer_type=args.optimizer,
        base_model=args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        apps_data_path=args.apps_data_path,
        logging_steps=args.logging_steps,
    )
    
    print(f"Model saved to {model_path}")
    print(f"Training metrics saved to {metrics}")

if __name__ == "__main__":
    main()

