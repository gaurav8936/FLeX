########## format to run this ########################
######### python lora_training_simplified.py --config_range "1:2" ########

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
from tqdm import tqdm
import os
import time
from datetime import datetime
import logging
import csv
import shutil
import argparse
import random, string
from peft import PeftModel

# --------------------------------------
# CONFIGURABLE PATHS AND SETTINGS
# --------------------------------------
# Base directories
BASE_MODELS_DIR = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/08.Models"
BASE_TRACKING_DIR = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/07.Tracking"

# Specific directories
LORA_ADAPTERS_DIR = os.path.join(BASE_MODELS_DIR, "22.Round_3_CrossLingual_**TRAINING**_MBPP_Lora_Adapters")
MERGED_MODELS_DIR = os.path.join(BASE_MODELS_DIR, "22.CrossLingual_**TRAINING**_MBPP_Merged")

# Tracking file - saved directly in BASE_TRACKING_DIR
TRACKING_FILE = os.path.join(BASE_TRACKING_DIR, "22.Round_3_CrossLingual_**TRAINING**_MBPP_Merged_Tracking.csv")
BACKUP_DIR = os.path.join(BASE_TRACKING_DIR, "22.Round_3_CrossLingual_**TRAINING**_MBPP_Merged_Backup")
LOG_DIR = os.path.join(BASE_TRACKING_DIR, "22.Round_3_CrossLingual_**TRAINING**_MBPP_Merged_Logs")

# Default model
DEFAULT_MODEL = "codellama/CodeLlama-7b-hf"

# Dataset settings
DATASET_NAME = "mbpp"
DATASET_SPLIT = "train"

# Training settings
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_LENGTH = 512
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_GRADIENT_CLIP = 1.0

# LoRA settings
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]

# --------------------------------------
# END OF CONFIGURABLE SETTINGS
# --------------------------------------

# Generate a 3-character random string for model identifier
model_id = ''.join(random.choices(string.ascii_lowercase, k=3))

from b_20250309_LoRA_MBPP_Train_for_CrossLingual_Merged_Config_v100 import get_experiment_configurations

def setup_logging(log_dir=LOG_DIR):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = os.path.join(log_dir, f"{timestamp}_lora_training.log")

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
    backup_file = os.path.join(backup_dir, f"{timestamp}_tracking_backup.csv")
    
    if os.path.exists(tracking_file):
        shutil.copy2(tracking_file, backup_file)
        logging.info(f"Backed up tracking file to {backup_file}")

def update_tracking_file(tracking_file, backup_dir, config, model_dir, timestamp, training_time):
    """Update tracking CSV file with details of the current run"""
    # Create backup of existing tracking file
    backup_tracking_file(tracking_file, backup_dir)
    
    # Create the parent directory of the tracking file if it doesn't exist
    os.makedirs(os.path.dirname(tracking_file), exist_ok=True)
    
    # Extract key parameters to track
    model_name = config["model"]["name"]
    lora_r = config["lora"]["r"]
    lora_alpha = config["lora"]["alpha"]
    lora_dropout = config["lora"]["dropout"]
    lora_bias = config["lora"]["bias"]
    target_modules = ",".join(config["lora"]["target_modules"])
    num_epochs = config["training"]["num_epochs"]
    batch_size = config["training"]["batch_size"]
    max_length = config["training"]["max_length"]
    gradient_clip = config["training"]["gradient_clip"]
    learning_rate = config["training"]["optimizer"]["lr"]
    optimizer = config["training"]["optimizer"]["name"]
    weight_decay = config["training"]["optimizer"].get("weight_decay", 0)
    scheduler = config["training"].get("scheduler", {}).get("name", "none")
    dataset = config["training"]["dataset"]["name"]
    
    # Define field names for the CSV file
    field_names = [
        "timestamp", "model_dir", "base_model", "lora_r", "lora_alpha", 
        "lora_dropout", "lora_bias", "target_modules", "num_epochs", "batch_size", 
        "max_length", "gradient_clip", "learning_rate", "optimizer", "weight_decay",
        "scheduler", "dataset", "training_time_minutes"
    ]
    
    # Prepare data for the new row
    row_data = {
        "timestamp": timestamp,
        "model_dir": model_dir,
        "base_model": model_name,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_bias": lora_bias,
        "target_modules": target_modules,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "max_length": max_length,
        "gradient_clip": gradient_clip,
        "learning_rate": learning_rate,
        "optimizer": optimizer,
        "weight_decay": weight_decay,
        "scheduler": scheduler,
        "dataset": dataset,
        "training_time_minutes": f"{training_time:.2f}"
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

def merge_lora_into_base_model(base_model_path, lora_model_path, output_path):
    """Merge LoRA weights into the base model"""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading base model from {base_model_path}...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    logger.info(f"Loading LoRA adapter from {lora_model_path}...")
    # Load LoRA model
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    logger.info("Merging LoRA weights into base model...")
    # Merge LoRA weights with base model
    merged_model = model.merge_and_unload()
    
    logger.info(f"Saving merged model to {output_path}...")
    # Save the merged model
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(os.path.join(output_path, "tokenizer"))
    
    logger.info("Model merging complete!")
    return output_path

def train_lora_model(config, model_number=1):
    # Unpack config
    model_params = config["model"]
    training_params = config["training"]
    lora_params = config["lora"]
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Create model directory name based on key parameters
    model_dir_name = (
        f"{timestamp}_{model_id}_"
        f"alpha{lora_params['alpha']}_"
        f"dropout{lora_params['dropout']}_"
        f"r{lora_params['r']}_"
        f"lr{training_params['optimizer']['lr']}_"
        f"epochs{training_params['num_epochs']}"
    )
    
    # Create save paths for both LoRA adapter and merged model
    # Use the same name for LoRA adapter but in the LoRA directory
    merged_model_name = model_dir_name + "_merged"
    lora_save_path = os.path.join(LORA_ADAPTERS_DIR, model_dir_name + "_Lora_Adapters")
    merged_save_path = os.path.join(MERGED_MODELS_DIR, merged_model_name)
    
    # Update model save path in config
    model_params["save_path"] = lora_save_path
    
    # Setup logging
    logger, log_timestamp = setup_logging()
    logger.info(f"Starting LoRA training #{model_number} with parameters:")
    logger.info(f"Model: {model_params['name']}")
    logger.info(f"Save path: {model_params['save_path']}")
    logger.info(f"Training parameters: {training_params}")
    logger.info(f"LoRA parameters: {lora_params}")
    
    # Create output directories
    os.makedirs(model_params["save_path"], exist_ok=True)
    os.makedirs(os.path.dirname(merged_save_path), exist_ok=True)
    
    # ------------------------
    # Step 1: Load the Model & Tokenizer
    # ------------------------
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_params["name"])
    
    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model with specified parameters
    model_load_kwargs = {
        "torch_dtype": getattr(torch, model_params["dtype"]),
        "device_map": model_params["device_map"],
    }
    
    # Add optional parameters if specified
    if model_params.get("load_in_8bit", False):
        model_load_kwargs["load_in_8bit"] = True
    if model_params.get("load_in_4bit", False):
        model_load_kwargs["load_in_4bit"] = True
    
    model = AutoModelForCausalLM.from_pretrained(
        model_params["name"],
        **model_load_kwargs
    )
    
    # Prepare model for k-bit training if needed
    if model_params.get("load_in_8bit", False) or model_params.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)
    
    # ------------------------
    # Step 2: Configure LoRA
    # ------------------------
    logger.info("Applying LoRA configuration...")
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_params["r"],
        lora_alpha=lora_params["alpha"],
        lora_dropout=lora_params["dropout"],
        bias=lora_params["bias"],
        target_modules=lora_params["target_modules"],
        modules_to_save=lora_params.get("modules_to_save", None),
        fan_in_fan_out=lora_params.get("fan_in_fan_out", False),
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    logger.info("Trainable parameters:")
    model.print_trainable_parameters()
    
    # ------------------------
    # Step 3: Load and Prepare Data
    # ------------------------
    logger.info("Loading and preparing dataset...")
    
    # Load the dataset specified in config
    dataset_name = training_params["dataset"]["name"]
    dataset_split = training_params["dataset"]["split"]
    
    if dataset_name == "mbpp":
        dataset = load_dataset(dataset_name, split=dataset_split)
        logger.info(f"Loaded {len(dataset)} {dataset_name} problems.")
        
        # Format training data for MBPP
        training_data = []
        for item in dataset:
            # Prompt is the problem description
            prompt = item["text"] + "\n\n"
            
            # Add test cases as comments
            for test in item["test_list"]:
                prompt += f"# {test}\n"
            
            prompt += "\n"
            
            # Target completion
            completion = item["code"]
            
            training_data.append({
                "prompt": prompt,
                "completion": completion
            })
    else:
        # Handle other datasets - add more options as needed
        raise ValueError(f"Dataset {dataset_name} not supported yet")
    
    logger.info(f"Prepared {len(training_data)} training examples.")
    
    # ------------------------
    # Step 4: Training Loop
    # ------------------------
    # Optimizer setup
    optimizer_name = training_params["optimizer"]["name"]
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=training_params["optimizer"]["lr"],
            weight_decay=training_params["optimizer"].get("weight_decay", 0.01),
            betas=training_params["optimizer"].get("betas", (0.9, 0.999))
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_params["optimizer"]["lr"],
            betas=training_params["optimizer"].get("betas", (0.9, 0.999))
        )
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported")
    
    # Learning rate scheduler setup
    scheduler = None
    if training_params.get("scheduler", {}):
        scheduler_name = training_params["scheduler"]["name"]
        if scheduler_name == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=training_params["scheduler"].get("start_factor", 1.0),
                end_factor=training_params["scheduler"].get("end_factor", 0.1),
                total_iters=training_params["scheduler"].get("total_iters", 
                                                            training_params["num_epochs"] * len(training_data) // training_params["batch_size"])
            )
        elif scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=training_params["scheduler"].get("t_max", 
                                                     training_params["num_epochs"] * len(training_data) // training_params["batch_size"]),
                eta_min=training_params["scheduler"].get("eta_min", 0)
            )
    
    # Set model to training mode
    model.train()
    
    # Main training loop
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(training_params["num_epochs"]):
        total_loss = 0
        progress_bar = tqdm(training_data, desc=f"Epoch {epoch+1}/{training_params['num_epochs']}")
        
        for idx, item in enumerate(progress_bar):
            # Create input text by combining prompt and completion
            full_text = item["prompt"] + item["completion"]
            
            # Tokenize
            inputs = tokenizer(
                full_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=training_params["max_length"]
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Create labels (same as input_ids for causal LM)
            inputs["labels"] = inputs["input_ids"].clone()
            
            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Update weights every batch_size samples or at the end
            if (idx + 1) % training_params["batch_size"] == 0 or idx == len(training_data) - 1:
                # Gradient clipping if configured
                if training_params.get("gradient_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        training_params["gradient_clip"]
                    )
                    
                optimizer.step()
                optimizer.zero_grad()
                
                # Step the scheduler if it exists
                if scheduler:
                    scheduler.step()
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": total_loss / (idx + 1),
                    "lr": optimizer.param_groups[0]["lr"]
                })
        
        # End of epoch
        avg_loss = total_loss / len(training_data)
        logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.6f}")
        
        # Save checkpoint after each epoch or at specified interval
        if (epoch + 1) % training_params.get("save_every", 1) == 0:
            checkpoint_path = f"{model_params['save_path']}/checkpoint-epoch-{epoch+1}"
            model.save_pretrained(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Calculate training time
    training_time = (time.time() - start_time) / 60  # in minutes
    logger.info(f"Training completed in {training_time:.2f} minutes")
    
    # ------------------------
    # Step 5: Save the final model
    # ------------------------
    model.save_pretrained(model_params["save_path"])
    tokenizer.save_pretrained(f"{model_params['save_path']}/tokenizer")
    logger.info(f"Training complete! LoRA adapter saved to {model_params['save_path']}")
    
    # Merge LoRA into base model
    logger.info("Merging LoRA weights into base model...")
    merge_lora_into_base_model(model_params["name"], model_params["save_path"], merged_save_path)
    logger.info(f"Merged model saved to {merged_save_path}")
    
    # Save training configuration
    config_file = os.path.join(merged_save_path, "training_config.txt")
    with open(config_file, 'w') as f:
        f.write(f"Base Model: {model_params['name']}\n")
        f.write(f"Training Timestamp: {log_timestamp}\n")
        f.write(f"Training Duration: {training_time:.2f} minutes\n")
        f.write(f"Training Configuration:\n")
        # Write out the entire configuration
        for section, params in config.items():
            f.write(f"\n[{section}]\n")
            for k, v in params.items():
                f.write(f"  {k}: {v}\n")
    
    logger.info(f"Training configuration saved to {config_file}")
    
    # Update tracking file
    update_tracking_file(TRACKING_FILE, BACKUP_DIR, config, merged_model_name, log_timestamp, training_time)
    
    logger.info("Training complete!")
    
    return merged_save_path, merged_model_name, training_time

def main():
    parser = argparse.ArgumentParser(description="Specify the range of configurations to use.")
    parser.add_argument("--config_range", type=str, default=":1",
                        help="Specify the range of configurations to use, e.g., ':1', '1:2', '2:'")
    args = parser.parse_args()

    all_configs = get_experiment_configurations()

    # Convert string range into slice
    start, end = (int(x) if x else None for x in args.config_range.split(":"))
    configs = all_configs[start:end]

    # Create all necessary directories
    os.makedirs(LORA_ADAPTERS_DIR, exist_ok=True)
    os.makedirs(MERGED_MODELS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(TRACKING_FILE), exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    logger, _ = setup_logging()
    logger.info(f"Starting training loop for {len(configs)} configurations")
    logger.info(f"Using directories:")
    logger.info(f"  LoRA adapters: {LORA_ADAPTERS_DIR}")
    logger.info(f"  Merged models: {MERGED_MODELS_DIR}")
    logger.info(f"  Tracking file: {TRACKING_FILE}")
    logger.info(f"  Backup: {BACKUP_DIR}")
    logger.info(f"  Logs: {LOG_DIR}")

    for i, config in enumerate(configs, 1):
        logger.info(f"Training model {i}/{len(configs)}")
        try:
            model_path, model_name, training_time = train_lora_model(config, model_number=i)
            logger.info(f"Successfully trained and merged model {i}: {model_name} in {training_time:.2f} minutes")
        except Exception as e:
            logger.error(f"Error training model {i}: {str(e)}", exc_info=True)

    logger.info("All model training completed")

if __name__ == "__main__":
    main()