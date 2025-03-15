#!/usr/bin/env python3
# Run multiple LoRA fine-tuning jobs with different configurations

import subprocess
import time
import os
from datetime import datetime

# Base script to run
base_script = "a_20250308_lora_finetuning_APPS_AdamOnly_UNMERGED_for_23CrossLingual_v1.00.py"

# Define variations to explore
configurations = [
    # Different learning rates
    {"optimizer": "adamw", "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "learning_rate": 0.0002, "num_epochs": 3},
    {"optimizer": "adamw", "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "learning_rate": 0.0003, "num_epochs": 3},
    {"optimizer": "adamw", "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "learning_rate": 0.0001, "num_epochs": 3},
    
    # Different lora_r values
    {"optimizer": "adamw", "lora_r": 16, "lora_alpha": 16, "lora_dropout": 0.05, "learning_rate": 0.0002, "num_epochs": 3},
    {"optimizer": "adamw", "lora_r": 4, "lora_alpha": 16, "lora_dropout": 0.05, "learning_rate": 0.0002, "num_epochs": 3},
    
    # Different lora_alpha values
    {"optimizer": "adamw", "lora_r": 8, "lora_alpha": 32, "lora_dropout": 0.05, "learning_rate": 0.0002, "num_epochs": 3},
    {"optimizer": "adamw", "lora_r": 8, "lora_alpha": 8, "lora_dropout": 0.05, "learning_rate": 0.0002, "num_epochs": 3},
    
    # Different dropout values
    {"optimizer": "adamw", "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.1, "learning_rate": 0.0002, "num_epochs": 3},
    {"optimizer": "adamw", "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "learning_rate": 0.0002, "num_epochs": 3},
    
    # One with sophia optimizer
    {"optimizer": "sophia", "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.05, "learning_rate": 0.0002, "num_epochs": 3},
]

# Create log directory
log_dir = "run_logs"
os.makedirs(log_dir, exist_ok=True)

# Run each configuration
for i, config in enumerate(configurations):
    # Create command with all parameters
    cmd = ["python", base_script]
    for param, value in config.items():
        cmd.extend([f"--{param}", str(value)])
    
    # Prepare log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_{i+1}_{timestamp}.log")
    
    print(f"Starting run {i+1}/{len(configurations)}")
    print(f"Configuration: {config}")
    print(f"Log file: {log_file}")
    
    # Run the command and log output
    with open(log_file, 'w') as f:
        f.write(f"Run {i+1}/{len(configurations)}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"Configuration: {config}\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.flush()
        
        # Run the process and capture output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output to log file and console
        for line in process.stdout:
            print(line, end='')
            f.write(line)
            f.flush()
        
        process.wait()
        
        f.write(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        f.write(f"\nExit code: {process.returncode}\n")
    
    print(f"Completed run {i+1}/{len(configurations)}")
    print(f"Exit code: {process.returncode}")
    print("-" * 50)
    
    # Optional: Add a delay between runs
    if i < len(configurations) - 1:
        time.sleep(30)  # 30 seconds between runs

print("All runs completed!")