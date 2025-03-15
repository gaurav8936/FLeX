#!/usr/bin/env python3
# Validation/Training Loss Ratio Plot

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from e_0_20250305_plot_utils_v1_00 import get_linestyle, save_plot

def plot_val_train_ratio(optimizer_data, run_dir):
    """Create validation/training loss ratio plot to measure generalization"""
    # Define pastel colors directly
    pastel_colors = ["SkyBlue", "Coral"]  # Named colors or hex codes
    
    # Create a figure with a white background
    plt.figure(figsize=(12, 8), dpi=80)
    plt.gca().set_facecolor("white")  # Ensures the plot area is white
    
    for idx, (optimizer, runs) in enumerate(optimizer_data.items()):
        for run in runs:
            timestamp = run.get("timestamp")
            metrics = run["metrics"]
            
            # Read validation loss from CSV file (like in e_2)
            metrics_dir = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/11.Round_2_Sophia_Adam_**TRAINING**_APPS/03.json_csv_logs_metrics_plot_scope_only"
            val_loss_file = f"{optimizer}_{timestamp}_val_loss.csv"
            val_loss_path = os.path.join(metrics_dir, val_loss_file)
            
            print(f"Looking for validation file for ratio plot: {val_loss_path}")
            
            if os.path.exists(val_loss_path) and "loss_log" in metrics:
                print(f"Found validation file for ratio plot")
                
                # Read the validation loss data
                val_data = pd.read_csv(val_loss_path)
                
                # Extract epochs and validation losses
                validation_epochs = val_data['epoch'].values
                validation_losses = val_data['val_loss'].values
                
                # Now find matching training losses for each validation epoch
                training_losses = []
                valid_epochs = []
                valid_val_losses = []
                
                # Calculate steps per epoch to match training and validation
                if len(metrics["loss_log"]) > 0 and len(validation_epochs) > 0:
                    total_steps = metrics["loss_log"][-1]["step"]
                    max_epoch = max(validation_epochs)
                    if max_epoch > 0:
                        steps_per_epoch = total_steps / max_epoch
                        
                        for i, epoch in enumerate(validation_epochs):
                            # Find the training loss at the end of this epoch
                            target_step = int(epoch * steps_per_epoch)
                            
                            # Find the closest step
                            closest_idx = None
                            closest_diff = float('inf')
                            
                            for j, entry in enumerate(metrics["loss_log"]):
                                diff = abs(entry["step"] - target_step)
                                if diff < closest_diff:
                                    closest_diff = diff
                                    closest_idx = j
                            
                            if closest_idx is not None:
                                train_loss = metrics["loss_log"][closest_idx]["loss"]
                                training_losses.append(train_loss)
                                valid_epochs.append(epoch)
                                valid_val_losses.append(validation_losses[i])
                
                # If we found matching validation and training losses, compute and plot the ratio
                if valid_epochs and training_losses:
                    ratios = [val / train for val, train in zip(valid_val_losses, training_losses)]
                    
                    # Use shorter labels
                    label = f"{optimizer.upper()}"
                    
                    # Assign pastel colors directly
                    line_color = pastel_colors[idx % len(pastel_colors)]
                    
                    plt.plot(valid_epochs, ratios, label=label, 
                            color=line_color,
                            linestyle=get_linestyle(optimizer),
                            linewidth=3,
                            marker='o',
                            markersize=8)
    
    plt.title("Validation/Training Loss Ratio", fontsize=18, color="dimgrey")
    plt.xlabel("Epoch", fontsize=14, color="dimgrey")
    plt.ylabel("Validation Loss / Training Loss", fontsize=14, color="dimgrey")
    
    # Darker grey gridlines
    plt.grid(True, color='dimgray', linestyle=':', alpha=0.8)
    
    # Add a horizontal line at y=1.0 to show the parity point
    plt.axhline(y=1.0, color='dimgray', linestyle='--', alpha=0.6, linewidth=2)
    
    # Thicker x and y axis lines
    plt.gca().spines['bottom'].set_color('dimgray')
    plt.gca().spines['bottom'].set_linewidth(4.0)
    plt.gca().spines['left'].set_color('dimgray')
    plt.gca().spines['left'].set_linewidth(4.0)
    
    # Lighter top and right axis lines
    plt.gca().spines['top'].set_color('silver')
    plt.gca().spines['right'].set_color('silver')
    
    # Make legend compact
    plt.legend(fontsize=14, loc='upper right', framealpha=0.7)
    
    # Adjust tick label size and color
    plt.xticks(fontsize=14, color="darkgray")
    plt.yticks(fontsize=14, color="darkgray")
    
    plt.tight_layout(pad=0.5)
    
    # Save the plot
    save_plot(plt, run_dir, "e_15_val_train_ratio.png", dpi=80)
    
    return plt.gcf()