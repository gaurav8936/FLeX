#!/usr/bin/env python3
# Validation Loss Comparison Plot

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from e_0_20250305_plot_utils_v1_00 import setup_plot_style, save_plot

def plot_validation_loss(optimizer_data, run_dir):
    """Create validation loss comparison plot with properly sized elements"""
    
    # Define pastel colors directly
    pastel_colors = ["SkyBlue", "Coral"]  # Named colors or hex codes can be used
    
    # Create a figure with a white background
    plt.figure(figsize=(12, 8), dpi=80)
    plt.gca().set_facecolor("white")  # Ensures the plot area is white
    
    # Print debug information
    print("Optimizers in data:", list(optimizer_data.keys()))
    
    for idx, (optimizer, runs) in enumerate(optimizer_data.items()):
        print(f"Processing optimizer: {optimizer}")
        
        for run in runs:
            timestamp = run.get("timestamp")
            print(f"  Run timestamp: {timestamp}")
            
            # The JSON metrics file path should be included in the run data
            # Construct the validation loss file path from it
            metrics_dir = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/11.Round_2_Sophia_Adam_**TRAINING**_APPS/03.json_csv_logs_metrics_plot_scope_only"
            val_loss_file = f"{optimizer}_{timestamp}_val_loss.csv"
            val_loss_path = os.path.join(metrics_dir, val_loss_file)
            
            print(f"  Looking for validation file: {val_loss_path}")
            
            # Check if the validation loss file exists
            if os.path.exists(val_loss_path):
                print(f"  Found validation file: {val_loss_path}")
                
                # Read the validation loss data
                val_data = pd.read_csv(val_loss_path)
                
                # Extract epochs and validation losses
                val_epochs = val_data['epoch'].values
                val_losses = val_data['val_loss'].values
                
                print(f"  Validation data: {list(zip(val_epochs, val_losses))}")
                
                # Use shorter labels
                label = f"{optimizer.upper()}"
                
                # Assign pastel colors directly
                line_color = pastel_colors[idx % len(pastel_colors)]
                plt.plot(val_epochs, val_losses, marker='o', label=label, 
                         color=line_color,
                         linewidth=3)  # Thickened lines
            else:
                print(f"  Validation file not found: {val_loss_path}")
                
                # Fall back to using best_validation_loss if available
                if 'metrics' in run and 'best_validation_loss' in run['metrics']:
                    best_val = run['metrics']['best_validation_loss']
                    print(f"  Using best_validation_loss: {best_val}")
                    
                    line_color = pastel_colors[idx % len(pastel_colors)]
                    plt.plot([1], [best_val], marker='o', label=optimizer.upper(), 
                            color=line_color, linewidth=3)
    
    plt.title("Validation Loss Comparison", fontsize=18, color="dimgrey")
    plt.xlabel("Epoch", fontsize=14, color="dimgrey")
    plt.ylabel("Validation Loss", fontsize=14, color="dimgrey")
    
    # Darker grey gridlines
    plt.grid(True, color='dimgray', linestyle=':', alpha=0.8)  
    
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
    save_plot(plt, run_dir, "e_2_validation_loss_comparison.png", dpi=80)
    
    return plt.gcf()