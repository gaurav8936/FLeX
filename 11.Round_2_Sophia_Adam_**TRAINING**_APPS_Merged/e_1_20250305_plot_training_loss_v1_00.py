#!/usr/bin/env python3
# Training Loss Comparison Plot

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from e_0_20250305_plot_utils_v1_00 import setup_plot_style, get_linestyle, save_plot

def plot_training_loss(optimizer_data, run_dir):
    """Create training loss comparison plot with properly sized elements"""
    
    # Define pastel colors directly
    pastel_colors = ["SkyBlue", "Coral"]  # Named colors or hex codes can be used
    
    # Create a figure with a white background
    plt.figure(figsize=(12, 8), dpi=80)
    plt.gca().set_facecolor("white")  # Ensures the plot area is white

    for idx, (optimizer, runs) in enumerate(optimizer_data.items()):
        for run in runs:
            metrics = run["metrics"]
            if "loss_log" in metrics:
                steps = [entry["step"] for entry in metrics["loss_log"]]
                losses = [entry["loss"] for entry in metrics["loss_log"]]
                
                # Apply smoothing
                if len(steps) > 5:
                    window_size = 5
                    weights = np.ones(window_size) / window_size
                    losses_smooth = np.convolve(losses, weights, mode='same')
                    losses_smooth[:window_size//2] = losses[:window_size//2]
                    losses_smooth[-window_size//2:] = losses[-window_size//2:]
                else:
                    losses_smooth = losses
                
                # Use shorter labels
                model_name = os.path.basename(run['model_dir'])
                if len(model_name) > 20:
                    model_name = model_name[:17] + "..."
                label = f"{optimizer.upper()}"
                
                # Assign pastel colors directly
                line_color = pastel_colors[idx % len(pastel_colors)]  # Rotate colors if more than 2 optimizers
                plt.plot(steps, losses_smooth, label=label, 
                         color=line_color,
                         linestyle=get_linestyle(optimizer),
                         linewidth=3)  # Thickened lines
    
    plt.title("Training Loss Comparison", fontsize=18, color="dimgrey")
    plt.xlabel("Steps", fontsize=14, color="dimgrey")
    plt.ylabel("Loss", fontsize=14, color="dimgrey")
    
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
    save_plot(plt, run_dir, "e_1_training_loss_comparison.png", dpi=80)
    
    return plt.gcf()
