#!/usr/bin/env python3
# Best Validation Loss Comparison Plot

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from e_0_20250305_plot_utils_v1_00 import setup_plot_style, save_plot

def plot_best_val_loss(optimizer_data, run_dir):
    """Create best validation loss comparison plot with properly sized elements"""
    
    # Define pastel colors directly
    pastel_colors = ["SkyBlue", "Coral"]  # Named colors or hex codes can be used
    
    # Create a figure with a white background
    plt.figure(figsize=(12, 8), dpi=80)
    ax = plt.gca()
    ax.set_facecolor("white")  # Ensures the plot area is white
    
    optimizer_names = []
    best_val_losses = []
    model_names = []
    
    for optimizer, runs in optimizer_data.items():
        for run in runs:
            metrics = run["metrics"]
            optimizer_names.append(optimizer.upper())
            
            # Handle None values by defaulting to 0
            val_loss = metrics.get("best_validation_loss", 0)
            if val_loss is None:
                val_loss = 0
            best_val_losses.append(val_loss)
            
            model_names.append(run["model_dir"])
    
    df = pd.DataFrame({
        "Optimizer": optimizer_names,
        "Best Validation Loss": best_val_losses,
        "Model": model_names
    })
    
    # Create color mapping for consistent colors
    optimizer_unique = df["Optimizer"].unique()
    color_mapping = {optimizer: pastel_colors[i % len(pastel_colors)] 
                     for i, optimizer in enumerate(optimizer_unique)}
    
    # Plot validation loss
    bar_width = 0.6
    positions = np.arange(len(df))
    
    bars = ax.bar(positions, df["Best Validation Loss"], 
            width=bar_width, 
            color=[color_mapping[opt] for opt in df["Optimizer"]],
            edgecolor='white',
            linewidth=1)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=12, color='dimgrey')
    
    plt.title("Best Validation Loss Comparison", fontsize=18, color="dimgrey")
    plt.ylabel("Best Validation Loss", fontsize=14, color="dimgrey")
    plt.xticks(positions, df["Optimizer"], fontsize=14, color="darkgray")
    
    # Darker grey gridlines (only on y-axis)
    plt.grid(True, axis='y', color='dimgray', linestyle=':', alpha=0.8)
    
    # Thicker x and y axis lines
    plt.gca().spines['bottom'].set_color('dimgray')  
    plt.gca().spines['bottom'].set_linewidth(4.0)  
    plt.gca().spines['left'].set_color('dimgray')  
    plt.gca().spines['left'].set_linewidth(4.0)  
    # Lighter top and right axis lines
    plt.gca().spines['top'].set_color('silver')  
    plt.gca().spines['right'].set_color('silver')  
    
    # Adjust tick label size and color
    plt.yticks(fontsize=14, color="darkgray")
    
    plt.tight_layout(pad=0.5)
    
    # Save the plot
    save_plot(plt, run_dir, "e_8_best_val_loss_comparison.png", dpi=80)
    
    return plt.gcf()