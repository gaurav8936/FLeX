#!/usr/bin/env python3
# Resources (Time & Memory) Comparison Plot

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from e_0_20250305_plot_utils_v1_00 import setup_plot_style, save_plot

def plot_resources(optimizer_data, run_dir):
    """Create training time and memory usage comparison plot with properly sized elements"""
    
    # Define pastel colors directly
    pastel_colors = ["SkyBlue", "Coral"]  # Named colors or hex codes can be used
    
    # Create a figure with a white background
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), dpi=80)
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")
    
    optimizer_names = []
    training_times = []
    memory_usages = []
    model_names = []
    
    for optimizer, runs in optimizer_data.items():
        for run in runs:
            metrics = run["metrics"]
            optimizer_names.append(optimizer.upper())
            
            # Handle None values for training time
            training_time = metrics.get("total_training_time", 0)
            if training_time is None:
                training_time = 0
            training_times.append(training_time)
            
            # Handle None values for memory usage
            memory_usage = metrics.get("peak_memory_usage", 0)
            if memory_usage is None:
                memory_usage = 0
            memory_usages.append(memory_usage)
            
            model_names.append(run["model_dir"])
    
    df = pd.DataFrame({
        "Optimizer": optimizer_names,
        "Training Time (minutes)": training_times,
        "Peak Memory (MB)": memory_usages,
        "Model": model_names
    })
    
    # Create color mapping for consistent colors
    optimizer_unique = df["Optimizer"].unique()
    color_mapping = {optimizer: pastel_colors[i % len(pastel_colors)] 
                     for i, optimizer in enumerate(optimizer_unique)}
    
    # Plot training time
    bar_width = 0.6
    positions = np.arange(len(df))
    
    # First subplot - Training Time
    bars1 = ax1.bar(positions, df["Training Time (minutes)"], 
              width=bar_width, 
              color=[color_mapping[opt] for opt in df["Optimizer"]],
              edgecolor='white',
              linewidth=1)
    
    ax1.set_title("Training Time Comparison", fontsize=18, color="dimgrey")
    ax1.set_ylabel("Training Time (minutes)", fontsize=14, color="dimgrey")
    ax1.set_xticks(positions)
    ax1.set_xticklabels(df["Optimizer"], fontsize=14, color="darkgray")
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=12, color='dimgrey')
    
    # Second subplot - Memory Usage
    bars2 = ax2.bar(positions, df["Peak Memory (MB)"], 
              width=bar_width, 
              color=[color_mapping[opt] for opt in df["Optimizer"]],
              edgecolor='white',
              linewidth=1)
    
    ax2.set_title("Peak Memory Usage Comparison", fontsize=18, color="dimgrey")
    ax2.set_ylabel("Peak Memory (MB)", fontsize=14, color="dimgrey")
    ax2.set_xticks(positions)
    ax2.set_xticklabels(df["Optimizer"], fontsize=14, color="darkgray")
    
    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=12, color='dimgrey')
    
    # Apply styling to both axes
    for ax in [ax1, ax2]:
        # Darker grey gridlines (only on y-axis)
        ax.grid(True, axis='y', color='dimgray', linestyle=':', alpha=0.8)
        
        # Thicker x and y axis lines
        ax.spines['bottom'].set_color('dimgray')
        ax.spines['bottom'].set_linewidth(4.0)
        ax.spines['left'].set_color('dimgray')
        ax.spines['left'].set_linewidth(4.0)
        # Lighter top and right axis lines
        ax.spines['top'].set_color('silver')
        ax.spines['right'].set_color('silver')
        
        # Adjust tick label size and color
        ax.tick_params(axis='y', labelsize=14, labelcolor="darkgray")
    
    plt.tight_layout(pad=2.0)
    
    # Save the plot
    save_plot(fig, run_dir, "e_7_resources_comparison.png", dpi=80)
    
    return fig