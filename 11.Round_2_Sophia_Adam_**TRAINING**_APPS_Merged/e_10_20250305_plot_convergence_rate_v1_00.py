#!/usr/bin/env python3
# Convergence Rate Comparison Plot

import os
import matplotlib.pyplot as plt
import numpy as np
from e_0_20250305_plot_utils_v1_00 import setup_plot_style, get_linestyle, extract_metric_series, save_plot

def plot_convergence_rate(optimizer_data, run_dir):
    """Create convergence rate comparison plot (normalized loss reduction) with properly sized elements"""
    
    # Define pastel colors directly
    pastel_colors = ["SkyBlue", "Coral"]  # Named colors or hex codes can be used
    
    # Create a figure with a white background
    plt.figure(figsize=(12, 8), dpi=80)
    ax = plt.gca()
    ax.set_facecolor("white")  # Ensures the plot area is white
    
    for idx, (optimizer, runs) in enumerate(optimizer_data.items()):
        for run in runs:
            metrics = run["metrics"]
            if "loss_log" in metrics:
                steps, losses = extract_metric_series(metrics, "loss")
                
                if steps and losses and len(losses) > 1:
                    # Normalize losses to starting point
                    initial_loss = losses[0]
                    normalized_losses = [loss/initial_loss for loss in losses]
                    
                    # Apply smoothing
                    if len(steps) > 5:
                        window_size = 5
                        weights = np.ones(window_size) / window_size
                        normalized_losses_smooth = np.convolve(normalized_losses, weights, mode='same')
                        normalized_losses_smooth[:window_size//2] = normalized_losses[:window_size//2]
                        normalized_losses_smooth[-window_size//2:] = normalized_losses[-window_size//2:]
                    else:
                        normalized_losses_smooth = normalized_losses
                    
                    # Use shorter labels
                    model_name = os.path.basename(run['model_dir'])
                    if len(model_name) > 20:
                        model_name = model_name[:17] + "..."
                    label = f"{optimizer.upper()}"
                    
                    # Assign pastel colors directly
                    line_color = pastel_colors[idx % len(pastel_colors)]
                    plt.plot(steps, normalized_losses_smooth, label=label, 
                             color=line_color,
                             linestyle=get_linestyle(optimizer),
                             linewidth=3)  # Thickened lines
    
    plt.title("Convergence Rate Comparison", fontsize=18, color="dimgrey")
    plt.xlabel("Steps", fontsize=14, color="dimgrey")
    plt.ylabel("Normalized Loss (Loss/Initial Loss)", fontsize=14, color="dimgrey")
    
    # Add a horizontal line at y=1.0 to show the starting point
    plt.axhline(y=1.0, color='dimgray', linestyle='--', alpha=0.5, linewidth=2)
    
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
    save_plot(plt, run_dir, "e_10_convergence_rate_comparison.png", dpi=80)
    
    return plt.gcf()