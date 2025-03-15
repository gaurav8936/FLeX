#!/usr/bin/env python3
# Gradient Norm Comparison Plot

import os
import matplotlib.pyplot as plt
import numpy as np
from e_0_20250305_plot_utils_v1_00 import setup_plot_style, get_linestyle, extract_metric_series, save_plot

def plot_gradient_norm(optimizer_data, run_dir):
    """Create gradient norm comparison plot with properly sized elements"""
    
    # Define pastel colors directly
    pastel_colors = ["SkyBlue", "Coral"]  # Named colors or hex codes can be used
    
    # Create a figure with a white background
    plt.figure(figsize=(12, 8), dpi=80)
    plt.gca().set_facecolor("white")  # Ensures the plot area is white
    
    for idx, (optimizer, runs) in enumerate(optimizer_data.items()):
        for run in runs:
            metrics = run["metrics"]
            
            # Extract gradient norm data
            if "gradient_norm_log" in metrics:
                steps, gradient_norms = extract_metric_series(metrics, "gradient_norm")
                
                if steps and gradient_norms:
                    # Apply smoothing
                    if len(steps) > 5:
                        window_size = 5
                        weights = np.ones(window_size) / window_size
                        norms_smooth = np.convolve(gradient_norms, weights, mode='same')
                        norms_smooth[:window_size//2] = gradient_norms[:window_size//2]
                        norms_smooth[-window_size//2:] = gradient_norms[-window_size//2:]
                    else:
                        norms_smooth = gradient_norms
                    
                    # Use shorter labels
                    model_name = os.path.basename(run['model_dir'])
                    if len(model_name) > 20:
                        model_name = model_name[:17] + "..."
                    label = f"{optimizer.upper()}"
                    
                    # Assign pastel colors directly
                    line_color = pastel_colors[idx % len(pastel_colors)]
                    plt.plot(steps, norms_smooth, label=label, 
                             color=line_color,
                             linestyle=get_linestyle(optimizer),
                             linewidth=3)  # Thickened lines
    
    plt.title("Gradient Norm Comparison", fontsize=18, color="dimgrey")
    plt.xlabel("Steps", fontsize=14, color="dimgrey")
    plt.ylabel("Gradient Norm", fontsize=14, color="dimgrey")
    
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
    save_plot(plt, run_dir, "e_5_gradient_norm_comparison.png", dpi=80)
    
    return plt.gcf()