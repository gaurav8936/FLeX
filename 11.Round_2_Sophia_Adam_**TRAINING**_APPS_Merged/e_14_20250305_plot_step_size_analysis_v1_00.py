#!/usr/bin/env python3
# Step Size Analysis Plot

import os
import matplotlib.pyplot as plt
import numpy as np
from e_0_20250305_plot_utils_v1_00 import get_linestyle, save_plot

def plot_step_size_analysis(optimizer_data, run_dir):
    """Create step size analysis plot showing parameter update magnitudes"""
    # Define pastel colors directly
    pastel_colors = ["SkyBlue", "Coral"]  # Named colors or hex codes
    
    # Create a figure with a white background
    plt.figure(figsize=(12, 8), dpi=80)
    plt.gca().set_facecolor("white")  # Ensures the plot area is white
    
    for idx, (optimizer, runs) in enumerate(optimizer_data.items()):
        for run in runs:
            metrics = run["metrics"]
            
            # Try to extract update magnitude data
            for possible_key in ["update_magnitude_log", "update_norm_log", "parameter_update_log"]:
                if possible_key in metrics:
                    steps = [entry["step"] for entry in metrics[possible_key]]
                    magnitudes = [entry[possible_key.replace("_log", "")] for entry in metrics[possible_key]]
                    
                    if steps and magnitudes:
                        # Use shorter labels
                        model_name = os.path.basename(run['model_dir'])
                        if len(model_name) > 20:
                            model_name = model_name[:17] + "..."
                        label = f"{optimizer.upper()}"
                        
                        # Assign pastel colors directly
                        line_color = pastel_colors[idx % len(pastel_colors)]
                        
                        plt.plot(steps, magnitudes, label=label, 
                                color=line_color,
                                linestyle=get_linestyle(optimizer),
                                linewidth=3)  # Thickened lines
                        break
            
            # If no explicit update magnitude logs, try to calculate from gradient and learning rate
            if len(plt.gca().get_lines()) == 0 and "gradient_norm_log" in metrics and "lr_log" in metrics:
                steps_grad = [entry["step"] for entry in metrics["gradient_norm_log"]]
                grad_norms = [entry["gradient_norm"] for entry in metrics["gradient_norm_log"]]
                
                steps_lr = [entry["step"] for entry in metrics["lr_log"]]
                learning_rates = [entry["lr"] for entry in metrics["lr_log"]]
                
                # Make sure we have matching steps
                valid_indices = []
                step_values = []
                estimated_magnitudes = []
                
                for i, step in enumerate(steps_grad):
                    if step in steps_lr:
                        lr_idx = steps_lr.index(step)
                        valid_indices.append(i)
                        step_values.append(step)
                        # Approximate update magnitude as gradient_norm * learning_rate
                        estimated_magnitudes.append(grad_norms[i] * learning_rates[lr_idx])
                
                if step_values and estimated_magnitudes:
                    # Use shorter labels
                    model_name = os.path.basename(run['model_dir'])
                    if len(model_name) > 20:
                        model_name = model_name[:17] + "..."
                    label = f"{optimizer.upper()} (estimated)"
                    
                    # Assign pastel colors directly
                    line_color = pastel_colors[idx % len(pastel_colors)]
                    
                    plt.plot(step_values, estimated_magnitudes, label=label, 
                            color=line_color,
                            linestyle=get_linestyle(optimizer),
                            linewidth=3)  # Thickened lines
    
    plt.title("Parameter Update Magnitude", fontsize=18, color="dimgrey")
    plt.xlabel("Steps", fontsize=14, color="dimgrey")
    plt.ylabel("Update Magnitude", fontsize=14, color="dimgrey")
    
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
    
    # Add a note if we're using estimated values
    has_estimated = any("estimated" in line.get_label() for line in plt.gca().get_lines())
    if has_estimated:
        plt.figtext(0.5, 0.01, "Note: Some values are estimated as gradient_norm × learning_rate", 
                  ha='center', fontsize=12, alpha=0.7, color="dimgrey")
    
    # Make legend compact
    plt.legend(fontsize=14, loc='upper right', framealpha=0.7)
    
    # Adjust tick label size and color
    plt.xticks(fontsize=14, color="darkgray")
    plt.yticks(fontsize=14, color="darkgray")
    
    plt.tight_layout(pad=0.5)
    
    # Save the plot
    save_plot(plt, run_dir, "e_14_step_size_analysis.png", dpi=80)
    
    return plt.gcf()