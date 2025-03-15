#!/usr/bin/env python3
# Hessian-Related Metrics Plot

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from e_0_20250305_plot_utils_v1_00 import setup_plot_style, extract_metric_series, save_plot

def plot_hessian_metrics(optimizer_data, run_dir):
    """Create plot for Hessian-related metrics in Sophia optimizer with properly sized elements"""
    
    # Define pastel colors directly
    pastel_colors = ["SkyBlue", "Coral"]  # Named colors or hex codes can be used
    
    # Check if we have Sophia data with Hessian metrics
    sophia_runs = optimizer_data.get("sophia", [])
    
    if not sophia_runs:
        # No Sophia data available
        return None
    
    # Create a figure with a white background
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), dpi=80)
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")
    
    # Use the Coral color for Sophia
    sophia_color = pastel_colors[1]  # Coral
    
    for run_idx, run in enumerate(sophia_runs):
        metrics = run["metrics"]
        
        # Plot clipping rates if available
        if "clipped_updates_log" in metrics and metrics["clipped_updates_log"]:
            # Extract clipping data - use the correct field name from the entries
            steps = [entry["step"] for entry in metrics["clipped_updates_log"]]
            clipping_rates = [entry["fraction"] for entry in metrics["clipped_updates_log"]]
            
            if steps and clipping_rates:
                # Apply smoothing
                if len(steps) > 5:
                    window_size = 5
                    weights = np.ones(window_size) / window_size
                    clipping_rates_smooth = np.convolve(clipping_rates, weights, mode='same')
                    clipping_rates_smooth[:window_size//2] = clipping_rates[:window_size//2]
                    clipping_rates_smooth[-window_size//2:] = clipping_rates[-window_size//2:]
                else:
                    clipping_rates_smooth = clipping_rates
                
                # Use shorter labels
                model_name = os.path.basename(run['model_dir'])
                if len(model_name) > 20:
                    model_name = model_name[:17] + "..."
                label = "SOPHIA"
                
                ax1.plot(steps, clipping_rates_smooth, label=label, 
                       color=sophia_color,
                       linewidth=3)  # Thickened lines
        
        # Plot Hessian diagonal magnitudes if available
        if "hessian_diag_log" in metrics and metrics["hessian_diag_log"]:
            # Extract Hessian data - use the correct field name from the entries
            steps = [entry["step"] for entry in metrics["hessian_diag_log"]]
            hessian_diags = [entry["hessian_diag"] for entry in metrics["hessian_diag_log"]]
            
            if steps and hessian_diags:
                # Apply smoothing
                if len(steps) > 5:
                    window_size = 5
                    weights = np.ones(window_size) / window_size
                    hessian_diags_smooth = np.convolve(hessian_diags, weights, mode='same')
                    hessian_diags_smooth[:window_size//2] = hessian_diags[:window_size//2]
                    hessian_diags_smooth[-window_size//2:] = hessian_diags[-window_size//2:]
                else:
                    hessian_diags_smooth = hessian_diags
                
                # Use shorter labels
                model_name = os.path.basename(run['model_dir'])
                if len(model_name) > 20:
                    model_name = model_name[:17] + "..."
                label = "SOPHIA"
                
                ax2.plot(steps, hessian_diags_smooth, label=label, 
                       color=sophia_color,
                       linewidth=3)  # Thickened lines
    
    # If we don't have specific Hessian metrics but we have clipped_gradients
    # as a fallback, try to use those instead
    if all(len(ax.get_lines()) == 0 for ax in [ax1, ax2]):
        for run_idx, run in enumerate(sophia_runs):
            metrics = run["metrics"]
            
            for i, (metric_name, title, ylabel) in enumerate([
                ("clipped_gradients", "Gradient Clipping Frequency", "Clipping Rate"),
                ("update_magnitude", "Update Magnitude", "Average Magnitude")
            ]):
                ax = [ax1, ax2][i]
                if f"{metric_name}_log" in metrics and metrics[f"{metric_name}_log"]:
                    steps, values = extract_metric_series(metrics, metric_name)
                    
                    if steps and values:
                        # Apply smoothing
                        if len(steps) > 5:
                            window_size = 5
                            weights = np.ones(window_size) / window_size
                            values_smooth = np.convolve(values, weights, mode='same')
                            values_smooth[:window_size//2] = values[:window_size//2]
                            values_smooth[-window_size//2:] = values[-window_size//2:]
                        else:
                            values_smooth = values
                        
                        # Use shorter labels
                        model_name = os.path.basename(run['model_dir'])
                        if len(model_name) > 20:
                            model_name = model_name[:17] + "..."
                        label = "SOPHIA"
                        
                        ax.plot(steps, values_smooth, label=label, 
                              color=sophia_color,
                              linewidth=3)  # Thickened lines
    
    # Add a note if no Hessian metrics were found
    if all(len(ax.get_lines()) == 0 for ax in [ax1, ax2]):
        fig.text(0.5, 0.5, "No Hessian-related metrics available in the data", 
                ha='center', va='center', fontsize=18, color="dimgrey")
    else:
        # Style both subplots
        for i, (ax, title, ylabel) in enumerate([
            (ax1, "Update Clipping Rate", "Fraction of Clipped Updates"),
            (ax2, "Hessian Diagonal Magnitude", "Average Diagonal Value")
        ]):
            ax.set_title(title, fontsize=18, color="dimgrey")
            ax.set_xlabel("Steps", fontsize=14, color="dimgrey")
            ax.set_ylabel(ylabel, fontsize=14, color="dimgrey")
            
            # Darker grey gridlines
            ax.grid(True, color='dimgray', linestyle=':', alpha=0.8)
            
            # Thicker x and y axis lines
            ax.spines['bottom'].set_color('dimgray')
            ax.spines['bottom'].set_linewidth(4.0)
            ax.spines['left'].set_color('dimgray')
            ax.spines['left'].set_linewidth(4.0)
            # Lighter top and right axis lines
            ax.spines['top'].set_color('silver')
            ax.spines['right'].set_color('silver')
            
            # Make legend compact
            if len(ax.get_lines()) > 0:
                ax.legend(fontsize=14, loc='upper right', framealpha=0.7)
            
            # Adjust tick label size and color
            ax.tick_params(axis='both', which='major', labelsize=14, labelcolor="darkgray")
    
    plt.tight_layout(pad=2.0)
    
    # Save the plot
    save_plot(fig, run_dir, "e_13_hessian_metrics.png", dpi=80)
    
    return fig