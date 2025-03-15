#!/usr/bin/env python3
# Epoch-wise Improvement Plot

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from e_0_20250305_plot_utils_v1_00 import get_linestyle, save_plot

def plot_epoch_improvement(optimizer_data, run_dir):
    """Create epoch-wise improvement plot showing relative progress per epoch"""
    # Define pastel colors directly
    pastel_colors = ["SkyBlue", "Coral"]  # Named colors or hex codes
    
    # Create a figure with a white background
    plt.figure(figsize=(12, 8), dpi=80)
    plt.gca().set_facecolor("white")  # Ensures the plot area is white
    
    for idx, (optimizer, runs) in enumerate(optimizer_data.items()):
        for run in runs:
            metrics = run["metrics"]
            
            # Need to compute epoch-wise aggregation of loss values
            if "loss_log" in metrics:
                steps = [entry["step"] for entry in metrics["loss_log"]]
                losses = [entry["loss"] for entry in metrics["loss_log"]]
                
                # Try to determine epochs from the data
                epochs = []
                epoch_losses = []
                epoch_improvements = []
                
                # Method 1: Use explicit epoch information if available
                if "epoch_log" in metrics:
                    epochs = [entry["epoch"] for entry in metrics["epoch_log"]]
                    epoch_losses = [entry["loss"] for entry in metrics["epoch_log"]]
                
                # Method 2: Try to infer epochs from steps and number of total epochs
                elif "num_epochs" in metrics and len(steps) > 0:
                    num_epochs = metrics["num_epochs"]
                    max_step = steps[-1]
                    
                    if num_epochs > 0 and max_step > 0:
                        steps_per_epoch = max_step / num_epochs
                        
                        # Create bins for each epoch
                        epoch_bins = {}
                        for i, (step, loss) in enumerate(zip(steps, losses)):
                            epoch = int(step / steps_per_epoch)
                            if epoch not in epoch_bins:
                                epoch_bins[epoch] = []
                            epoch_bins[epoch].append(loss)
                        
                        # Compute average loss for each epoch
                        for epoch in sorted(epoch_bins.keys()):
                            epochs.append(epoch)
                            epoch_losses.append(np.mean(epoch_bins[epoch]))
                
                # Method 3: As a fallback, arbitrarily divide steps into epochs
                elif steps:
                    # Divide into 5 artificial "epochs" for visualization
                    num_artificial_epochs = 5
                    step_bins = np.array_split(range(len(steps)), num_artificial_epochs)
                    
                    for i, bin_indices in enumerate(step_bins):
                        if len(bin_indices) > 0:
                            epochs.append(i)
                            epoch_losses.append(np.mean([losses[idx] for idx in bin_indices]))
                
                # Calculate improvements between consecutive epochs
                if len(epoch_losses) > 1:
                    initial_loss = epoch_losses[0]
                    for i in range(1, len(epoch_losses)):
                        # Calculate relative improvement: (previous_loss - current_loss) / initial_loss
                        improvement = (epoch_losses[i-1] - epoch_losses[i]) / initial_loss
                        epoch_improvements.append(improvement)
                    
                    improvement_epochs = epochs[1:]  # Improvements start from the second epoch
                    
                    # Use shorter labels
                    model_name = os.path.basename(run['model_dir'])
                    if len(model_name) > 20:
                        model_name = model_name[:17] + "..."
                    label = f"{optimizer.upper()}"
                    
                    # Assign pastel colors directly
                    line_color = pastel_colors[idx % len(pastel_colors)]
                    
                    plt.plot(improvement_epochs, epoch_improvements, 
                            label=label, 
                            color=line_color,
                            linestyle=get_linestyle(optimizer),
                            linewidth=3,
                            marker='o',
                            markersize=8)
    
    plt.title("Epoch-wise Relative Improvement", fontsize=18, color="dimgrey")
    plt.xlabel("Epoch", fontsize=14, color="dimgrey")
    plt.ylabel("Relative Loss Improvement\n(Δloss / initial_loss)", fontsize=14, color="dimgrey")
    
    # Darker grey gridlines
    plt.grid(True, color='dimgray', linestyle=':', alpha=0.8)
    
    # Add a horizontal line at y=0 to show the no-improvement point
    plt.axhline(y=0, color='dimgray', linestyle='--', alpha=0.6, linewidth=2)
    
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
    save_plot(plt, run_dir, "e_16_epoch_improvement.png", dpi=80)
    
    return plt.gcf()