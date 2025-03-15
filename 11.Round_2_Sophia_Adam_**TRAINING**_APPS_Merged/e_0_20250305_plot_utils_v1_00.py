#!/usr/bin/env python3
# Common utilities for visualization modules

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set up common styling
def setup_plot_style():
    """Configure common plot styling"""
    plt.style.use('ggplot')
    sns.set(font_scale=1.2)
    return {"ADAMW": "#1f77b4", "SOPHIA": "#ff7f0e"}

# Extract data from metrics
def extract_metric_series(metrics, metric_name):
    """Extract a metric series from metrics JSON"""
    if f"{metric_name}_log" in metrics:
        data = metrics[f"{metric_name}_log"]
        
        # Different metrics might have different key names in their entries
        # For example, 'loss_variance_log' might have entries with 'variance' as the key
        # instead of 'loss_variance'
        value_key = metric_name
        if metric_name == "loss_variance" and len(data) > 0 and "variance" in data[0]:
            value_key = "variance"
            
        steps = [entry["step"] for entry in data]
        values = [entry[value_key] for entry in data]
        return steps, values
    return [], []

def get_optimizer_colors():
    """Get standard colors for optimizers"""
    return {"ADAMW": "#1f77b4", "SOPHIA": "#ff7f0e"}

def get_linestyle(optimizer):
    """Get standard line style for optimizer"""
    return "-" if optimizer.lower() == "adamw" else "--"

def save_plot(fig_or_plt, run_dir, filename, dpi=300):
    """Save plot to the run directory"""
    # If matplotlib.pyplot is passed, use the current figure
    if fig_or_plt == plt:
        fig_or_plt.savefig(os.path.join(run_dir, filename), dpi=dpi)
        fig_or_plt.close()
    else:
        # If a figure object is passed
        fig_or_plt.savefig(os.path.join(run_dir, filename), dpi=dpi)
        plt.close(fig_or_plt)