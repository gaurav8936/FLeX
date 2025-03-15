#!/usr/bin/env python3
# Efficiency Ratio Plot

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from e_0_20250305_plot_utils_v1_00 import setup_plot_style, save_plot

def plot_efficiency_ratio(optimizer_data, run_dir):
    """Create efficiency ratio plot showing performance per computation cost with properly sized elements"""
    
    # Define pastel colors directly
    pastel_colors = ["SkyBlue", "Coral"]  # Named colors or hex codes can be used
    
    # Create a figure with a white background
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), dpi=80)
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")
    
    optimizer_names = []
    loss_improvements = []
    time_ratios = []
    memory_ratios = []
    model_names = []
    
    for optimizer, runs in optimizer_data.items():
        for run in runs:
            metrics = run["metrics"]
            
            # Calculate loss improvement
            if "loss_log" in metrics and len(metrics["loss_log"]) > 1:
                initial_loss = metrics["loss_log"][0]["loss"]
                final_loss = metrics["loss_log"][-1]["loss"]
                loss_improvement = initial_loss - final_loss
                
                # Get resource usage
                training_time = metrics.get("total_training_time", 1)
                if training_time is None or training_time <= 0:
                    training_time = 1
                
                memory_usage = metrics.get("peak_memory_usage", 1)
                if memory_usage is None or memory_usage <= 0:
                    memory_usage = 1
                
                # Calculate efficiency ratios
                time_ratio = loss_improvement / training_time
                memory_ratio = loss_improvement / memory_usage
                
                optimizer_names.append(optimizer.upper())
                loss_improvements.append(loss_improvement)
                time_ratios.append(time_ratio)
                memory_ratios.append(memory_ratio)
                model_names.append(run["model_dir"])
    
    if optimizer_names:
        df = pd.DataFrame({
            "Optimizer": optimizer_names,
            "Loss Improvement": loss_improvements,
            "Time Efficiency": time_ratios,
            "Memory Efficiency": memory_ratios,
            "Model": model_names
        })
        
        # Create color mapping for consistent colors
        optimizer_unique = df["Optimizer"].unique()
        color_mapping = {optimizer: pastel_colors[i % len(pastel_colors)] 
                        for i, optimizer in enumerate(optimizer_unique)}
        
        # Plot time efficiency
        positions1 = np.arange(len(df))
        bar_width = 0.6
        
        # First subplot - Time Efficiency
        bars1 = ax1.bar(positions1, df["Time Efficiency"], 
                width=bar_width, 
                color=[color_mapping[opt] for opt in df["Optimizer"]],
                edgecolor='white',
                linewidth=1)
        
        ax1.set_title("Loss Improvement per Training Time", fontsize=18, color="dimgrey")
        ax1.set_ylabel("Loss Reduction / Minute", fontsize=14, color="dimgrey")
        ax1.set_xticks(positions1)
        ax1.set_xticklabels(df["Optimizer"], fontsize=14, color="darkgray")
        
        # Add value labels inside bars instead of on top
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height/2,  # Position in middle of bar
                    f'{height:.5f}',
                    ha='center', va='center', fontsize=12, color='white')  # Changed to white and centered
        
        # Second subplot - Memory Efficiency
        positions2 = np.arange(len(df))
        bars2 = ax2.bar(positions2, df["Memory Efficiency"], 
                width=bar_width, 
                color=[color_mapping[opt] for opt in df["Optimizer"]],
                edgecolor='white',
                linewidth=1)
        
        ax2.set_title("Loss Improvement per Memory Usage", fontsize=18, color="dimgrey")
        ax2.set_ylabel("Loss Reduction / MB", fontsize=14, color="dimgrey")
        ax2.set_xticks(positions2)
        ax2.set_xticklabels(df["Optimizer"], fontsize=14, color="darkgray")
        
        # Add value labels inside bars instead of on top
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height/2,  # Position in middle of bar
                    f'{height:.7f}',
                    ha='center', va='center', fontsize=12, color='white')  # Changed to white and centered
        
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
        save_plot(fig, run_dir, "e_12_efficiency_ratio_comparison.png", dpi=80)
    else:
        plt.close(fig)
        fig = None
        
    return fig

# #!/usr/bin/env python3
# # Efficiency Ratio Plot

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from e_0_20250305_plot_utils_v1_00 import setup_plot_style, save_plot

# def plot_efficiency_ratio(optimizer_data, run_dir):
#     """Create efficiency ratio plot showing performance per computation cost with properly sized elements"""
    
#     # Define pastel colors directly
#     pastel_colors = ["SkyBlue", "Coral"]  # Named colors or hex codes can be used
    
#     # Create a figure with a white background
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), dpi=80)
#     ax1.set_facecolor("white")
#     ax2.set_facecolor("white")
    
#     optimizer_names = []
#     loss_improvements = []
#     time_ratios = []
#     memory_ratios = []
#     model_names = []
    
#     for optimizer, runs in optimizer_data.items():
#         for run in runs:
#             metrics = run["metrics"]
            
#             # Calculate loss improvement
#             if "loss_log" in metrics and len(metrics["loss_log"]) > 1:
#                 initial_loss = metrics["loss_log"][0]["loss"]
#                 final_loss = metrics["loss_log"][-1]["loss"]
#                 loss_improvement = initial_loss - final_loss
                
#                 # Get resource usage
#                 training_time = metrics.get("total_training_time", 1)
#                 if training_time is None or training_time <= 0:
#                     training_time = 1
                
#                 memory_usage = metrics.get("peak_memory_usage", 1)
#                 if memory_usage is None or memory_usage <= 0:
#                     memory_usage = 1
                
#                 # Calculate efficiency ratios
#                 time_ratio = loss_improvement / training_time
#                 memory_ratio = loss_improvement / memory_usage
                
#                 optimizer_names.append(optimizer.upper())
#                 loss_improvements.append(loss_improvement)
#                 time_ratios.append(time_ratio)
#                 memory_ratios.append(memory_ratio)
#                 model_names.append(run["model_dir"])
    
#     if optimizer_names:
#         df = pd.DataFrame({
#             "Optimizer": optimizer_names,
#             "Loss Improvement": loss_improvements,
#             "Time Efficiency": time_ratios,
#             "Memory Efficiency": memory_ratios,
#             "Model": model_names
#         })
        
#         # Create color mapping for consistent colors
#         optimizer_unique = df["Optimizer"].unique()
#         color_mapping = {optimizer: pastel_colors[i % len(pastel_colors)] 
#                         for i, optimizer in enumerate(optimizer_unique)}
        
#         # Plot time efficiency
#         positions1 = np.arange(len(df))
#         bar_width = 0.6
        
#         # First subplot - Time Efficiency
#         bars1 = ax1.bar(positions1, df["Time Efficiency"], 
#                 width=bar_width, 
#                 color=[color_mapping[opt] for opt in df["Optimizer"]],
#                 edgecolor='white',
#                 linewidth=1)
        
#         ax1.set_title("Loss Improvement per Training Time", fontsize=18, color="dimgrey")
#         ax1.set_ylabel("Loss Reduction / Minute", fontsize=14, color="dimgrey")
#         ax1.set_xticks(positions1)
#         ax1.set_xticklabels(df["Optimizer"], fontsize=14, color="darkgray")
        
#         # Add value labels on top of bars
#         for bar in bars1:
#             height = bar.get_height()
#             ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
#                     f'{height:.5f}',
#                     ha='center', va='bottom', fontsize=12, color='dimgrey')
        
#         # Second subplot - Memory Efficiency
#         positions2 = np.arange(len(df))
#         bars2 = ax2.bar(positions2, df["Memory Efficiency"], 
#                 width=bar_width, 
#                 color=[color_mapping[opt] for opt in df["Optimizer"]],
#                 edgecolor='white',
#                 linewidth=1)
        
#         ax2.set_title("Loss Improvement per Memory Usage", fontsize=18, color="dimgrey")
#         ax2.set_ylabel("Loss Reduction / MB", fontsize=14, color="dimgrey")
#         ax2.set_xticks(positions2)
#         ax2.set_xticklabels(df["Optimizer"], fontsize=14, color="darkgray")
        
#         # Add value labels on top of bars
#         for bar in bars2:
#             height = bar.get_height()
#             ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
#                     f'{height:.7f}',
#                     ha='center', va='bottom', fontsize=12, color='dimgrey')
        
#         # Apply styling to both axes
#         for ax in [ax1, ax2]:
#             # Darker grey gridlines (only on y-axis)
#             ax.grid(True, axis='y', color='dimgray', linestyle=':', alpha=0.8)
            
#             # Thicker x and y axis lines
#             ax.spines['bottom'].set_color('dimgray')
#             ax.spines['bottom'].set_linewidth(4.0)
#             ax.spines['left'].set_color('dimgray')
#             ax.spines['left'].set_linewidth(4.0)
#             # Lighter top and right axis lines
#             ax.spines['top'].set_color('silver')
#             ax.spines['right'].set_color('silver')
            
#             # Adjust tick label size and color
#             ax.tick_params(axis='y', labelsize=14, labelcolor="darkgray")
        
#         plt.tight_layout(pad=2.0)
        
#         # Save the plot
#         save_plot(fig, run_dir, "e_12_efficiency_ratio_comparison.png", dpi=80)
#     else:
#         plt.close(fig)
#         fig = None
        
#     return fig