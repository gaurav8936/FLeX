#!/usr/bin/env python3
# Steps to Target Loss Comparison Plot

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from e_0_20250305_plot_utils_v1_00 import save_plot

def plot_steps_to_target(optimizer_data, run_dir):
    """Create steps to target loss comparison plot with properly sized elements"""
    
    # Define pastel colors directly
    pastel_colors = ["SkyBlue", "Coral"]  # Named colors or hex codes can be used
    
    # Create a figure with a white background
    plt.figure(figsize=(12, 8), dpi=80)
    ax = plt.gca()
    ax.set_facecolor("white")  # Ensures the plot area is white
    
    # Metrics directory - same as where we found the JSON files
    # metrics_dir = os.path.dirname(os.path.dirname(run_dir))
    metrics_dir = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/11.Round_2_Sophia_Adam_**TRAINING**_APPS/03.json_csv_logs_metrics_plot_scope_only"

    # After defining metrics_dir
    print(f"Looking for CSV files in directory: {metrics_dir}")
    print(f"Optimizer data: {list(optimizer_data.keys())}")

    
    # Dynamically determine target losses based on minimum loss values
    all_losses = []
    optimizer_data_with_loss = {}
    
    for optimizer, runs in optimizer_data.items():
        optimizer_data_with_loss[optimizer] = []
        
        for run in runs:
            timestamp = run.get("timestamp")
            if timestamp:
                # Construct CSV file path
                loss_log_file = f"{optimizer}_{timestamp}_loss_log.csv"
                loss_log_path = os.path.join(metrics_dir, loss_log_file)
                
                if os.path.exists(loss_log_path):
                    print(f"Found loss log file: {loss_log_path}")
                    
                    # Read CSV file
                    try:
                        loss_df = pd.read_csv(loss_log_path)
                        
                        if 'step' in loss_df.columns and 'loss' in loss_df.columns:
                            # Store the data for this optimizer
                            optimizer_data_with_loss[optimizer].append({
                                "loss_df": loss_df,
                                "model_dir": run.get("model_dir", "unknown")
                            })
                            
                            # Add losses to calculate percentiles
                            all_losses.extend(loss_df['loss'].tolist())
                    except Exception as e:
                        print(f"Error reading loss log file {loss_log_path}: {e}")
    
    # If we collected loss data
    if all_losses:
        # Determine target losses based on percentiles of all observed losses
        # This ensures we have meaningful targets that are achievable
        all_losses.sort()
        
        # Calculate percentiles (20%, 40%, 60%, 80%)
        percentiles = [20, 40, 60, 80]
        target_losses = [np.percentile(all_losses, p) for p in percentiles]
        
        # Round to 2 decimal places for display
        target_losses = [round(t, 2) for t in target_losses]
        print(f"Calculated target losses: {target_losses}")
        
        # Store the results for plotting
        plot_data = []
        
        # For each optimizer, find steps to reach targets
        for optimizer, runs in optimizer_data_with_loss.items():
            for run_data in runs:
                loss_df = run_data["loss_df"]
                model_dir = run_data["model_dir"]
                
                # For each target, find the first step that reaches it
                for target in target_losses:
                    # Find rows where loss is less than or equal to target
                    below_target = loss_df[loss_df['loss'] <= target]
                    
                    if not below_target.empty:
                        # Get the first step that reached this target
                        first_step = below_target['step'].min()
                        
                        plot_data.append({
                            "Target Loss": target,
                            "Optimizer": optimizer.upper(),
                            "Steps": first_step,
                            "Model": model_dir
                        })
        
        # Create DataFrame from collected data
        if plot_data:
            df = pd.DataFrame(plot_data)
            
            # Create pivot table
            pivot_df = df.pivot_table(index="Target Loss", columns="Optimizer", values="Steps", aggfunc="first")
            pivot_df = pivot_df.sort_index(ascending=False)
            
            # Map the optimizers to our consistent pastel colors
            color_map = {col: pastel_colors[i % len(pastel_colors)] for i, col in enumerate(pivot_df.columns)}
            
            # Clear the current axis
            ax.clear()
            
            # Set up positions for horizontal bars
            y_positions = range(len(pivot_df.index))
            y_labels = [str(idx) for idx in pivot_df.index]
            
            # Plot each optimizer's bars
            bar_width = 0.35
            for i, column in enumerate(pivot_df.columns):
                # Get data for this optimizer
                data = pivot_df[column].values
                
                # Handle NaN values (targets that weren't reached)
                data = np.nan_to_num(data, nan=0)
                
                # Plot horizontal bars
                ax.barh([p + (i * bar_width) for p in y_positions], data, 
                       color=color_map[column], 
                       height=bar_width,
                       edgecolor='white',
                       linewidth=1,
                       label=column)
            
            # Set the y-tick labels to the target loss values
            ax.set_yticks([p + bar_width/2 for p in y_positions])
            ax.set_yticklabels(y_labels)
            
            plt.title("Steps to Reach Target Loss", fontsize=18, color="dimgrey")
            plt.xlabel("Number of Steps", fontsize=14, color="dimgrey")
            plt.ylabel("Target Loss", fontsize=14, color="dimgrey")
            
            # Darker grey gridlines (only on x-axis)
            plt.grid(True, axis='x', color='dimgray', linestyle=':', alpha=0.8)
            
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
            save_plot(plt, run_dir, "e_6_steps_to_target_loss.png", dpi=80)
            return plt.gcf()
    
    print("No data found for steps to target loss plot")
    plt.close()
    return None

# def plot_steps_to_target(optimizer_data, run_dir):
#     """Create steps to target loss comparison plot with properly sized elements"""
    
#     # Define pastel colors directly
#     pastel_colors = ["SkyBlue", "Coral"]  # Named colors or hex codes can be used
    
#     # Create a figure with a white background
#     plt.figure(figsize=(12, 8), dpi=80)
#     ax = plt.gca()
#     ax.set_facecolor("white")  # Ensures the plot area is white
    
#     target_losses = []
#     optimizer_names = []
#     steps_to_target = []
#     model_names = []
    
#     for optimizer, runs in optimizer_data.items():
#         for run in runs:
#             metrics = run["metrics"]
#             if "steps_to_target_loss" in metrics:
#                 for target_loss, steps in metrics["steps_to_target_loss"].items():
#                     if steps is not None:
#                         target_losses.append(float(target_loss))
#                         optimizer_names.append(optimizer.upper())
#                         steps_to_target.append(steps)
#                         model_names.append(run["model_dir"])
    
#     if target_losses:
#         df = pd.DataFrame({
#             "Target Loss": target_losses,
#             "Optimizer": optimizer_names,
#             "Steps": steps_to_target,
#             "Model": model_names
#         })
        
#         pivot_df = df.pivot_table(index="Target Loss", columns="Optimizer", values="Steps", aggfunc="first")
#         pivot_df = pivot_df.sort_index(ascending=False)
        
#         # Map the optimizers to our consistent pastel colors
#         color_map = {col: pastel_colors[i % len(pastel_colors)] for i, col in enumerate(pivot_df.columns)}
        
#         # Fix: Use matplotlib's ax.barh directly instead of pandas plot with problematic parameters
#         # First, clear the current axis
#         ax.clear()
        
#         # Set up positions for horizontal bars
#         y_positions = range(len(pivot_df.index))
#         y_labels = [str(idx) for idx in pivot_df.index]
        
#         # Keep track of left positions for stacking bars
#         lefts = np.zeros(len(pivot_df.index))
        
#         # Plot each optimizer's bars
#         for i, column in enumerate(pivot_df.columns):
#             # Get data for this optimizer
#             data = pivot_df[column].values
            
#             # Plot horizontal bars
#             ax.barh(y_positions, data, 
#                    color=color_map[column], 
#                    height=0.7,
#                    edgecolor='white',
#                    linewidth=1,
#                    label=column)
        
#         # Set the y-tick labels to the target loss values
#         ax.set_yticks(y_positions)
#         ax.set_yticklabels(y_labels)
        
#         plt.title("Steps to Reach Target Loss", fontsize=18, color="dimgrey")
#         plt.xlabel("Number of Steps", fontsize=14, color="dimgrey")
#         plt.ylabel("Target Loss", fontsize=14, color="dimgrey")
        
#         # Darker grey gridlines (only on x-axis)
#         plt.grid(True, axis='x', color='dimgray', linestyle=':', alpha=0.8)
        
#         # Thicker x and y axis lines
#         plt.gca().spines['bottom'].set_color('dimgray')  
#         plt.gca().spines['bottom'].set_linewidth(4.0)  
#         plt.gca().spines['left'].set_color('dimgray')  
#         plt.gca().spines['left'].set_linewidth(4.0)  
#         # Lighter top and right axis lines
#         plt.gca().spines['top'].set_color('silver')  
#         plt.gca().spines['right'].set_color('silver')  
        
#         # Make legend compact
#         plt.legend(fontsize=14, loc='upper right', framealpha=0.7)
        
#         # Adjust tick label size and color
#         plt.xticks(fontsize=14, color="darkgray")
#         plt.yticks(fontsize=14, color="darkgray")
        
#         plt.tight_layout(pad=0.5)
        
    #     # Save the plot
    #     save_plot(plt, run_dir, "e_6_steps_to_target_loss.png", dpi=80)
    # else:
    #     plt.close()
    #     return None
        
    # return plt.gcf()