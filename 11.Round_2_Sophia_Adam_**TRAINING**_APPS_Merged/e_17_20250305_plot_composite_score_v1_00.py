#!/usr/bin/env python3
# Weighted Composite Score Plot

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from e_0_20250305_plot_utils_v1_00 import get_linestyle, save_plot

def plot_composite_score(optimizer_data, run_dir, 
                        loss_weight=0.7, time_weight=0.15, memory_weight=0.15):
    """Create weighted composite score combining multiple metrics"""
    # Define pastel colors directly
    pastel_colors = ["SkyBlue", "Coral"]  # Named colors or hex codes
    component_colors = {"Loss Score": "SkyBlue", "Time Score": "Coral", "Memory Score": "LightGreen"}
    
    # First create the original 1x2 plot
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), dpi=80)
    
    # Create a separate 1x2 plot for log scale
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 8), dpi=80)
    
    # Set white background for all axes
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor("white")
    
    optimizer_names = []
    scores = []
    log_scores = []
    score_components = []
    log_score_components = []
    model_names = []
    
    # First, collect the base metrics for each optimizer
    all_data = []
    for optimizer, runs in optimizer_data.items():
        for run in runs:
            metrics = run["metrics"]
            
            # Get final loss
            final_loss = None
            if "loss_log" in metrics and metrics["loss_log"]:
                final_loss = metrics["loss_log"][-1]["loss"]
            
            # Get best validation loss if available
            val_loss = metrics.get("best_validation_loss", None)
            
            # Get training time and memory usage
            training_time = metrics.get("total_training_time", None)
            memory_usage = metrics.get("peak_memory_usage", None)
            
            # Debug print
            # print(f"\nDEBUG - Raw metrics for {optimizer}:")
            # print(f"  final_loss: {final_loss}")
            # print(f"  val_loss: {val_loss}")
            # print(f"  training_time: {training_time}")
            # print(f"  memory_usage: {memory_usage}")
            
            if all(metric is not None for metric in [final_loss, training_time, memory_usage]):
                all_data.append({
                    "optimizer": optimizer.upper(),
                    "model": os.path.basename(run["model_dir"]),
                    "final_loss": final_loss,
                    "val_loss": val_loss,
                    "training_time": training_time,
                    "memory_usage": memory_usage
                })
    
    # Normalize metrics and compute the composite score
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Debug print the collected data
        # print("\nDEBUG - Collected data for all optimizers:")
        # print(df)
        
        # -------- ORIGINAL LINEAR NORMALIZATION --------
        # Normalize each metric to [0, 1] scale (lower is better for all metrics)
        for column in ["final_loss", "training_time", "memory_usage"]:
            if df[column].max() > df[column].min():
                df[f"{column}_norm"] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
            else:
                df[f"{column}_norm"] = 0  # If all values are the same
        
        # Debug print normalized values
        # print("\nDEBUG - Normalized values:")
        # print(df[["optimizer", "final_loss_norm", "training_time_norm", "memory_usage_norm"]])
        
        # Compute the weighted composite score (lower is better)
        df["composite_score"] = (
            loss_weight * df["final_loss_norm"] +
            time_weight * df["training_time_norm"] +
            memory_weight * df["memory_usage_norm"]
        )
        
        # Convert to a "higher is better" score by inverting
        df["score"] = 1 - df["composite_score"]
        
        # Debug print the final scores
        # print("\nDEBUG - Final scores:")
        # print(df[["optimizer", "composite_score", "score"]])
        
        # -------- ENHANCED LOG SCALE NORMALIZATION --------
        # Create a copy of the DataFrame for log-scale normalization
        df_log = df.copy()
        
        # Instead of using log normalization directly, apply a more distinct approach for loss
        # that emphasizes the logarithmic differences
        if len(df_log) > 1:  # Only proceed if we have multiple optimizers to compare
            # Calculate the ratio between the largest and smallest loss
            loss_min = df_log["final_loss"].min()
            loss_max = df_log["final_loss"].max()
            
            if loss_min > 0 and loss_max > loss_min:
                # Take log ratio and scale to make the difference more pronounced
                log_ratio = np.log(loss_max / loss_min)
                
                # Apply a more aggressive scaling for small differences
                scaling_factor = 1.0
                if log_ratio < 1.0:
                    scaling_factor = 2.0
                
                # The optimizer with the minimum loss gets norm=0, the others get scaled values
                for i, row in df_log.iterrows():
                    if row["final_loss"] == loss_min:
                        df_log.at[i, "final_loss_log_norm"] = 0.0
                    else:
                        # Use a more dramatic scaling to highlight the difference
                        ratio = row["final_loss"] / loss_min
                        df_log.at[i, "final_loss_log_norm"] = min(1.0, np.log(ratio) / log_ratio * scaling_factor)
            else:
                df_log["final_loss_log_norm"] = df_log["final_loss_norm"]
        else:
            df_log["final_loss_log_norm"] = 0.0
        
        # Use a more balanced approach for time and memory
        for column in ["training_time", "memory_usage"]:
            # Get the original normalized values
            df_log[f"{column}_log_norm"] = df_log[f"{column}_norm"]
            
            # If the difference is small, reduce its impact in the composite score
            min_val = df_log[column].min()
            max_val = df_log[column].max()
            
            if min_val > 0 and max_val > min_val:
                # Calculate percentage difference
                percent_diff = (max_val - min_val) / min_val
                
                # If the difference is less than 10%, reduce its impact by scaling down
                if percent_diff < 0.10:  # 10% difference
                    scaling_factor = percent_diff / 0.10  # Proportionally reduce impact
                    df_log[f"{column}_log_norm"] *= scaling_factor
        
        # Debug print log-normalized values
        # print("\nDEBUG - Enhanced Log-Normalized values:")
        # print(df_log[["optimizer", "final_loss_log_norm", "training_time_log_norm", "memory_usage_log_norm"]])
        
        # Compute the weighted log-scale composite score
        df_log["log_composite_score"] = (
            loss_weight * df_log["final_loss_log_norm"] +
            time_weight * df_log["training_time_log_norm"] +
            memory_weight * df_log["memory_usage_log_norm"]
        )
        
        # Convert to a "higher is better" score by inverting
        df_log["log_score"] = 1 - df_log["log_composite_score"]
        
        # Debug print the final log scores
        # print("\nDEBUG - Final log scores:")
        # print(df_log[["optimizer", "log_composite_score", "log_score"]])
        
        # -------- COLLECT DATA FOR PLOTTING --------
        # For original linear scale
        for _, row in df.iterrows():
            optimizer_names.append(row["optimizer"])
            scores.append(row["score"])
            
            # Calculate component scores
            loss_score = loss_weight * (1 - row["final_loss_norm"])
            time_score = time_weight * (1 - row["training_time_norm"])
            memory_score = memory_weight * (1 - row["memory_usage_norm"])
            
            # Debug print component scores
            # print(f"\nDEBUG - Component scores for {row['optimizer']}:")
            # print(f"  Loss Score: {loss_score}")
            # print(f"  Time Score: {time_score}")
            # print(f"  Memory Score: {memory_score}")
            
            # Collect the weighted component values for the stacked bar chart
            score_components.append({
                "optimizer": row["optimizer"],
                "model": row["model"],
                "Loss Score": loss_score,
                "Time Score": time_score,
                "Memory Score": memory_score
            })
            
            model_names.append(row["model"])
        
        # For log scale
        for _, row in df_log.iterrows():
            log_scores.append(row["log_score"])
            
            # Calculate log component scores
            log_loss_score = loss_weight * (1 - row["final_loss_log_norm"])
            log_time_score = time_weight * (1 - row["training_time_log_norm"])
            log_memory_score = memory_weight * (1 - row["memory_usage_log_norm"])
            
            # Debug print log component scores
            # print(f"\nDEBUG - Enhanced Log component scores for {row['optimizer']}:")
            # print(f"  Log Loss Score: {log_loss_score}")
            # print(f"  Log Time Score: {log_time_score}")
            # print(f"  Log Memory Score: {log_memory_score}")
            
            # Collect the weighted log component values for the stacked bar chart
            log_score_components.append({
                "optimizer": row["optimizer"],
                "model": row["model"],
                "Loss Score": log_loss_score,
                "Time Score": log_time_score,
                "Memory Score": log_memory_score
            })
        
        # Create the color map for optimizers
        optimizer_color_map = {}
        for i, opt in enumerate(df["optimizer"].unique()):
            optimizer_color_map[opt] = pastel_colors[i % len(pastel_colors)]
        
        # -------- PLOT 1: LINEAR SCALE COMPOSITE SCORE --------
        ax1_bars = sns.barplot(x="optimizer", y="score", data=df, ax=ax1, 
                             palette=optimizer_color_map)
                  
        # Customize bar edge color and width
        for i, bar in enumerate(ax1_bars.patches):
            bar.set_edgecolor('dimgray')
            bar.set_linewidth(1.5)
        
        ax1.set_title(f"Weighted Composite Score\n(Loss:{loss_weight} Time:{time_weight} Memory:{memory_weight})", 
                    fontsize=18, color="dimgrey")
        ax1.set_xlabel("Optimizer", fontsize=14, color="dimgrey")
        ax1.set_ylabel("Score (higher is better)", fontsize=14, color="dimgrey")
        
        # Darker grey gridlines
        ax1.grid(True, axis='y', color='dimgray', linestyle=':', alpha=0.8)
        
        # Set y-axis limit to make comparison easier
        ax1.set_ylim(0, 1.1)
        
        # Add value labels on top of bars with improved formatting
        for i, v in enumerate(scores):
            ax1.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=14, color="dimgrey")
        
        # Thicker x and y axis lines for first figure
        for ax in [ax1, ax2]:
            ax.spines['bottom'].set_color('dimgray')
            ax.spines['bottom'].set_linewidth(4.0)
            ax.spines['left'].set_color('dimgray')
            ax.spines['left'].set_linewidth(4.0)
            ax.spines['top'].set_color('silver')
            ax.spines['right'].set_color('silver')
            ax.tick_params(axis='both', labelsize=14, colors="darkgray")
        
        # -------- PLOT 2: LINEAR SCALE COMPONENT BREAKDOWN --------
        # Create a DataFrame for the stacked bar chart
        components_df = pd.DataFrame(score_components)
        
        # Debug print the component contributions data
        # print("\nDEBUG - Component contributions for stacked bar chart:")
        # print(components_df)
        
        components_melted = pd.melt(components_df, 
                                   id_vars=["optimizer", "model"],
                                   value_vars=["Loss Score", "Time Score", "Memory Score"],
                                   var_name="Component", value_name="Value")
        
        # Debug print melted data
        # print("\nDEBUG - Melted component data:")
        # print(components_melted)
        
        # Plot the stacked bar chart of component contributions
        ax2_bars = sns.barplot(x="optimizer", y="Value", hue="Component", data=components_melted, 
                             ax=ax2, palette=component_colors)
                  
        # Customize bar edge color and width for stacked bars
        for bar in ax2_bars.patches:
            bar.set_edgecolor('dimgray')
            bar.set_linewidth(1.5)
            
        ax2.set_title("Score Component Breakdown", fontsize=18, color="dimgrey")
        ax2.set_xlabel("Optimizer", fontsize=14, color="dimgrey")
        ax2.set_ylabel("Component Contribution", fontsize=14, color="dimgrey")
        
        # Darker grey gridlines
        ax2.grid(True, axis='y', color='dimgray', linestyle=':', alpha=0.8)
        
        # Enhance legend for first component plot
        legend1 = ax2.legend(title="Component", fontsize=14, framealpha=0.7)
        legend1.get_title().set_fontsize(14)
        legend1.get_title().set_color("dimgrey")
        
        # Save the original linear scale plots
        plt.figure(fig1.number)
        plt.tight_layout(pad=0.5)
        save_plot(plt, run_dir, "e_17_composite_score.png", dpi=80)
        
        # -------- PLOT 3: ENHANCED LOG SCALE COMPOSITE SCORE --------
        ax3_bars = sns.barplot(x="optimizer", y="log_score", data=df_log, ax=ax3, 
                             palette=optimizer_color_map)
                  
        # Customize bar edge color and width
        for i, bar in enumerate(ax3_bars.patches):
            bar.set_edgecolor('dimgray')
            bar.set_linewidth(1.5)
        
        ax3.set_title(f"Enhanced Perspective: Weighted Score\n(Emphasizing Loss Difference)", 
                    fontsize=18, color="dimgrey")
        ax3.set_xlabel("Optimizer", fontsize=14, color="dimgrey")
        ax3.set_ylabel("Score (higher is better)", fontsize=14, color="dimgrey")
        
        # Darker grey gridlines
        ax3.grid(True, axis='y', color='dimgray', linestyle=':', alpha=0.8)
        
        # Set y-axis limit to make comparison easier
        ax3.set_ylim(0, 1.1)
        
        # Add value labels on top of bars with improved formatting
        for i, v in enumerate(log_scores):
            ax3.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=14, color="dimgrey")
        
        # -------- PLOT 4: ENHANCED LOG SCALE COMPONENT BREAKDOWN --------
        # Create a DataFrame for the log stacked bar chart
        log_components_df = pd.DataFrame(log_score_components)
        
        # Debug print the log component contributions data
        print("\nDEBUG - Log component contributions for stacked bar chart:")
        # print(log_components_df)
        
        log_components_melted = pd.melt(log_components_df, 
                                      id_vars=["optimizer", "model"],
                                      value_vars=["Loss Score", "Time Score", "Memory Score"],
                                      var_name="Component", value_name="Value")
        
        # Debug print melted log data
        # print("\nDEBUG - Melted log component data:")
        # print(log_components_melted)
        
        # Plot the log stacked bar chart of component contributions
        ax4_bars = sns.barplot(x="optimizer", y="Value", hue="Component", data=log_components_melted, 
                             ax=ax4, palette=component_colors)
                  
        # Customize bar edge color and width for stacked bars
        for bar in ax4_bars.patches:
            bar.set_edgecolor('dimgray')
            bar.set_linewidth(1.5)
            
        ax4.set_title("Enhanced Perspective: Component Breakdown", fontsize=18, color="dimgrey")
        ax4.set_xlabel("Optimizer", fontsize=14, color="dimgrey")
        ax4.set_ylabel("Component Contribution", fontsize=14, color="dimgrey")
        
        # Darker grey gridlines
        ax4.grid(True, axis='y', color='dimgray', linestyle=':', alpha=0.8)
        
        # Thicker x and y axis lines for second figure
        for ax in [ax3, ax4]:
            ax.spines['bottom'].set_color('dimgray')
            ax.spines['bottom'].set_linewidth(4.0)
            ax.spines['left'].set_color('dimgray')
            ax.spines['left'].set_linewidth(4.0)
            ax.spines['top'].set_color('silver')
            ax.spines['right'].set_color('silver')
            ax.tick_params(axis='both', labelsize=14, colors="darkgray")
        
        # Enhance legend for second component plot
        legend2 = ax4.legend(title="Component", fontsize=14, framealpha=0.7)
        legend2.get_title().set_fontsize(14)
        legend2.get_title().set_color("dimgrey")
        
        # Save the log scale plots separately
        plt.figure(fig2.number)
        plt.tight_layout(pad=0.5)
        save_plot(plt, run_dir, "e_17_enhanced_perspective.png", dpi=80)
    else:
        plt.close(fig1)
        plt.close(fig2)
        fig1 = None
        fig2 = None
        
    return fig1