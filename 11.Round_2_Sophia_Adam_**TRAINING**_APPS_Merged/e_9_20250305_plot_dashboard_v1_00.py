#!/usr/bin/env python3
# Summary Dashboard Plot

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from e_0_20250305_plot_utils_v1_00 import setup_plot_style, get_linestyle, extract_metric_series, save_plot

def plot_dashboard(optimizer_data, run_dir):
    """Create summary dashboard visualization with properly sized elements"""
    
    # Define pastel colors directly
    pastel_colors = ["SkyBlue", "Coral"]  # Named colors or hex codes can be used
    
    # Create the figure
    fig = plt.figure(figsize=(15, 14), dpi=80)
    # Adjust GridSpec to make right column plots same size as left column
    # Use 2 columns with equal width instead of 3 columns with unequal width
    grid = plt.GridSpec(3, 2, figure=fig, wspace=0.3, hspace=0.4)
    
    # Style settings
    title_fontsize = 16
    axis_label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 12
    title_color = "dimgrey"
    label_color = "dimgrey"
    tick_color = "darkgray"
    grid_color = "dimgray"
    grid_alpha = 0.8
    grid_style = ":"
    line_width = 3
    
    # Styling function for each subplot
    def style_axes(ax):
        ax.set_facecolor("white")
        ax.grid(True, color=grid_color, linestyle=grid_style, alpha=grid_alpha)
        
        # Thicker x and y axis lines
        ax.spines['bottom'].set_color('dimgray')
        ax.spines['bottom'].set_linewidth(4.0)
        ax.spines['left'].set_color('dimgray')
        ax.spines['left'].set_linewidth(4.0)
        # Lighter top and right axis lines
        ax.spines['top'].set_color('silver')
        ax.spines['right'].set_color('silver')
        
        # Adjust tick label size and color
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, labelcolor=tick_color)
    
    # 1. Training Loss Plot (top left)
    ax1 = fig.add_subplot(grid[0, 0])
    for idx, (optimizer, runs) in enumerate(optimizer_data.items()):
        for run in runs:
            metrics = run["metrics"]
            if "loss_log" in metrics:
                steps, losses = extract_metric_series(metrics, "loss")
                
                # Apply smoothing
                if len(steps) > 5:
                    window_size = 5
                    weights = np.ones(window_size) / window_size
                    losses_smooth = np.convolve(losses, weights, mode='same')
                    losses_smooth[:window_size//2] = losses[:window_size//2]
                    losses_smooth[-window_size//2:] = losses[-window_size//2:]
                else:
                    losses_smooth = losses
                
                label = f"{optimizer.upper()}"
                ax1.plot(steps, losses_smooth, label=label, 
                        color=pastel_colors[idx % len(pastel_colors)],
                        linestyle=get_linestyle(optimizer),
                        linewidth=line_width)
    
    ax1.set_title("Training Loss", fontsize=title_fontsize, color=title_color)
    ax1.set_xlabel("Steps", fontsize=axis_label_fontsize, color=label_color)
    ax1.set_ylabel("Loss", fontsize=axis_label_fontsize, color=label_color)
    ax1.legend(fontsize=legend_fontsize, framealpha=0.7)
    style_axes(ax1)
    
    # 2. Validation Loss Plot (top right) - Updated to use CSV files like e_2
    ax2 = fig.add_subplot(grid[0, 1])
    for idx, (optimizer, runs) in enumerate(optimizer_data.items()):
        for run in runs:
            timestamp = run.get("timestamp")
            print(f"Processing optimizer: {optimizer}, timestamp: {timestamp}")
            
            # The JSON metrics file path should be included in the run data
            # Construct the validation loss file path from it
            metrics_dir = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/11.Round_2_Sophia_Adam_**TRAINING**_APPS/03.json_csv_logs_metrics_plot_scope_only"
            val_loss_file = f"{optimizer}_{timestamp}_val_loss.csv"
            val_loss_path = os.path.join(metrics_dir, val_loss_file)
            
            print(f"Looking for validation file: {val_loss_path}")
            
            # Check if the validation loss file exists
            if os.path.exists(val_loss_path):
                print(f"Found validation file: {val_loss_path}")
                
                # Read the validation loss data
                val_data = pd.read_csv(val_loss_path)
                
                # Extract epochs and validation losses
                val_epochs = val_data['epoch'].values
                val_losses = val_data['val_loss'].values
                
                # Use shorter labels
                label = f"{optimizer.upper()}"
                
                # Assign pastel colors directly
                line_color = pastel_colors[idx % len(pastel_colors)]
                ax2.plot(val_epochs, val_losses, marker='o', label=label, 
                         color=line_color,
                         linewidth=line_width)
            else:
                print(f"Validation file not found: {val_loss_path}")
                
                # Fall back to using best_validation_loss if available
                metrics = run["metrics"]
                if 'best_validation_loss' in metrics:
                    best_val = metrics['best_validation_loss']
                    print(f"Using best_validation_loss: {best_val}")
                    
                    # If we don't have detailed validation data, just use the best validation loss
                    val_epochs = [metrics.get("num_epochs", 1)]
                    val_losses = [best_val]
                    
                    line_color = pastel_colors[idx % len(pastel_colors)]
                    ax2.plot(val_epochs, val_losses, marker='o', label=optimizer.upper(), 
                            color=line_color, linewidth=line_width)
    
    ax2.set_title("Validation Loss", fontsize=title_fontsize, color=title_color)
    ax2.set_xlabel("Epoch", fontsize=axis_label_fontsize, color=label_color)
    ax2.set_ylabel("Loss", fontsize=axis_label_fontsize, color=label_color)
    ax2.legend(fontsize=legend_fontsize, framealpha=0.7)
    style_axes(ax2)
    
    # 3. Perplexity Plot (middle left)
    ax3 = fig.add_subplot(grid[1, 0])
    for idx, (optimizer, runs) in enumerate(optimizer_data.items()):
        for run in runs:
            metrics = run["metrics"]
            if "perplexity_log" in metrics:
                steps, perplexities = extract_metric_series(metrics, "perplexity")
                
                if steps and perplexities:
                    # Apply smoothing
                    if len(steps) > 5:
                        window_size = 5
                        weights = np.ones(window_size) / window_size
                        perplexities_smooth = np.convolve(perplexities, weights, mode='same')
                        perplexities_smooth[:window_size//2] = perplexities[:window_size//2]
                        perplexities_smooth[-window_size//2:] = perplexities[-window_size//2:]
                    else:
                        perplexities_smooth = perplexities
                    
                    label = f"{optimizer.upper()}"
                    ax3.plot(steps, perplexities_smooth, label=label, 
                           color=pastel_colors[idx % len(pastel_colors)],
                           linestyle=get_linestyle(optimizer),
                           linewidth=line_width)
    
    ax3.set_title("Perplexity", fontsize=title_fontsize, color=title_color)
    ax3.set_xlabel("Steps", fontsize=axis_label_fontsize, color=label_color)
    ax3.set_ylabel("Perplexity", fontsize=axis_label_fontsize, color=label_color)
    ax3.legend(fontsize=legend_fontsize, framealpha=0.7)
    style_axes(ax3)
    
    # 4. Training Stability Plot (middle right)
    ax4 = fig.add_subplot(grid[1, 1])
    for idx, (optimizer, runs) in enumerate(optimizer_data.items()):
        for run in runs:
            metrics = run["metrics"]
            if "loss_variance_log" in metrics:
                steps, variances = extract_metric_series(metrics, "loss_variance")
                
                if steps and variances:
                    # Apply smoothing
                    if len(steps) > 5:
                        window_size = 5
                        weights = np.ones(window_size) / window_size
                        variances_smooth = np.convolve(variances, weights, mode='same')
                        variances_smooth[:window_size//2] = variances[:window_size//2]
                        variances_smooth[-window_size//2:] = variances[-window_size//2:]
                    else:
                        variances_smooth = variances
                    
                    label = f"{optimizer.upper()}"
                    ax4.plot(steps, variances_smooth, label=label, 
                           color=pastel_colors[idx % len(pastel_colors)],
                           linestyle=get_linestyle(optimizer),
                           linewidth=line_width)
    
    ax4.set_title("Training Stability", fontsize=title_fontsize, color=title_color)
    ax4.set_xlabel("Steps", fontsize=axis_label_fontsize, color=label_color)
    ax4.set_ylabel("Loss Variance", fontsize=axis_label_fontsize, color=label_color)
    ax4.legend(fontsize=legend_fontsize, framealpha=0.7)
    style_axes(ax4)
    
    # 5. Performance Metrics (bottom)
    ax5 = fig.add_subplot(grid[2, :])
    style_axes(ax5)
    
    # Collect data for performance metrics
    optimizer_names = []
    training_times = []
    memory_usages = []
    best_val_losses = []
    model_names = []
    
    for optimizer, runs in optimizer_data.items():
        for run in runs:
            metrics = run["metrics"]
            optimizer_names.append(optimizer.upper())
            
            # Training time (minutes)
            training_time = metrics.get("total_training_time", 0)
            if training_time is None:
                training_time = 0
            training_times.append(training_time)
            
            # Memory usage (GB)
            memory_usage = metrics.get("peak_memory_usage", 0)
            if memory_usage is None:
                memory_usage = 0
            memory_usages.append(memory_usage / 1024)  # Convert to GB
            
            # Best validation loss
            val_loss = metrics.get("best_validation_loss", 0)
            if val_loss is None:
                val_loss = 0
            best_val_losses.append(val_loss)
            
            model_names.append(run["model_dir"])
    
    metrics_df = pd.DataFrame({
        "Optimizer": optimizer_names,
        "Training Time (min)": training_times,
        "Peak Memory (GB)": memory_usages,
        "Best Val Loss": best_val_losses,
        "Model": model_names
    })
    
    # Map optimizers to consistent colors
    color_map = {optimizer: pastel_colors[i % len(pastel_colors)] 
                for i, optimizer in enumerate(metrics_df["Optimizer"].unique())}
    
    # Plot the bar chart
    ax = metrics_df.set_index("Optimizer").plot(kind="bar", ax=ax5, 
                                          color=[color_map.get(opt, "gray") for opt in metrics_df["Optimizer"]],
                                          edgecolor='white',
                                          linewidth=1,
                                          width=0.7)
    
    # Add values on top of each bar
    for container in ax.containers:
        ax5.bar_label(container, fmt='%.2f', fontsize=10, color='dimgray')
    
    ax5.set_title("Performance Metrics Comparison", fontsize=title_fontsize, color=title_color)
    ax5.set_ylabel("Value", fontsize=axis_label_fontsize, color=label_color)
    
    # Add main title
    plt.suptitle("Optimizer Comparison Dashboard", fontsize=20, color=title_color, y=0.98)
    
    # Fix tight_layout warning by using figure-level adjustments instead
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)
    
    # Save the plot
    save_plot(fig, run_dir, "e_9_optimizer_comparison_dashboard.png", dpi=80)
    
    return fig



# #!/usr/bin/env python3
# # Summary Dashboard Plot

# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# from e_0_20250305_plot_utils_v1_00 import setup_plot_style, get_linestyle, extract_metric_series, save_plot

# def plot_dashboard(optimizer_data, run_dir):
#     """Create summary dashboard visualization with properly sized elements"""
    
#     # Define pastel colors directly
#     pastel_colors = ["SkyBlue", "Coral"]  # Named colors or hex codes can be used
    
#     # Create the figure
#     fig = plt.figure(figsize=(15, 14), dpi=80)
#     # Adjust GridSpec to make right column plots same size as left column
#     # Use 2 columns with equal width instead of 3 columns with unequal width
#     grid = plt.GridSpec(3, 2, figure=fig, wspace=0.3, hspace=0.4)
    
#     # Style settings
#     title_fontsize = 16
#     axis_label_fontsize = 14
#     tick_fontsize = 12
#     legend_fontsize = 12
#     title_color = "dimgrey"
#     label_color = "dimgrey"
#     tick_color = "darkgray"
#     grid_color = "dimgray"
#     grid_alpha = 0.8
#     grid_style = ":"
#     line_width = 3
    
#     # Styling function for each subplot
#     def style_axes(ax):
#         ax.set_facecolor("white")
#         ax.grid(True, color=grid_color, linestyle=grid_style, alpha=grid_alpha)
        
#         # Thicker x and y axis lines
#         ax.spines['bottom'].set_color('dimgray')
#         ax.spines['bottom'].set_linewidth(4.0)
#         ax.spines['left'].set_color('dimgray')
#         ax.spines['left'].set_linewidth(4.0)
#         # Lighter top and right axis lines
#         ax.spines['top'].set_color('silver')
#         ax.spines['right'].set_color('silver')
        
#         # Adjust tick label size and color
#         ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, labelcolor=tick_color)
    
#     # 1. Training Loss Plot (top left)
#     ax1 = fig.add_subplot(grid[0, 0])
#     for idx, (optimizer, runs) in enumerate(optimizer_data.items()):
#         for run in runs:
#             metrics = run["metrics"]
#             if "loss_log" in metrics:
#                 steps, losses = extract_metric_series(metrics, "loss")
                
#                 # Apply smoothing
#                 if len(steps) > 5:
#                     window_size = 5
#                     weights = np.ones(window_size) / window_size
#                     losses_smooth = np.convolve(losses, weights, mode='same')
#                     losses_smooth[:window_size//2] = losses[:window_size//2]
#                     losses_smooth[-window_size//2:] = losses[-window_size//2:]
#                 else:
#                     losses_smooth = losses
                
#                 label = f"{optimizer.upper()}"
#                 ax1.plot(steps, losses_smooth, label=label, 
#                         color=pastel_colors[idx % len(pastel_colors)],
#                         linestyle=get_linestyle(optimizer),
#                         linewidth=line_width)
    
#     ax1.set_title("Training Loss", fontsize=title_fontsize, color=title_color)
#     ax1.set_xlabel("Steps", fontsize=axis_label_fontsize, color=label_color)
#     ax1.set_ylabel("Loss", fontsize=axis_label_fontsize, color=label_color)
#     ax1.legend(fontsize=legend_fontsize, framealpha=0.7)
#     style_axes(ax1)
    
#     # 2. Validation Loss Plot (top right) - Updated to use CSV files like e_2
#     ax2 = fig.add_subplot(grid[0, 1])
#     for idx, (optimizer, runs) in enumerate(optimizer_data.items()):
#         for run in runs:
#             timestamp = run.get("timestamp")
#             print(f"Processing optimizer: {optimizer}, timestamp: {timestamp}")
            
#             # The JSON metrics file path should be included in the run data
#             # Construct the validation loss file path from it
#             metrics_dir = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/11.Round_2_Sophia_Adam_**TRAINING**_APPS/03.json_csv_logs_metrics_plot_scope_only"
#             val_loss_file = f"{optimizer}_{timestamp}_val_loss.csv"
#             val_loss_path = os.path.join(metrics_dir, val_loss_file)
            
#             print(f"Looking for validation file: {val_loss_path}")
            
#             # Check if the validation loss file exists
#             if os.path.exists(val_loss_path):
#                 print(f"Found validation file: {val_loss_path}")
                
#                 # Read the validation loss data
#                 val_data = pd.read_csv(val_loss_path)
                
#                 # Extract epochs and validation losses
#                 val_epochs = val_data['epoch'].values
#                 val_losses = val_data['val_loss'].values
                
#                 # Use shorter labels
#                 label = f"{optimizer.upper()}"
                
#                 # Assign pastel colors directly
#                 line_color = pastel_colors[idx % len(pastel_colors)]
#                 ax2.plot(val_epochs, val_losses, marker='o', label=label, 
#                          color=line_color,
#                          linewidth=line_width)
#             else:
#                 print(f"Validation file not found: {val_loss_path}")
                
#                 # Fall back to using best_validation_loss if available
#                 metrics = run["metrics"]
#                 if 'best_validation_loss' in metrics:
#                     best_val = metrics['best_validation_loss']
#                     print(f"Using best_validation_loss: {best_val}")
                    
#                     # If we don't have detailed validation data, just use the best validation loss
#                     val_epochs = [metrics.get("num_epochs", 1)]
#                     val_losses = [best_val]
                    
#                     line_color = pastel_colors[idx % len(pastel_colors)]
#                     ax2.plot(val_epochs, val_losses, marker='o', label=optimizer.upper(), 
#                             color=line_color, linewidth=line_width)
    
#     ax2.set_title("Validation Loss", fontsize=title_fontsize, color=title_color)
#     ax2.set_xlabel("Epoch", fontsize=axis_label_fontsize, color=label_color)
#     ax2.set_ylabel("Loss", fontsize=axis_label_fontsize, color=label_color)
#     ax2.legend(fontsize=legend_fontsize, framealpha=0.7)
#     style_axes(ax2)
    
#     # 3. Perplexity Plot (middle left)
#     ax3 = fig.add_subplot(grid[1, 0])
#     for idx, (optimizer, runs) in enumerate(optimizer_data.items()):
#         for run in runs:
#             metrics = run["metrics"]
#             if "perplexity_log" in metrics:
#                 steps, perplexities = extract_metric_series(metrics, "perplexity")
                
#                 if steps and perplexities:
#                     # Apply smoothing
#                     if len(steps) > 5:
#                         window_size = 5
#                         weights = np.ones(window_size) / window_size
#                         perplexities_smooth = np.convolve(perplexities, weights, mode='same')
#                         perplexities_smooth[:window_size//2] = perplexities[:window_size//2]
#                         perplexities_smooth[-window_size//2:] = perplexities[-window_size//2:]
#                     else:
#                         perplexities_smooth = perplexities
                    
#                     label = f"{optimizer.upper()}"
#                     ax3.plot(steps, perplexities_smooth, label=label, 
#                            color=pastel_colors[idx % len(pastel_colors)],
#                            linestyle=get_linestyle(optimizer),
#                            linewidth=line_width)
    
#     ax3.set_title("Perplexity", fontsize=title_fontsize, color=title_color)
#     ax3.set_xlabel("Steps", fontsize=axis_label_fontsize, color=label_color)
#     ax3.set_ylabel("Perplexity", fontsize=axis_label_fontsize, color=label_color)
#     ax3.legend(fontsize=legend_fontsize, framealpha=0.7)
#     style_axes(ax3)
    
#     # 4. Training Stability Plot (middle right)
#     ax4 = fig.add_subplot(grid[1, 1])
#     for idx, (optimizer, runs) in enumerate(optimizer_data.items()):
#         for run in runs:
#             metrics = run["metrics"]
#             if "loss_variance_log" in metrics:
#                 steps, variances = extract_metric_series(metrics, "loss_variance")
                
#                 if steps and variances:
#                     # Apply smoothing
#                     if len(steps) > 5:
#                         window_size = 5
#                         weights = np.ones(window_size) / window_size
#                         variances_smooth = np.convolve(variances, weights, mode='same')
#                         variances_smooth[:window_size//2] = variances[:window_size//2]
#                         variances_smooth[-window_size//2:] = variances[-window_size//2:]
#                     else:
#                         variances_smooth = variances
                    
#                     label = f"{optimizer.upper()}"
#                     ax4.plot(steps, variances_smooth, label=label, 
#                            color=pastel_colors[idx % len(pastel_colors)],
#                            linestyle=get_linestyle(optimizer),
#                            linewidth=line_width)
    
#     ax4.set_title("Training Stability", fontsize=title_fontsize, color=title_color)
#     ax4.set_xlabel("Steps", fontsize=axis_label_fontsize, color=label_color)
#     ax4.set_ylabel("Loss Variance", fontsize=axis_label_fontsize, color=label_color)
#     ax4.legend(fontsize=legend_fontsize, framealpha=0.7)
#     style_axes(ax4)
    
#     # 5. Performance Metrics (bottom)
#     ax5 = fig.add_subplot(grid[2, :])
#     style_axes(ax5)
    
#     # Collect data for performance metrics
#     optimizer_names = []
#     training_times = []
#     memory_usages = []
#     best_val_losses = []
#     model_names = []
    
#     for optimizer, runs in optimizer_data.items():
#         for run in runs:
#             metrics = run["metrics"]
#             optimizer_names.append(optimizer.upper())
            
#             # Training time (minutes)
#             training_time = metrics.get("total_training_time", 0)
#             if training_time is None:
#                 training_time = 0
#             training_times.append(training_time)
            
#             # Memory usage (GB)
#             memory_usage = metrics.get("peak_memory_usage", 0)
#             if memory_usage is None:
#                 memory_usage = 0
#             memory_usages.append(memory_usage / 1024)  # Convert to GB
            
#             # Best validation loss
#             val_loss = metrics.get("best_validation_loss", 0)
#             if val_loss is None:
#                 val_loss = 0
#             best_val_losses.append(val_loss)
            
#             model_names.append(run["model_dir"])
    
#     metrics_df = pd.DataFrame({
#         "Optimizer": optimizer_names,
#         "Training Time (min)": training_times,
#         "Peak Memory (GB)": memory_usages,
#         "Best Val Loss": best_val_losses,
#         "Model": model_names
#     })
    
#     # Map optimizers to consistent colors
#     color_map = {optimizer: pastel_colors[i % len(pastel_colors)] 
#                 for i, optimizer in enumerate(metrics_df["Optimizer"].unique())}
    
#     metrics_df.set_index("Optimizer").plot(kind="bar", ax=ax5, 
#                                           color=[color_map.get(opt, "gray") for opt in metrics_df["Optimizer"]],
#                                           edgecolor='white',
#                                           linewidth=1,
#                                           width=0.7)
    
#     ax5.set_title("Performance Metrics Comparison", fontsize=title_fontsize, color=title_color)
#     ax5.set_ylabel("Value", fontsize=axis_label_fontsize, color=label_color)
    
#     # Add main title
#     plt.suptitle("Optimizer Comparison Dashboard", fontsize=20, color=title_color, y=0.98)
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
    
#     # Save the plot
#     save_plot(fig, run_dir, "e_9_optimizer_comparison_dashboard.png", dpi=80)
    
#     return fig