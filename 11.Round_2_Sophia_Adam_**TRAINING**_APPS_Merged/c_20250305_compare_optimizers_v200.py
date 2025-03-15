import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
from datetime import datetime

def load_metrics_file(file_path):
    """Load metrics from a JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_csv_file(file_path):
    """Load data from a CSV file"""
    return pd.read_csv(file_path)

def create_optimizer_comparison_plots(metrics_dir, output_dir=None):
    """Create visualizations comparing Adam and Sophia optimizers"""
    if output_dir is None:
        # output_dir = os.path.join(metrics_dir, "visualizations")
        output_dir = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/11.Round_2_Sophia_Adam_**TRAINING**_APPS/06.Visualizations"
    
    # Create base directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for this visualization run
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Create a new directory with the timestamp
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
 
    json_files = glob.glob(os.path.join(metrics_dir, "*_metrics.json"))
    csv_files = glob.glob(os.path.join(metrics_dir, "*_loss_log.csv"))
    val_files = glob.glob(os.path.join(metrics_dir, "*_val_loss.csv"))
    
    # Group files by optimizer
    optimizer_data = {}
    
    for json_file in json_files:
        metrics = load_metrics_file(json_file)
        optimizer = metrics["optimizer"].lower()
        timestamp = metrics["timestamp"]
        
        if optimizer not in optimizer_data:
            optimizer_data[optimizer] = []
        
        # Find corresponding CSV files
        loss_csv = None
        val_csv = None
        
        for csv_file in csv_files:
            if timestamp in csv_file:
                loss_csv = csv_file
                break
        
        for val_file in val_files:
            if timestamp in val_file:
                val_csv = val_file
                break
        
        optimizer_data[optimizer].append({
            "metrics": metrics,
            "loss_csv": loss_csv,
            "val_csv": val_csv,
            "timestamp": timestamp,
            "model_dir": metrics.get("model_dir", "unknown")
        })
    
    # Set up style
    plt.style.use('ggplot')
    sns.set(font_scale=1.2)
    colors = {"ADAMW": "#1f77b4", "SOPHIA": "#ff7f0e"}
    
    # 1. Training Loss Comparison
    plt.figure(figsize=(8, 4))
    
    for optimizer, runs in optimizer_data.items():
        for run in runs:
            if run["loss_csv"]:
                df = load_csv_file(run["loss_csv"])
                label = f"{optimizer.upper()} ({run['model_dir']})"
                plt.plot(df["step"], df["loss"], label=label, color=colors.get(optimizer, "gray"), 
                        linestyle="-" if optimizer == "adamw" else "--")
    
    plt.title("Training Loss Comparison", fontsize=12)
    plt.xlabel("Steps", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_loss_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Validation Loss Comparison
    plt.figure(figsize=(7, 4))
    
    for optimizer, runs in optimizer_data.items():
        for run in runs:
            if run["val_csv"]:
                df = load_csv_file(run["val_csv"])
                label = f"{optimizer.upper()} ({run['model_dir']})"
                plt.plot(df["epoch"], df["val_loss"], marker='o', label=label, color=colors.get(optimizer, "gray"))
    
    plt.title("Validation Loss Comparison", fontsize=12)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Validation Loss", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "validation_loss_comparison.png"), dpi=300)
    plt.close()
    
    # 3. Perplexity Comparison
    plt.figure(figsize=(8, 8))
    
    for optimizer, runs in optimizer_data.items():
        for run in runs:
            if run["loss_csv"]:
                df = load_csv_file(run["loss_csv"])
                if "perplexity" in df.columns:
                    label = f"{optimizer.upper()} ({run['model_dir']})"
                    plt.plot(df["step"], df["perplexity"], label=label, color=colors.get(optimizer, "gray"),
                            linestyle="-" if optimizer == "adamw" else "--")
    
    plt.title("Perplexity Comparison", fontsize=12)
    plt.xlabel("Steps", fontsize=10)
    plt.ylabel("Perplexity", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "perplexity_comparison.png"), dpi=300)
    plt.close()
    
    # 4. Training Stability (Loss Variance)
    plt.figure(figsize=(8, 8))
    
    for optimizer, runs in optimizer_data.items():
        for run in runs:
            if run["loss_csv"]:
                df = load_csv_file(run["loss_csv"])
                if "loss_variance" in df.columns:
                    # Convert 'N/A' to NaN
                    df["loss_variance"] = pd.to_numeric(df["loss_variance"], errors='coerce')
                    # Drop NaN values
                    df = df.dropna(subset=["loss_variance"])
                    if not df.empty:
                        label = f"{optimizer.upper()} ({run['model_dir']})"
                        plt.plot(df["step"], df["loss_variance"], label=label, color=colors.get(optimizer, "gray"),
                                linestyle="-" if optimizer == "adamw" else "--")
    
    plt.title("Training Stability (Loss Variance)", fontsize=12)
    plt.xlabel("Steps", fontsize=10)
    plt.ylabel("Loss Variance", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_variance_comparison.png"), dpi=300)
    plt.close()
    
    # 5. Gradient Norm Comparison
    plt.figure(figsize=(8, 8))
    
    for optimizer, runs in optimizer_data.items():
        for run in runs:
            if run["loss_csv"]:
                df = load_csv_file(run["loss_csv"])
                label = f"{optimizer.upper()} ({run['model_dir']})"
                plt.plot(df["step"], df["gradient_norm"], label=label, color=colors.get(optimizer, "gray"),
                        linestyle="-" if optimizer == "adamw" else "--")
    
    plt.title("Gradient Norm Comparison", fontsize=12)
    plt.xlabel("Steps", fontsize=10)
    plt.ylabel("Gradient Norm", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "gradient_norm_comparison.png"), dpi=300)
    plt.close()
    
    # 6. Steps to Target Loss
    fig, ax = plt.subplots(figsize=(8, 5))
    
    target_losses = []
    optimizer_names = []
    steps_to_target = []
    model_names = []
    
    for optimizer, runs in optimizer_data.items():
        for run in runs:
            metrics = run["metrics"]
            for target_loss, steps in metrics.get("steps_to_target_loss", {}).items():
                if steps is not None:
                    target_losses.append(float(target_loss))
                    optimizer_names.append(optimizer.upper())
                    steps_to_target.append(steps)
                    model_names.append(run["model_dir"])
    
    if target_losses:
        df = pd.DataFrame({
            "Target Loss": target_losses,
            "Optimizer": optimizer_names,
            "Steps": steps_to_target,
            "Model": model_names
        })
        
        pivot_df = df.pivot_table(index="Target Loss", columns="Optimizer", values="Steps", aggfunc="first")
        pivot_df = pivot_df.sort_index(ascending=False)
        
        pivot_df.plot(kind="barh", ax=ax)
        
        ax.set_title("Steps to Reach Target Loss", fontsize=12)
        ax.set_xlabel("Steps", fontsize=10)
        ax.set_ylabel("Target Loss", fontsize=10)
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "steps_to_target_loss.png"), dpi=300)
        plt.close()
    
    # 7. Training Time and Memory Usage Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    optimizer_names = []
    training_times = []
    memory_usages = []
    model_names = []
    
    for optimizer, runs in optimizer_data.items():
        for run in runs:
            metrics = run["metrics"]
            optimizer_names.append(optimizer.upper())
            # Handle None values for training time
            training_time = metrics.get("total_training_time")
            if training_time is None:
                training_time = 0
            training_times.append(training_time)
            # Handle None values for memory usage
            memory_usage = metrics.get("peak_memory_usage")
            if memory_usage is None:
                memory_usage = 0
            memory_usages.append(memory_usage)
            model_names.append(run["model_dir"])
    
    df = pd.DataFrame({
        "Optimizer": optimizer_names,
        "Training Time (minutes)": training_times,
        "Peak Memory (MB)": memory_usages,
        "Model": model_names
    })
    
    # Fix the palette issue by using hue explicitly
    sns.barplot(x="Optimizer", y="Training Time (minutes)", hue="Optimizer", data=df, ax=ax1, palette=colors, legend=False)
    ax1.set_title("Training Time Comparison", fontsize=10)
    ax1.grid(True, axis='y', alpha=0.3)
    
    sns.barplot(x="Optimizer", y="Peak Memory (MB)", hue="Optimizer", data=df, ax=ax2, palette=colors, legend=False)
    ax2.set_title("Peak Memory Usage Comparison", fontsize=10)
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "resources_comparison.png"), dpi=300)
    plt.close()
    
    # 8. Best Validation Loss Comparison
    plt.figure(figsize=(10, 6))
    
    optimizer_names = []
    best_val_losses = []
    model_names = []
    
    for optimizer, runs in optimizer_data.items():
        for run in runs:
            metrics = run["metrics"]
            optimizer_names.append(optimizer.upper())
            # Handle None values by defaulting to 0
            val_loss = metrics.get("best_validation_loss")
            if val_loss is None:
                val_loss = 0
            best_val_losses.append(val_loss)
            model_names.append(run["model_dir"])
    
    df = pd.DataFrame({
        "Optimizer": optimizer_names,
        "Best Validation Loss": best_val_losses,
        "Model": model_names
    })
    
    # Use hue explicitly to avoid FutureWarning
    ax = sns.barplot(x="Optimizer", y="Best Validation Loss", hue="Optimizer", data=df, palette=colors, legend=False)
    
    # Add value labels on top of bars
    for i, v in enumerate(best_val_losses):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.title("Best Validation Loss Comparison", fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "best_val_loss_comparison.png"), dpi=300)
    plt.close()

    # 9. Summary Dashboard
    fig = plt.figure(figsize=(10, 10))
    grid = plt.GridSpec(3, 3, figure=fig, wspace=0.3, hspace=0.3)
    
    # Training Loss Plot
    ax1 = fig.add_subplot(grid[0, :2])
    for optimizer, runs in optimizer_data.items():
        for run in runs:
            if run["loss_csv"]:
                df = load_csv_file(run["loss_csv"])
                label = f"{optimizer.upper()}"
                ax1.plot(df["step"], df["loss"], label=label, color=colors.get(optimizer, "gray"),
                        linestyle="-" if optimizer == "adamw" else "--")
    ax1.set_title("Training Loss", fontsize=10)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Validation Loss Plot
    ax2 = fig.add_subplot(grid[0, 2])
    for optimizer, runs in optimizer_data.items():
        for run in runs:
            if run["val_csv"]:
                df = load_csv_file(run["val_csv"])
                label = f"{optimizer.upper()}"
                ax2.plot(df["epoch"], df["val_loss"], marker='o', label=label, color=colors.get(optimizer, "gray"))
    ax2.set_title("Validation Loss", fontsize=10)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Perplexity Plot
    ax3 = fig.add_subplot(grid[1, :2])
    for optimizer, runs in optimizer_data.items():
        for run in runs:
            if run["loss_csv"]:
                df = load_csv_file(run["loss_csv"])
                if "perplexity" in df.columns:
                    label = f"{optimizer.upper()}"
                    ax3.plot(df["step"], df["perplexity"], label=label, color=colors.get(optimizer, "gray"),
                            linestyle="-" if optimizer == "adamw" else "--")
    ax3.set_title("Perplexity", fontsize=10)
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Perplexity")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Training Stability Plot
    ax4 = fig.add_subplot(grid[1, 2])
    for optimizer, runs in optimizer_data.items():
        for run in runs:
            if run["loss_csv"]:
                df = load_csv_file(run["loss_csv"])
                if "loss_variance" in df.columns:
                    df["loss_variance"] = pd.to_numeric(df["loss_variance"], errors='coerce')
                    df = df.dropna(subset=["loss_variance"])
                    if not df.empty:
                        label = f"{optimizer.upper()}"
                        ax4.plot(df["step"], df["loss_variance"], label=label, color=colors.get(optimizer, "gray"),
                                linestyle="-" if optimizer == "adamw" else "--")
    ax4.set_title("Training Stability", fontsize=10)
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("Loss Variance")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Performance Metrics
    ax5 = fig.add_subplot(grid[2, :])
    metrics_df = pd.DataFrame({
        "Optimizer": optimizer_names,
        "Training Time (min)": training_times,
        "Peak Memory (GB)": [m/1024 if m is not None else 0 for m in memory_usages],  # Convert to GB, handle None values
        "Best Val Loss": best_val_losses,
        "Model": model_names
    })
    metrics_df.set_index("Optimizer").plot(kind="bar", ax=ax5)
    ax5.set_title("Performance Metrics Comparison", fontsize=10)
    ax5.set_ylabel("Value")
    ax5.grid(True, axis='y', alpha=0.3)
    
    # Add title
    plt.suptitle("Optimizer Comparison Dashboard", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(run_dir, "optimizer_comparison_dashboard.png"), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {run_dir}")
    return run_dir  # Return the created directory path for reference

if __name__ == "__main__":
    # Set the directory where your metrics files are stored
    metrics_dir = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/11.Round_2_Sophia_Adam_**TRAINING**_APPS/03.json_csv_logs_metrics_plot_scope_only"
    
    create_optimizer_comparison_plots(metrics_dir)