#!/usr/bin/env python3
# Main visualization controller for optimizer comparison
# Usage: python e_0_20250305_visualization_main_v1.00.py --metrics_dir /path/to/metrics

import os
import glob
import json
import argparse
from datetime import datetime

# Import individual plot modules
from e_1_20250305_plot_training_loss_v1_00 import plot_training_loss
from e_2_20250305_plot_validation_loss_v1_00 import plot_validation_loss
from e_3_20250305_plot_perplexity_v1_00 import plot_perplexity
from e_4_20250305_plot_loss_variance_v1_00 import plot_loss_variance
from e_5_20250305_plot_gradient_norm_v1_00 import plot_gradient_norm
from e_6_20250305_plot_steps_to_target_v1_00 import plot_steps_to_target
from e_7_20250305_plot_resources_v1_00 import plot_resources
from e_8_20250305_plot_best_val_loss_v1_00 import plot_best_val_loss
from e_9_20250305_plot_dashboard_v1_00 import plot_dashboard

# Import new plot modules
from e_10_20250305_plot_convergence_rate_v1_00 import plot_convergence_rate
from e_11_20250305_plot_lr_dynamics_v1_00 import plot_lr_dynamics
from e_12_20250305_plot_efficiency_ratio_v1_00 import plot_efficiency_ratio
from e_13_20250305_plot_hessian_metrics_v1_00 import plot_hessian_metrics
from e_14_20250305_plot_step_size_analysis_v1_00 import plot_step_size_analysis
from e_15_20250305_plot_val_train_ratio_v1_00 import plot_val_train_ratio
from e_16_20250305_plot_epoch_improvement_v1_00 import plot_epoch_improvement
from e_17_20250305_plot_composite_score_v1_00 import plot_composite_score

def load_metrics_files(metrics_dir):
    """
    Load all metrics from JSON files and organize by optimizer
    Returns a dictionary with optimizer_data structure
    """
    json_files = glob.glob(os.path.join(metrics_dir, "*_metrics.json"))
    optimizer_data = {}
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            metrics = json.load(f)
        
        optimizer = metrics["optimizer"].lower()
        timestamp = metrics["timestamp"]
        
        if optimizer not in optimizer_data:
            optimizer_data[optimizer] = []
        
        optimizer_data[optimizer].append({
            "metrics": metrics,
            "timestamp": timestamp,
            "model_dir": metrics.get("model_dir", "unknown")
        })
    
    return optimizer_data

def create_visualization_directory(output_base_dir):
    """Create a timestamped directory for visualizations"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_dir = os.path.join(output_base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def generate_all_plots(metrics_dir, output_dir=None):
    """Generate all optimizer comparison plots"""
    if output_dir is None:
        output_dir = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/11.Round_2_Sophia_Adam_**TRAINING**_APPS/06.Visualizations"
    
    # Create base directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamped directory for this visualization run
    run_dir = create_visualization_directory(output_dir)
    
    # Load metrics data
    optimizer_data = load_metrics_files(metrics_dir)
    
    # Generate each plot type
    plot_training_loss(optimizer_data, run_dir)
    plot_validation_loss(optimizer_data, run_dir)
    plot_perplexity(optimizer_data, run_dir)
    plot_loss_variance(optimizer_data, run_dir)
    plot_gradient_norm(optimizer_data, run_dir)
    plot_steps_to_target(optimizer_data, run_dir)
    plot_resources(optimizer_data, run_dir)
    plot_best_val_loss(optimizer_data, run_dir)
    plot_dashboard(optimizer_data, run_dir)
    
    # # Generate new plot types
    plot_convergence_rate(optimizer_data, run_dir)
    plot_lr_dynamics(optimizer_data, run_dir)
    plot_efficiency_ratio(optimizer_data, run_dir)
    plot_hessian_metrics(optimizer_data, run_dir)
    plot_step_size_analysis(optimizer_data, run_dir)
    plot_val_train_ratio(optimizer_data, run_dir)
    plot_epoch_improvement(optimizer_data, run_dir)
    plot_composite_score(optimizer_data, run_dir)
    
    print(f"Visualizations saved to {run_dir}")
    return run_dir

def main():
    parser = argparse.ArgumentParser(description='Generate optimizer comparison visualizations')
    parser.add_argument('--metrics_dir', type=str, 
                      default="/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/11.Round_2_Sophia_Adam_**TRAINING**_APPS/03.json_csv_logs_metrics_plot_scope_only",
                      help='Directory containing metrics JSON files')
    parser.add_argument('--output_dir', type=str, 
                      default=None,
                      help='Output directory for visualizations (defaults to standard path)')
    
    args = parser.parse_args()
    
    generate_all_plots(args.metrics_dir, args.output_dir)

if __name__ == "__main__":
    main()