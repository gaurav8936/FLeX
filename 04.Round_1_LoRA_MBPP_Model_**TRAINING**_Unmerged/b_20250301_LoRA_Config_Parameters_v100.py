# Example of adding multiple configurations for experimentation
# Add these to the configs list in the main script when ready to run more experiments

def get_experiment_configurations():
    """
    Returns a list of configurations for LoRA fine-tuning experiments
    """
    # 01.Base Config 
    base_config = {
        "logging": {
            "log_dir": "logs",
        },
        "model": {
            "name": "codellama/CodeLlama-7b-hf",
            "dtype": "float16",
            "device_map": "auto",
            "load_in_8bit": False,
            "load_in_4bit": False,
        },
        "training": {
            "num_epochs": 3,
            "batch_size": 4,
            "max_length": 512,
            "save_every": 1,
            "gradient_clip": 1.0,
            "dataset": {
                "name": "mbpp",
                "split": "train",
            },
            "optimizer": {
                "name": "adamw",
                "lr": 2e-4,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
            },
            "scheduler": {
                "name": "cosine",
                "warmup_steps": 100,
                "eta_min": 1e-5,
            },
        },
        "lora": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": ["q_proj", "v_proj"],
            "modules_to_save": None,
            "fan_in_fan_out": False,
        }
    }
    
    # 02.Experiment with different LoRA ranks (r)
    config_r4 = deep_copy_config(base_config)
    config_r4["lora"]["r"] = 4

    # 03.szadfdsf
    config_r16 = deep_copy_config(base_config)
    config_r16["lora"]["r"] = 16
    
    # 04.Experiment with different alpha values
    config_alpha8 = deep_copy_config(base_config)
    config_alpha8["lora"]["alpha"] = 8

    # 05.dasdfs
    config_alpha32 = deep_copy_config(base_config)
    config_alpha32["lora"]["alpha"] = 32
    
    # 06.Experiment with different dropout values
    config_dropout0 = deep_copy_config(base_config)
    config_dropout0["lora"]["dropout"] = 0.0

    # 07.TBD
    config_dropout10 = deep_copy_config(base_config)
    config_dropout10["lora"]["dropout"] = 0.1
    
    # 08.Experiment with different target modules
    config_qkv = deep_copy_config(base_config)
    config_qkv["lora"]["target_modules"] = ["q_proj", "k_proj", "v_proj"]

    # 09.TBD
    config_qkvo = deep_copy_config(base_config)
    config_qkvo["lora"]["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # 10.Experiment with learning rates
    config_lr1e4 = deep_copy_config(base_config)
    config_lr1e4["training"]["optimizer"]["lr"] = 1e-4

    # 11 TBD
    config_lr5e4 = deep_copy_config(base_config)
    config_lr5e4["training"]["optimizer"]["lr"] = 5e-4
    
    # 12.Experiment with optimizer
    config_adam = deep_copy_config(base_config)
    config_adam["training"]["optimizer"]["name"] = "adam"
    
    # 13.Experiment with batch size
    config_batch8 = deep_copy_config(base_config)
    config_batch8["training"]["batch_size"] = 8

    # 14.Cross-lingual Transfer Learning
    config_cross_lingual = deep_copy_config(base_config)
    config_cross_lingual["training"]["dataset"]["name"] = "code_train_java"  # Replace with actual Java dataset
    config_cross_lingual["training"]["dataset"]["split"] = "train"
    config_cross_lingual["lora"]["r"] = 16  # Higher rank for cross-lingual
    
    # 15.Higher Alpha with Lower Rank
    config_alpha32_r4 = deep_copy_config(base_config)
    config_alpha32_r4["lora"]["alpha"] = 32
    config_alpha32_r4["lora"]["r"] = 4
    
    # 16.Extended Training Epochs
    config_more_epochs = deep_copy_config(base_config)
    config_more_epochs["training"]["num_epochs"] = 5
    
    # 17.Multiple Layer Targeting (Including MLP)
    config_mlp_attn = deep_copy_config(base_config)
    config_mlp_attn["lora"]["target_modules"] = ["q_proj", "v_proj", "down_proj", "up_proj"]
    
    # 18.Higher Learning Rate
    config_higher_lr = deep_copy_config(base_config)
    config_higher_lr["training"]["optimizer"]["lr"] = 3e-4
    
    # 19.Warm-Up Steps Adjustment
    config_warmup = deep_copy_config(base_config)
    config_warmup["training"]["scheduler"]["warmup_steps"] = 200  # Double the warmup
    
    # 20.Weight Decay Variation
    config_weight_decay = deep_copy_config(base_config)
    config_weight_decay["training"]["optimizer"]["weight_decay"] = 0.05  # Higher weight decay
    
    # 21.Longer Sequence Length for Context
    config_long_context = deep_copy_config(base_config)
    config_long_context["training"]["max_length"] = 1024  # Double the context length
    
    # 22.Lower Learning Rate with Higher Epochs
    config_lowlr_highepoch = deep_copy_config(base_config)
    config_lowlr_highepoch["training"]["optimizer"]["lr"] = 1e-4
    config_lowlr_highepoch["training"]["num_epochs"] = 4
    
    # 23.Linear LR Scheduler
    config_linear_scheduler = deep_copy_config(base_config)
    config_linear_scheduler["training"]["scheduler"]["name"] = "linear"
    config_linear_scheduler["training"]["scheduler"]["start_factor"] = 1.0
    config_linear_scheduler["training"]["scheduler"]["end_factor"] = 0.1
    
    # 24.Combination of Target Modules and Dropout
    config_target_dropout = deep_copy_config(base_config)
    config_target_dropout["lora"]["target_modules"] = ["q_proj", "k_proj", "v_proj"]
    config_target_dropout["lora"]["dropout"] = 0.2

    # # ----- Additional Scenarios Not Already Covered -----
    
    # 25.Higher Rank Exploration: Increase LoRA rank to 32
    config_r32 = deep_copy_config(base_config)
    config_r32["lora"]["r"] = 32
    
    # 26.Cyclic Learning Rate Scheduler: Use a cyclic scheduler instead of cosine or linear
    config_cyclic_scheduler = deep_copy_config(base_config)
    config_cyclic_scheduler["training"]["scheduler"]["name"] = "cyclic"
    config_cyclic_scheduler["training"]["scheduler"]["cycle_length"] = 100  # Example parameter
    config_cyclic_scheduler["training"]["scheduler"]["cycle_mult"] = 1.0     # Example parameter
    
    # 27.Layer-Specific Tuning: Apply LoRA only to the top 4 Transformer layers
    # (Assuming the training code can handle a parameter "apply_to_layers")
    config_top_layers = deep_copy_config(base_config)
    config_top_layers["lora"]["apply_to_layers"] = "top4"  # Custom key indicating only last 4 layers are adapted
    
    # 28.Combination with BitFit: Enable bias updates along with LoRA updates
    config_lora_bitfit = deep_copy_config(base_config)
    config_lora_bitfit["lora"]["bias"] = "all"  # Update biases as well as low-rank matrices
    
    # 29.Batch Size and LR Interaction: Lower batch size to 2 with a higher learning rate
    config_batch2_lr5e4 = deep_copy_config(base_config)
    config_batch2_lr5e4["training"]["batch_size"] = 2
    config_batch2_lr5e4["training"]["optimizer"]["lr"] = 5e-4
    
    # Return all configurations
    # Return all configurations
    return [
        base_config,           # 1: Baseline - r=8, alpha=16, dropout=0.05, q_proj+v_proj
        config_r4,             # 2: Lower rank
        config_r16,            # 3: Higher rank
        config_alpha8,         # 4: Lower alpha
        config_alpha32,        # 5: Higher alpha
        config_dropout0,       # 6: No dropout
        config_dropout10,      # 7: Higher dropout
        config_qkv,            # 8: Target q, k, v projections
        config_qkvo,           # 9: Target all projections
        config_lr1e4,          # 10: Lower learning rate
        config_lr5e4,          # 11: Higher learning rate
        config_adam,           # 12: Adam optimizer
        config_batch8,         # 13: Larger batch size
        config_cross_lingual,  # 14: Cross-lingual Transfer Learning
        config_alpha32_r4,     # 15: Higher Alpha with Lower Rank
        config_more_epochs,    # 16: Extended Training Epochs
        config_mlp_attn,       # 17: Multiple Layer Targeting
        config_higher_lr,      # 18: Higher Learning Rate (not config_diff_lr)
        config_warmup,         # 19: Warm-Up Steps Adjustment
        config_weight_decay,   # 20: Weight Decay Variation
        config_long_context,   # 21: Longer Sequence Length
        config_lowlr_highepoch,# 22: Lower LR with Higher Epochs
        config_linear_scheduler,# 23: Linear LR Scheduler
        config_target_dropout,  # 24: Combination of Target Modules and Dropout
        config_r32,            # 25: Higher Rank Exploration: Increase LoRA rank to 32
        config_cyclic_scheduler,  # 26: Cyclic Learning Rate Scheduler: Use a cyclic scheduler
        config_top_layers,     # 27: Layer-Specific Tuning: Apply LoRA only to the top 4 Transformer layers
        config_lora_bitfit,    # 28: Combination with BitFit: Enable bias updates along with LoRA updates
        config_batch2_lr5e4    # 29: Batch Size and LR Interaction: Lower batch size to 2 with a higher learning rate
    ]

def deep_copy_config(config):
    """Create a deep copy of a nested dictionary configuration"""
    import copy
    return copy.deepcopy(config)

# Example of how to use this in the main script:
"""
from hyperparameter_configs import get_experiment_configurations

def main():
    # Get all configurations
    configs = get_experiment_configurations()
    
    # For testing, you might want to run just one config
    # configs = configs[:1]  # Just run the first configuration
    
    # Rest of the main function...
"""