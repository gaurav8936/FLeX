# --------------------------------------
# CONFIGURABLE DEFAULT SETTINGS
# --------------------------------------
# Default model name
DEFAULT_MODEL = "codellama/CodeLlama-7b-hf"

# Default training settings
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_LENGTH = 512
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_GRADIENT_CLIP = 1.0

# Default LoRA settings
DEFAULT_LORA_R = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_BIAS = "none"
DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]

# Default Fourier settings
DEFAULT_FOURIER_LAMBDA = 0.01
DEFAULT_FOURIER_THRESHOLD = 0.5
DEFAULT_FOURIER_LOW_WEIGHT = 1.0
DEFAULT_FOURIER_HIGH_WEIGHT = 0.1

# Default dataset
DEFAULT_DATASET = "mbpp"
DEFAULT_DATASET_SPLIT = "train"

def get_experiment_configurations():
    """
    Returns a list of configurations for LoRA fine-tuning experiments
    with Fourier transform regularization for cross-lingual transfer
    """
    # Base Config 
    base_config = {
        "logging": {
            "log_dir": "logs",
        },
        "model": {
            "name": DEFAULT_MODEL,
            "dtype": "float16",
            "device_map": "auto",
            "load_in_8bit": False,
            "load_in_4bit": False,
        },
        "training": {
            "num_epochs": DEFAULT_EPOCHS,
            "batch_size": DEFAULT_BATCH_SIZE,
            "max_length": DEFAULT_MAX_LENGTH,
            "save_every": 1,
            "gradient_clip": DEFAULT_GRADIENT_CLIP,
            "dataset": {
                "name": DEFAULT_DATASET,
                "split": DEFAULT_DATASET_SPLIT,
            },
            "optimizer": {
                "name": "adamw",
                "lr": DEFAULT_LEARNING_RATE,
                "weight_decay": DEFAULT_WEIGHT_DECAY,
                "betas": [0.9, 0.999],
            },
            "scheduler": {
                "name": "cosine",
                "warmup_steps": 100,
                "eta_min": 1e-5,
            },
            # Fourier regularization parameters
            "fourier_lambda": DEFAULT_FOURIER_LAMBDA,
            "fourier_threshold": DEFAULT_FOURIER_THRESHOLD,
            "fourier_low_weight": DEFAULT_FOURIER_LOW_WEIGHT,
            "fourier_high_weight": DEFAULT_FOURIER_HIGH_WEIGHT,
        },
        "lora": {
            "r": DEFAULT_LORA_R,
            "alpha": DEFAULT_LORA_ALPHA,
            "dropout": DEFAULT_LORA_DROPOUT,
            "bias": DEFAULT_LORA_BIAS,
            "target_modules": DEFAULT_TARGET_MODULES,
            "modules_to_save": None,
            "fan_in_fan_out": False,
        }
    }
    
    # 1. Baseline config with default Fourier settings
    # Default settings: lambda=0.01, threshold=0.5, weights=1.0 & 0.1
    
    # 2. Strong Fourier regularization
    config_strong_fourier = deep_copy_config(base_config)
    config_strong_fourier["training"]["fourier_lambda"] = 0.05
    
    # 3. Weak Fourier regularization
    config_weak_fourier = deep_copy_config(base_config)
    config_weak_fourier["training"]["fourier_lambda"] = 0.001
    
    # 4. Higher low-frequency emphasis
    config_higher_low_freq = deep_copy_config(base_config)
    config_higher_low_freq["training"]["fourier_low_weight"] = 2.0
    config_higher_low_freq["training"]["fourier_high_weight"] = 0.05
    
    # 5. Lower frequency threshold (more frequencies considered "high")
    config_lower_threshold = deep_copy_config(base_config)
    config_lower_threshold["training"]["fourier_threshold"] = 0.3
    
    # 6. Higher frequency threshold (fewer frequencies considered "high")
    config_higher_threshold = deep_copy_config(base_config)
    config_higher_threshold["training"]["fourier_threshold"] = 0.7
    
    # 7. Comprehensive target modules for better transfer
    config_comprehensive_targets = deep_copy_config(base_config)
    config_comprehensive_targets["lora"]["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    config_comprehensive_targets["lora"]["r"] = 16  # Higher rank for more capacity
    
    # 8. Higher rank with strong Fourier regularization
    config_high_rank_strong_fourier = deep_copy_config(base_config)
    config_high_rank_strong_fourier["lora"]["r"] = 16
    config_high_rank_strong_fourier["training"]["fourier_lambda"] = 0.03
    
    # 9. Focus on MLP layers with Fourier
    config_mlp_focus = deep_copy_config(base_config)
    config_mlp_focus["lora"]["target_modules"] = ["gate_proj", "up_proj", "down_proj"]
    config_mlp_focus["training"]["fourier_lambda"] = 0.02
    
    # 10. Gradually decreasing high-frequency penalty
    config_gradual = deep_copy_config(base_config)
    config_gradual["training"]["fourier_high_weight"] = 0.3  # Less aggressive penalty
    config_gradual["training"]["fourier_lambda"] = 0.02      # Slightly stronger overall
    
    # 11. Attention only with strong Fourier
    config_attn_strong = deep_copy_config(base_config)
    config_attn_strong["lora"]["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    config_attn_strong["training"]["fourier_lambda"] = 0.05
    
    # 12. Higher learning rate with Fourier
    config_higher_lr = deep_copy_config(base_config) 
    config_higher_lr["training"]["optimizer"]["lr"] = 3e-4
    config_higher_lr["training"]["fourier_lambda"] = 0.01
    
    return [
        base_config,                 # 1: Baseline with default Fourier settings
        config_strong_fourier,       # 2: Stronger Fourier regularization
        config_weak_fourier,         # 3: Weaker Fourier regularization
        config_higher_low_freq,      # 4: Higher emphasis on low frequencies
        config_lower_threshold,      # 5: Lower frequency threshold (more high frequencies)
        config_higher_threshold,     # 6: Higher frequency threshold (fewer high frequencies)
        config_comprehensive_targets,# 7: Target all key modules (attention + MLP)
        config_high_rank_strong_fourier, # 8: Higher rank with strong regularization
        config_mlp_focus,            # 9: Focus on MLP layers with Fourier
        config_gradual,              # 10: Gradual high-frequency penalty
    ]

def deep_copy_config(config):
    """Create a deep copy of a nested dictionary configuration"""
    import copy
    return copy.deepcopy(config)