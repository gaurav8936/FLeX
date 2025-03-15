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

# Default dataset
DEFAULT_DATASET = "mbpp"
DEFAULT_DATASET_SPLIT = "train"

def get_experiment_configurations():
    """
    Returns a list of configurations for LoRA fine-tuning experiments
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
    
    # Experiment configurations
    config_r4 = deep_copy_config(base_config)
    config_r4["lora"]["r"] = 4

    config_r16 = deep_copy_config(base_config)
    config_r16["lora"]["r"] = 16
    
    config_alpha8 = deep_copy_config(base_config)
    config_alpha8["lora"]["alpha"] = 8

    config_alpha32 = deep_copy_config(base_config)
    config_alpha32["lora"]["alpha"] = 32
    
    config_dropout0 = deep_copy_config(base_config)
    config_dropout0["lora"]["dropout"] = 0.0

    config_dropout10 = deep_copy_config(base_config)
    config_dropout10["lora"]["dropout"] = 0.1
    
    config_qkv = deep_copy_config(base_config)
    config_qkv["lora"]["target_modules"] = ["q_proj", "k_proj", "v_proj"]

    config_qkvo = deep_copy_config(base_config)
    config_qkvo["lora"]["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    config_lr1e4 = deep_copy_config(base_config)
    config_lr1e4["training"]["optimizer"]["lr"] = 1e-4

    config_lr5e4 = deep_copy_config(base_config)
    config_lr5e4["training"]["optimizer"]["lr"] = 5e-4
    
    config_adam = deep_copy_config(base_config)
    config_adam["training"]["optimizer"]["name"] = "adam"
    
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
    ]

def deep_copy_config(config):
    """Create a deep copy of a nested dictionary configuration"""
    import copy
    return copy.deepcopy(config)