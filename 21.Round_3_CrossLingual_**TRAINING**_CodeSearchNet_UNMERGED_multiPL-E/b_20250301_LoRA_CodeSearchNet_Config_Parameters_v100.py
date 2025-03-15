# Configuration file for CodeSearchNet LoRA training experiments
# best prior config
# python 22.CrossLingual_TRAINING_CodeSearchNet_unmerged_a.py --config_range "17:18"

def get_experiment_configurations():
    """
    Returns a list of configurations for LoRA fine-tuning experiments
    using CodeSearchNet dataset instead of MBPP
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
                "name": "code_search_net",
                "language": "python",
                "split": "train",
                "subset": 2000,  # Use only 2000 samples to make training manageable
                # "subset": 100,  # Use only 100 samples to make training manageable
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

    # 03.Higher rank
    config_r16 = deep_copy_config(base_config)
    config_r16["lora"]["r"] = 16
    
    # 04.Experiment with different alpha values
    config_alpha8 = deep_copy_config(base_config)
    config_alpha8["lora"]["alpha"] = 8

    # 05.Higher alpha
    config_alpha32 = deep_copy_config(base_config)
    config_alpha32["lora"]["alpha"] = 32
    
    # 06.Experiment with different dropout values
    config_dropout0 = deep_copy_config(base_config)
    config_dropout0["lora"]["dropout"] = 0.0

    # 07.Higher dropout
    config_dropout10 = deep_copy_config(base_config)
    config_dropout10["lora"]["dropout"] = 0.1
    
    # 08.Experiment with different target modules
    config_qkv = deep_copy_config(base_config)
    config_qkv["lora"]["target_modules"] = ["q_proj", "k_proj", "v_proj"]

    # 09.Target all projections
    config_qkvo = deep_copy_config(base_config)
    config_qkvo["lora"]["target_modules"] = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # 10.Experiment with learning rates
    config_lr1e4 = deep_copy_config(base_config)
    config_lr1e4["training"]["optimizer"]["lr"] = 1e-4

    # 11.Higher learning rate
    config_lr5e4 = deep_copy_config(base_config)
    config_lr5e4["training"]["optimizer"]["lr"] = 5e-4
    
    # 12.Experiment with Sophia optimizer
    config_sophia = deep_copy_config(base_config)
    config_sophia["training"]["optimizer"]["name"] = "sophia"
    config_sophia["training"]["optimizer"]["lr"] = 3e-4
    
    # 13.Experiment with batch size
    config_batch8 = deep_copy_config(base_config)
    config_batch8["training"]["batch_size"] = 8

    # 14.Cross-lingual - Java dataset
    config_java = deep_copy_config(base_config)
    config_java["training"]["dataset"]["language"] = "java"
    
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

    # 25.Dataset Size Variation - 5000 examples
    config_larger_dataset = deep_copy_config(base_config)
    config_larger_dataset["training"]["dataset"]["subset"] = 5000
    
    # 26.Dataset Size Variation - 1000 examples
    config_smaller_dataset = deep_copy_config(base_config)
    config_smaller_dataset["training"]["dataset"]["subset"] = 1000
    
    # 27.Go language dataset
    config_go = deep_copy_config(base_config)
    config_go["training"]["dataset"]["language"] = "go"
    
    # 28.Ruby language dataset
    config_ruby = deep_copy_config(base_config)
    config_ruby["training"]["dataset"]["language"] = "ruby"
    
    # 29.Combination of Java + Python (Sequential training)
    config_java_python = deep_copy_config(base_config)
    config_java_python["training"]["dataset"]["language"] = "java+python"
    config_java_python["training"]["dataset"]["subset"] = 1000  # 1000 from each language
    
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
        config_sophia,         # 12: Sophia optimizer (instead of Adam)
        config_batch8,         # 13: Larger batch size
        config_java,           # 14: Cross-lingual Transfer - Java dataset
        config_alpha32_r4,     # 15: Higher Alpha with Lower Rank
        config_more_epochs,    # 16: Extended Training Epochs
        config_mlp_attn,       # 17: Multiple Layer Targeting
        config_higher_lr,      # 18: Higher Learning Rate
        config_warmup,         # 19: Warm-Up Steps Adjustment
        config_weight_decay,   # 20: Weight Decay Variation
        config_long_context,   # 21: Longer Sequence Length
        config_lowlr_highepoch,# 22: Lower LR with Higher Epochs
        config_linear_scheduler,# 23: Linear LR Scheduler
        config_target_dropout,  # 24: Combination of Target Modules and Dropout
        config_larger_dataset, # 25: Larger dataset (5000 examples)
        config_smaller_dataset,# 26: Smaller dataset (1000 examples)
        config_go,             # 27: Go language dataset
        config_ruby,           # 28: Ruby language dataset
        config_java_python,    # 29: Sequential Java+Python training
    ]

def deep_copy_config(config):
    """Create a deep copy of a nested dictionary configuration"""
    import copy
    return copy.deepcopy(config)