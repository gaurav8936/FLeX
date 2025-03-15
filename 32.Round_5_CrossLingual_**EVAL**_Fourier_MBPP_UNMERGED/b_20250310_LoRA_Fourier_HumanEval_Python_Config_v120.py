# CHEATSHEET #
# 20250310_bph_alpha16_dropout0.05_r8_lr0.0002_fou0.01_epochs3  --> INDEX 0
# 20250310_wov_alpha16_dropout0.05_r8_lr0.0002_fou0.05_epochs3  --> INDEX 1
# 20250310_rpl_alpha16_dropout0.05_r8_lr0.0002_fou0.001_epochs3  --> INDEX 2
# 20250310_kfg_alpha16_dropout0.05_r8_lr0.0002_fou0.01_epochs3  --> INDEX 3
# 20250310_cbg_alpha16_dropout0.05_r8_lr0.0002_fou0.01_epochs3  --> INDEX 4
# 20250310_kjg_alpha16_dropout0.05_r8_lr0.0002_fou0.01_epochs3  --> INDEX 5
# 20250310_bch_alpha16_dropout0.05_r16_lr0.0002_fou0.01_epochs3  --> INDEX 6
# 20250310_hiv_alpha16_dropout0.05_r16_lr0.0002_fou0.03_epochs3  --> INDEX 7
# 20250310_gkm_alpha16_dropout0.05_r8_lr0.0002_fou0.02_epochs3  --> INDEX 8
# 20250310_wdq_alpha16_dropout0.05_r8_lr0.0002_fou0.02_epochs3  --> INDEX 9

# "High beam + medium temp" (beams=8, temp=0.25, top_p=0.95) --> INDEX 0
# "High beam (12) + low temp (0.2)" (beams=12, temp=0.2, top_p=0.92) --> INDEX 1
# "Greedy with length penalty" (beams=10, do_sample=False, temp=1.0, length_penalty=1.05) --> INDEX 2
# "Very low temp + repetition penalty" (beams=8, temp=0.15, top_p=0.9, repetition_penalty=1.1) --> INDEX 3
# "High beam + low temp" (beams=10, temp=0.2, top_p=0.95) --> INDEX 4
# "Pure greedy with high beam" (beams=12, do_sample=False, temp=1.0) --> INDEX 5
# "Early stopping + optimal params" (beams=8, temp=0.18, top_p=0.92, early_stopping=True) --> INDEX 6
# "Diversity penalty" (beams=9, temp=0.15, diversity_penalty=0.2) --> INDEX 7
# "Balanced beam + low temp" (beams=10, temp=0.2, top_p=0.9) --> INDEX 8
# "Pure greedy" (beams=8, do_sample=False, temp=1.0) --> INDEX 9
# "High beam + repetition penalty" (beams=12, temp=0.15, top_p=0.95, repetition_penalty=1.15) --> INDEX 10
# "Balanced params" (beams=10, temp=0.22, top_p=0.92) --> INDEX 11
# "Greedy with length penalty (alt)" (beams=12, do_sample=False, temp=1.0, length_penalty=1.05) --> INDEX 12
# "Very low temp + repetition" (beams=8, temp=0.15, top_p=0.9, repetition_penalty=1.2) --> INDEX 13
# "Best tuned params" (beams=10, temp=0.18, top_p=0.92, repetition_penalty=1.05) --> INDEX 14


def get_evaluation_configurations(model_index_range=None, config_index_range=None):
    """
    Returns configurations for LoRA model evaluation, supporting model and config selection via index ranges
    
    Args:
        model_index_range (str, optional): Range of models to evaluate (e.g., "0:2" for first two models)
        config_index_range (str, optional): Range of configs to use (e.g., "0:3" for first three configs)
    
    Returns:
        list: List of configurations for evaluation
    """
    # List of available model paths with indices for easy reference
    model_paths = [
        # --> INDEX 0
        "20250310_bph_alpha16_dropout0.05_r8_lr0.0002_fou0.01_epochs3_Lora_Adapters",  
        # --> INDEX 1
        "20250310_wov_alpha16_dropout0.05_r8_lr0.0002_fou0.05_epochs3_Lora_Adapters",  
        # --> INDEX 2
        "20250310_rpl_alpha16_dropout0.05_r8_lr0.0002_fou0.001_epochs3_Lora_Adapters",  
        # --> INDEX 3
        "20250310_kfg_alpha16_dropout0.05_r8_lr0.0002_fou0.01_epochs3_Lora_Adapters",  
        # --> INDEX 4
        "20250310_cbg_alpha16_dropout0.05_r8_lr0.0002_fou0.01_epochs3_Lora_Adapters",  
        # --> INDEX 5
        "20250310_kjg_alpha16_dropout0.05_r8_lr0.0002_fou0.01_epochs3_Lora_Adapters", 
        # --> INDEX 6
        "20250310_bch_alpha16_dropout0.05_r16_lr0.0002_fou0.01_epochs3_Lora_Adapters",  
        # --> INDEX 7
        "20250310_hiv_alpha16_dropout0.05_r16_lr0.0002_fou0.03_epochs3_Lora_Adapters",  
        # --> INDEX 8
        "20250310_gkm_alpha16_dropout0.05_r8_lr0.0002_fou0.02_epochs3_Lora_Adapters",  
        # --> INDEX 9
        "20250310_wdq_alpha16_dropout0.05_r8_lr0.0002_fou0.02_epochs3_Lora_Adapters"  
    ]
    
    # Base path for models
    model_base_path = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/08.Models/31.Round_5_CrossLingual_**TRAINING**_Fourier_MBPP_UNMERGED/"
    
    # Parse model index range if provided
    if model_index_range:
        start, end = (int(x) if x else None for x in model_index_range.split(":"))
        selected_models = model_paths[start:end]
    else:
        # Default to first model if no range specified
        selected_models = [model_paths[0]]
    
    # Base configuration 
    def create_base_config(model_name):
        return {
            "logging": {
                "log_dir": "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/32.Round_5_CrossLingual_**EVAL**_Fourier_MBPP_UNMERGED/01.Logs",
            },
            "output": {
                "results_dir": "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/32.Round_5_CrossLingual_**EVAL**_Fourier_MBPP_UNMERGED/results_all",
            },
            "tracking": {
                "file": "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/07.Tracking/32.Round_5_CrossLingual_**EVAL**_Fourier_MBPP_UNMERGED.csv",
                "backup_dir": "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/07.Tracking/32.Round_5_CrossLingual_**EVAL**_Fourier_MBPP_UNMERGED_Backup",
            },
            "base_model": {
                "name": "codellama/CodeLlama-7b-hf",  # For reference only in tracking, not used for loading
                "dtype": "float16",
                "device_map": "auto",
                "max_position_embeddings": 256,
            },
            "lora_model": {
                "path": model_base_path + model_name,
            },
            "evaluation": {
                "task": "human_eval",
                "metric": "pass@1",
                "num_samples_per_problem": 1,
                "generation": {
                    "max_new_tokens": 256,
                    "temperature": 0.2,
                    "num_beams": 3,
                    "top_p": 0.90,
                    "do_sample": True,
                }
            }
        }

    # Function to create a config variant quickly
    def create_config_variant(base_config, description, **kwargs):
        """Create a config variant by specifying only the parameters that change"""
        config = deep_copy_config(base_config)
        config["evaluation"]["config_description"] = description
        
        # Update generation parameters
        for key, value in kwargs.items():
            if key in config["evaluation"]:
                config["evaluation"][key] = value
            elif key in config["evaluation"]["generation"]:
                config["evaluation"]["generation"][key] = value
            # Handle nested parameters if needed
            elif "." in key:
                parts = key.split(".")
                target = config
                for part in parts[:-1]:
                    target = target[part]
                target[parts[-1]] = value
        
        return config

    # List of configurations to evaluate
    all_configurations = []
    
    # Create the configs for each selected model
    for model_name in selected_models:
        base_config = create_base_config(model_name)
        
        # Define the list of configuration variants to use (with indices for easy reference)
        configs = [
            # Config 0: High beam + medium temp
            create_config_variant(base_config, "Config 0: High beam + medium temp",
                                num_beams=8, temperature=0.25, top_p=0.95),
            
            # Config 1: High beam + low temp
            create_config_variant(base_config, "Config 1: High beam (12) + low temp (0.2)",
                                num_beams=12, temperature=0.2, top_p=0.92),
            
            # Config 2: Greedy with length penalty
            create_config_variant(base_config, "Config 2: Greedy with length penalty",
                                num_beams=10, do_sample=False, temperature=1.0, 
                                length_penalty=1.05),
            
            # Config 3: Very low temp + repetition penalty
            create_config_variant(base_config, "Config 3: Very low temp + repetition penalty",
                                num_beams=8, temperature=0.15, top_p=0.9, 
                                repetition_penalty=1.1),
            
            # Config 4: High beam + low temp
            create_config_variant(base_config, "Config 4: High beam + low temp",
                                num_beams=10, temperature=0.2, top_p=0.95),
            
            # Config 5: Pure greedy with high beam
            create_config_variant(base_config, "Config 5: Pure greedy with high beam",
                                num_beams=12, do_sample=False, temperature=1.0),
            
            # Config 6: Early stopping + optimal params
            create_config_variant(base_config, "Config 6: Early stopping + optimal params",
                                num_beams=8, temperature=0.18, top_p=0.92, 
                                early_stopping=True),
            
            # Config 7: Diversity penalty
            create_config_variant(base_config, "Config 7: Diversity penalty",
                                num_beams=9, temperature=0.15, diversity_penalty=0.2),
            
            # Config 8: Balanced beam + low temp
            create_config_variant(base_config, "Config 8: Balanced beam + low temp",
                                num_beams=10, temperature=0.2, top_p=0.9),
            
            # Config 9: Pure greedy
            create_config_variant(base_config, "Config 9: Pure greedy",
                                num_beams=8, do_sample=False, temperature=1.0),
            
            # Config 10: High beam + repetition penalty
            create_config_variant(base_config, "Config 10: High beam + repetition penalty",
                                num_beams=12, temperature=0.15, top_p=0.95,
                                repetition_penalty=1.15),
                                
            # Config 11: Balanced params
            create_config_variant(base_config, "Config 11: Balanced params",
                                num_beams=10, temperature=0.22, top_p=0.92),
                                
            # Config 12: Greedy with length penalty (alt)
            create_config_variant(base_config, "Config 12: Greedy with length penalty (alt)",
                                num_beams=12, do_sample=False, temperature=1.0,
                                length_penalty=1.05),
                                
            # Config 13: Very low temp + repetition
            create_config_variant(base_config, "Config 13: Very low temp + repetition",
                                num_beams=8, temperature=0.15, top_p=0.9,
                                repetition_penalty=1.2),
                                
            # Config 14: Best tuned params
            create_config_variant(base_config, "Config 14: Best tuned params",
                                num_beams=10, temperature=0.18, top_p=0.92,
                                repetition_penalty=1.05)
        ]
        
        # Parse config index range if provided
        if config_index_range:
            start, end = (int(x) if x else None for x in config_index_range.split(":"))
            configs = configs[start:end]
        
        # Add the selected configs for this model to the combined list
        all_configurations.extend(configs)
    
    return all_configurations

def deep_copy_config(config):
    """Create a deep copy of a nested dictionary configuration"""
    import copy
    return copy.deepcopy(config)