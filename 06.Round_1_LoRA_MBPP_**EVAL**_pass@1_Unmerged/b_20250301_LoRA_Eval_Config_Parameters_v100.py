def get_evaluation_configurations():
    """
    Returns a list of configurations for LoRA model evaluation experiments
    """
    # Base configuration 
    base_config = {
        "logging": {
            "log_dir": "/home/ubuntu/01.Stanford/01.CS224N/01.Project/06.LoRA_Eval_pass@1_Scripts/logs",
        },
        "output": {
            "results_dir": "/home/ubuntu/01.Stanford/01.CS224N/01.Project/06.LoRA_Eval_pass@1_Scripts/results_all",
        },
        "tracking": {
            "file": "/home/ubuntu/01.Stanford/01.CS224N/01.Project/07.Tracking/lora_evals_tracking.csv",
            "backup_dir": "/home/ubuntu/01.Stanford/01.CS224N/01.Project/07.Tracking/12.Eval_Backup",
        },
        "base_model": {
            "name": "codellama/CodeLlama-7b-hf",
            "dtype": "float16",
            "device_map": "auto",
            "max_position_embeddings": 256,
        },
        "lora_model": {
            "path": "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_01_alpha16_dropout0.05_r8",
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

    # Different temperatures
    config_temp01 = deep_copy_config(base_config)
    config_temp01["evaluation"]["generation"]["temperature"] = 0.1
    config_temp01["evaluation"]["config_description"] = "Lower temperature (0.1)"

    config_temp03 = deep_copy_config(base_config)
    config_temp03["evaluation"]["generation"]["temperature"] = 0.3
    config_temp03["evaluation"]["config_description"] = "Higher temperature (0.3)"

    # Different beam settings
    config_beam1 = deep_copy_config(base_config)
    config_beam1["evaluation"]["generation"]["num_beams"] = 1
    config_beam1["evaluation"]["config_description"] = "Single beam"
    
    config_beam5 = deep_copy_config(base_config)
    config_beam5["evaluation"]["generation"]["num_beams"] = 5
    config_beam5["evaluation"]["config_description"] = "More beams (5)"

    # Top-p variations
    config_topp80 = deep_copy_config(base_config)
    config_topp80["evaluation"]["generation"]["top_p"] = 0.8
    config_topp80["evaluation"]["config_description"] = "Lower top_p (0.8)"
    
    config_topp95 = deep_copy_config(base_config)
    config_topp95["evaluation"]["generation"]["top_p"] = 0.95
    config_topp95["evaluation"]["config_description"] = "Higher top_p (0.95)"

    # Greedy sampling (no randomness)
    config_greedy = deep_copy_config(base_config)
    config_greedy["evaluation"]["generation"]["do_sample"] = False
    config_greedy["evaluation"]["generation"]["temperature"] = 1.0  # Doesn't matter when do_sample=False
    config_greedy["evaluation"]["generation"]["top_p"] = 1.0  # Doesn't matter when do_sample=False
    config_greedy["evaluation"]["config_description"] = "Greedy decoding (no sampling)"

    # More tokens
    config_more_tokens = deep_copy_config(base_config)
    config_more_tokens["evaluation"]["generation"]["max_new_tokens"] = 512
    config_more_tokens["evaluation"]["config_description"] = "More tokens (512)"

    # pass@10 config
    config_pass10 = deep_copy_config(base_config)
    config_pass10["evaluation"]["metric"] = "pass@10"
    config_pass10["evaluation"]["num_samples_per_problem"] = 10
    config_pass10["evaluation"]["generation"]["temperature"] = 0.8  # Higher temperature for diversity
    config_pass10["evaluation"]["generation"]["top_p"] = 0.95
    config_pass10["evaluation"]["generation"]["num_beams"] = 1  # No beam search for diversity
    config_pass10["evaluation"]["config_description"] = "pass@10 configuration"

######### New Set of 50 --- 10 per mdoel #######################

    # Model 1: 20250301_01_alpha16_dropout0.05_r8 configurations

    ###################### Completed ###########################
    # # High Beam + High Temperature (DONE)
    config_m1_1 = deep_copy_config(base_config)
    config_m1_1["evaluation"]["generation"]["num_beams"] = 8
    config_m1_1["evaluation"]["generation"]["temperature"] = 0.4
    config_m1_1["evaluation"]["generation"]["top_p"] = 0.95
    config_m1_1["evaluation"]["config_description"] = "High beam (8) + high temp (0.4)"

    ###################### Completed ###########################
    # # Pure Greedy + High Beam (Completed)
    config_m1_2 = deep_copy_config(base_config)
    config_m1_2["evaluation"]["generation"]["num_beams"] = 10
    config_m1_2["evaluation"]["generation"]["temperature"] = 1.0
    config_m1_2["evaluation"]["generation"]["do_sample"] = False
    config_m1_2["evaluation"]["generation"]["top_p"] = 1.0
    config_m1_2["evaluation"]["config_description"] = "Pure greedy + more beams (10)"

    ###################### Completed ###########################
    # # Ultra Greedy (Completed)
    config_m1_3 = deep_copy_config(base_config)
    config_m1_3["evaluation"]["generation"]["num_beams"] = 1
    config_m1_3["evaluation"]["generation"]["temperature"] = 1.0
    config_m1_3["evaluation"]["generation"]["do_sample"] = False
    config_m1_3["evaluation"]["generation"]["top_p"] = 1.0
    config_m1_3["evaluation"]["generation"]["repetition_penalty"] = 1.2
    config_m1_3["evaluation"]["config_description"] = "Pure greedy + repetition penalty"

    ###################### Completed ###########################
    # # Balanced Temperature (Completed)
    config_m1_4 = deep_copy_config(base_config)
    config_m1_4["evaluation"]["generation"]["num_beams"] = 5
    config_m1_4["evaluation"]["generation"]["temperature"] = 0.25
    config_m1_4["evaluation"]["generation"]["top_p"] = 0.95
    config_m1_4["evaluation"]["config_description"] = "Balanced temp (0.25) + beams (5)"
    
    # Extended Tokens
    config_m1_5 = deep_copy_config(base_config)
    config_m1_5["evaluation"]["generation"]["num_beams"] = 5
    config_m1_5["evaluation"]["generation"]["temperature"] = 0.3
    config_m1_5["evaluation"]["generation"]["top_p"] = 0.95
    config_m1_5["evaluation"]["generation"]["max_new_tokens"] = 384
    config_m1_5["evaluation"]["config_description"] = "Extended tokens (384) + good params"
    
    # Lower Top-K
    config_m1_6 = deep_copy_config(base_config)
    config_m1_6["evaluation"]["generation"]["num_beams"] = 5
    config_m1_6["evaluation"]["generation"]["temperature"] = 0.3
    config_m1_6["evaluation"]["generation"]["top_p"] = 0.95
    config_m1_6["evaluation"]["generation"]["top_k"] = 50
    config_m1_6["evaluation"]["config_description"] = "Top-K (50) + top beams + temp"
    
    # Higher Repetition Penalty
    config_m1_7 = deep_copy_config(base_config)
    config_m1_7["evaluation"]["generation"]["num_beams"] = 5
    config_m1_7["evaluation"]["generation"]["temperature"] = 0.3
    config_m1_7["evaluation"]["generation"]["top_p"] = 0.95
    config_m1_7["evaluation"]["generation"]["repetition_penalty"] = 1.3
    config_m1_7["evaluation"]["config_description"] = "Repetition penalty (1.3) + top params"
    
    # Ultra High Beam
    config_m1_8 = deep_copy_config(base_config)
    config_m1_8["evaluation"]["generation"]["num_beams"] = 15
    config_m1_8["evaluation"]["generation"]["temperature"] = 0.2
    config_m1_8["evaluation"]["generation"]["top_p"] = 0.9
    config_m1_8["evaluation"]["config_description"] = "Ultra high beam (15)"
    
    # Pass@10 Test
    config_m1_9 = deep_copy_config(base_config)
    config_m1_9["evaluation"]["metric"] = "pass@10"
    config_m1_9["evaluation"]["num_samples_per_problem"] = 10
    config_m1_9["evaluation"]["generation"]["temperature"] = 0.8
    config_m1_9["evaluation"]["generation"]["top_p"] = 0.95
    config_m1_9["evaluation"]["generation"]["num_beams"] = 1
    config_m1_9["evaluation"]["config_description"] = "Standard pass@10 configuration"
    
    # Minimum Temperature
    config_m1_10 = deep_copy_config(base_config)
    config_m1_10["evaluation"]["generation"]["num_beams"] = 5
    config_m1_10["evaluation"]["generation"]["temperature"] = 0.05
    config_m1_10["evaluation"]["generation"]["top_p"] = 0.95
    config_m1_10["evaluation"]["config_description"] = "Minimum temperature (0.05) + good params"
    
    # Model 2: 20250301_01_alpha16_dropout0.05_r8_lr0.0002_epochs4 configurations
    
    # First, update base path for model 2
    config_m2_base = deep_copy_config(base_config)
    config_m2_base["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_01_alpha16_dropout0.05_r8_lr0.0002_epochs4"

    ###################### Completed ###########################
    ## High Beam Search (Completed)
    config_m2_1 = deep_copy_config(config_m2_base)
    config_m2_1["evaluation"]["generation"]["num_beams"] = 8
    config_m2_1["evaluation"]["generation"]["temperature"] = 0.3
    config_m2_1["evaluation"]["generation"]["top_p"] = 0.95
    config_m2_1["evaluation"]["config_description"] = "High beam search (8) for 4-epoch model"

    ###################### Completed ###########################
    # # Greedy with High Beam (Completed)
    config_m2_2 = deep_copy_config(config_m2_base)
    config_m2_2["evaluation"]["generation"]["num_beams"] = 10
    config_m2_2["evaluation"]["generation"]["do_sample"] = False
    config_m2_2["evaluation"]["generation"]["temperature"] = 1.0
    config_m2_2["evaluation"]["config_description"] = "Greedy with high beam (10) for 4-epoch model"

    ###################### Completed ###########################
    # Medium Temperature
    config_m2_3 = deep_copy_config(config_m2_base)
    config_m2_3["evaluation"]["generation"]["num_beams"] = 5
    config_m2_3["evaluation"]["generation"]["temperature"] = 0.25
    config_m2_3["evaluation"]["generation"]["top_p"] = 0.95
    config_m2_3["evaluation"]["config_description"] = "Medium temperature for 4-epoch model"
    
    # Low Temperature + High Beams
    config_m2_4 = deep_copy_config(config_m2_base)
    config_m2_4["evaluation"]["generation"]["num_beams"] = 8
    config_m2_4["evaluation"]["generation"]["temperature"] = 0.1
    config_m2_4["evaluation"]["generation"]["top_p"] = 0.9
    config_m2_4["evaluation"]["config_description"] = "Low temp + high beams for 4-epoch model"
    
    # Very High Beam
    config_m2_5 = deep_copy_config(config_m2_base)
    config_m2_5["evaluation"]["generation"]["num_beams"] = 12
    config_m2_5["evaluation"]["generation"]["temperature"] = 0.2
    config_m2_5["evaluation"]["generation"]["top_p"] = 0.9
    config_m2_5["evaluation"]["config_description"] = "Very high beam search for 4-epoch model"
    
    # High Top-K
    config_m2_6 = deep_copy_config(config_m2_base)
    config_m2_6["evaluation"]["generation"]["num_beams"] = 5
    config_m2_6["evaluation"]["generation"]["temperature"] = 0.3
    config_m2_6["evaluation"]["generation"]["top_p"] = 0.95
    config_m2_6["evaluation"]["generation"]["top_k"] = 30
    config_m2_6["evaluation"]["config_description"] = "High top-k for 4-epoch model"
    
    # High Top-P
    config_m2_7 = deep_copy_config(config_m2_base)
    config_m2_7["evaluation"]["generation"]["num_beams"] = 3
    config_m2_7["evaluation"]["generation"]["temperature"] = 0.3
    config_m2_7["evaluation"]["generation"]["top_p"] = 0.98
    config_m2_7["evaluation"]["config_description"] = "Very high top-p for 4-epoch model"
    
    # Extended Generation
    config_m2_8 = deep_copy_config(config_m2_base)
    config_m2_8["evaluation"]["generation"]["num_beams"] = 5
    config_m2_8["evaluation"]["generation"]["temperature"] = 0.2
    config_m2_8["evaluation"]["generation"]["max_new_tokens"] = 512
    config_m2_8["evaluation"]["config_description"] = "Extended tokens for 4-epoch model"
    
    # Pure Greedy
    config_m2_9 = deep_copy_config(config_m2_base)
    config_m2_9["evaluation"]["generation"]["num_beams"] = 1
    config_m2_9["evaluation"]["generation"]["do_sample"] = False
    config_m2_9["evaluation"]["generation"]["temperature"] = 1.0
    config_m2_9["evaluation"]["config_description"] = "Pure greedy for 4-epoch model"
    
    # Pass@10 Evaluation
    config_m2_10 = deep_copy_config(config_m2_base)
    config_m2_10["evaluation"]["metric"] = "pass@10"
    config_m2_10["evaluation"]["num_samples_per_problem"] = 10
    config_m2_10["evaluation"]["generation"]["temperature"] = 0.8
    config_m2_10["evaluation"]["generation"]["top_p"] = 0.95
    config_m2_10["evaluation"]["config_description"] = "Pass@10 for 4-epoch model"
    
    # Model 3: 20250301_01_alpha16_dropout0.05_r16_lr0.0002_epochs3 configurations
    
    # First, update base path for model 3
    config_m3_base = deep_copy_config(base_config)
    config_m3_base["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_01_alpha16_dropout0.05_r16_lr0.0002_epochs3"

    ###################### Completed ###########################  34.76 (High beam + medium temp for r16 model) 
    # # High Beam + Medium Temp (Completed)
    config_m3_1 = deep_copy_config(config_m3_base)
    config_m3_1["evaluation"]["generation"]["num_beams"] = 8
    config_m3_1["evaluation"]["generation"]["temperature"] = 0.25
    config_m3_1["evaluation"]["generation"]["top_p"] = 0.95
    config_m3_1["evaluation"]["config_description"] = "High beam + medium temp for r16 model"

    ###################### Completed ###########################
    # # Pure Greedy with Beams (Completed)
    config_m3_2 = deep_copy_config(config_m3_base)
    config_m3_2["evaluation"]["generation"]["num_beams"] = 8
    config_m3_2["evaluation"]["generation"]["do_sample"] = False
    config_m3_2["evaluation"]["generation"]["temperature"] = 1.0
    config_m3_2["evaluation"]["config_description"] = "Pure greedy with beams for r16 model"

    ###################### Completed ###########################
    # # Low Temperature (Completed)
    config_m3_3 = deep_copy_config(config_m3_base)
    config_m3_3["evaluation"]["generation"]["num_beams"] = 5
    config_m3_3["evaluation"]["generation"]["temperature"] = 0.1
    config_m3_3["evaluation"]["generation"]["top_p"] = 0.9
    config_m3_3["evaluation"]["config_description"] = "Low temperature for r16 model"

    ###################### Completed ###########################
    # High Temperature
    config_m3_4 = deep_copy_config(config_m3_base)
    config_m3_4["evaluation"]["generation"]["num_beams"] = 3
    config_m3_4["evaluation"]["generation"]["temperature"] = 0.4
    config_m3_4["evaluation"]["generation"]["top_p"] = 0.95
    config_m3_4["evaluation"]["config_description"] = "High temperature for r16 model"
    
    # Aggressive Beam Search
    config_m3_5 = deep_copy_config(config_m3_base)
    config_m3_5["evaluation"]["generation"]["num_beams"] = 15
    config_m3_5["evaluation"]["generation"]["temperature"] = 0.2
    config_m3_5["evaluation"]["generation"]["top_p"] = 0.9
    config_m3_5["evaluation"]["config_description"] = "Aggressive beam search for r16 model"
    
    # Balanced Parameters
    config_m3_6 = deep_copy_config(config_m3_base)
    config_m3_6["evaluation"]["generation"]["num_beams"] = 5
    config_m3_6["evaluation"]["generation"]["temperature"] = 0.25
    config_m3_6["evaluation"]["generation"]["top_p"] = 0.92
    config_m3_6["evaluation"]["config_description"] = "Balanced parameters for r16 model"
    
    # Top-K Filtering
    config_m3_7 = deep_copy_config(config_m3_base)
    config_m3_7["evaluation"]["generation"]["num_beams"] = 5
    config_m3_7["evaluation"]["generation"]["temperature"] = 0.3
    config_m3_7["evaluation"]["generation"]["top_p"] = 0.95
    config_m3_7["evaluation"]["generation"]["top_k"] = 40
    config_m3_7["evaluation"]["config_description"] = "Top-k filtering for r16 model"
    
    # Extended Generation
    config_m3_8 = deep_copy_config(config_m3_base)
    config_m3_8["evaluation"]["generation"]["num_beams"] = 5
    config_m3_8["evaluation"]["generation"]["temperature"] = 0.25
    config_m3_8["evaluation"]["generation"]["max_new_tokens"] = 384
    config_m3_8["evaluation"]["config_description"] = "Extended generation for r16 model"
    
    # Pure Greedy
    config_m3_9 = deep_copy_config(config_m3_base)
    config_m3_9["evaluation"]["generation"]["num_beams"] = 1
    config_m3_9["evaluation"]["generation"]["do_sample"] = False
    config_m3_9["evaluation"]["generation"]["temperature"] = 1.0
    config_m3_9["evaluation"]["config_description"] = "Pure greedy for r16 model"

    ###################### Completed ########################### 34.76 (High beam + medium temp for r16 model)
    # Pass@10 Evaluation 
    config_m3_10 = deep_copy_config(config_m3_base)
    config_m3_10["evaluation"]["metric"] = "pass@10"
    config_m3_10["evaluation"]["num_samples_per_problem"] = 10
    config_m3_10["evaluation"]["generation"]["temperature"] = 0.8
    config_m3_10["evaluation"]["generation"]["top_p"] = 0.95
    config_m3_10["evaluation"]["config_description"] = "Pass@10 for r16 model"
    
    # Model 4: 20250301_01_alpha32_dropout0.05_r8_lr0.0002_epochs3 configurations
    
    # First, update base path for model 4
    config_m4_base = deep_copy_config(base_config)
    config_m4_base["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_01_alpha32_dropout0.05_r8_lr0.0002_epochs3"

    ###################### Completed ###########################
    # # Balanced Beam Search (Completed)
    config_m4_1 = deep_copy_config(config_m4_base)
    config_m4_1["evaluation"]["generation"]["num_beams"] = 8
    config_m4_1["evaluation"]["generation"]["temperature"] = 0.3
    config_m4_1["evaluation"]["generation"]["top_p"] = 0.95
    config_m4_1["evaluation"]["config_description"] = "Balanced beam search for alpha32 model"

    ###################### Completed ###########################
    # # Pure Greedy + High Beam (Completed)
    config_m4_2 = deep_copy_config(config_m4_base)
    config_m4_2["evaluation"]["generation"]["num_beams"] = 10
    config_m4_2["evaluation"]["generation"]["do_sample"] = False
    config_m4_2["evaluation"]["generation"]["temperature"] = 1.0
    config_m4_2["evaluation"]["config_description"] = "Pure greedy + high beam for alpha32 model"

    ###################### Completed ###########################
    # # Lower Temperature (Completed)
    config_m4_3 = deep_copy_config(config_m4_base)
    config_m4_3["evaluation"]["generation"]["num_beams"] = 5
    config_m4_3["evaluation"]["generation"]["temperature"] = 0.15
    config_m4_3["evaluation"]["generation"]["top_p"] = 0.9
    config_m4_3["evaluation"]["config_description"] = "Lower temperature for alpha32 model"

    ###################### Completed ###########################
    # Higher Temperature
    config_m4_4 = deep_copy_config(config_m4_base)
    config_m4_4["evaluation"]["generation"]["num_beams"] = 3
    config_m4_4["evaluation"]["generation"]["temperature"] = 0.35
    config_m4_4["evaluation"]["generation"]["top_p"] = 0.95
    config_m4_4["evaluation"]["config_description"] = "Higher temperature for alpha32 model"
    
    # Aggressive Beam Search
    config_m4_5 = deep_copy_config(config_m4_base)
    config_m4_5["evaluation"]["generation"]["num_beams"] = 12
    config_m4_5["evaluation"]["generation"]["temperature"] = 0.2
    config_m4_5["evaluation"]["generation"]["top_p"] = 0.9
    config_m4_5["evaluation"]["config_description"] = "Aggressive beam search for alpha32 model"
    
    # High Top-P
    config_m4_6 = deep_copy_config(config_m4_base)
    config_m4_6["evaluation"]["generation"]["num_beams"] = 5
    config_m4_6["evaluation"]["generation"]["temperature"] = 0.25
    config_m4_6["evaluation"]["generation"]["top_p"] = 0.98
    config_m4_6["evaluation"]["config_description"] = "High top-p for alpha32 model"
    
    # Top-K Filtering
    config_m4_7 = deep_copy_config(config_m4_base)
    config_m4_7["evaluation"]["generation"]["num_beams"] = 5
    config_m4_7["evaluation"]["generation"]["temperature"] = 0.25
    config_m4_7["evaluation"]["generation"]["top_k"] = 40
    config_m4_7["evaluation"]["generation"]["top_p"] = 0.95
    config_m4_7["evaluation"]["config_description"] = "Top-k filtering for alpha32 model"
    
    # Extended Generation
    config_m4_8 = deep_copy_config(config_m4_base)
    config_m4_8["evaluation"]["generation"]["num_beams"] = 5
    config_m4_8["evaluation"]["generation"]["temperature"] = 0.25
    config_m4_8["evaluation"]["generation"]["max_new_tokens"] = 384
    config_m4_8["evaluation"]["config_description"] = "Extended generation for alpha32 model"
    
    # Simple Greedy
    config_m4_9 = deep_copy_config(config_m4_base)
    config_m4_9["evaluation"]["generation"]["num_beams"] = 1
    config_m4_9["evaluation"]["generation"]["do_sample"] = False
    config_m4_9["evaluation"]["generation"]["temperature"] = 1.0
    config_m4_9["evaluation"]["config_description"] = "Simple greedy for alpha32 model"
    
    # Pass@10 Evaluation
    config_m4_10 = deep_copy_config(config_m4_base)
    config_m4_10["evaluation"]["metric"] = "pass@10"
    config_m4_10["evaluation"]["num_samples_per_problem"] = 10
    config_m4_10["evaluation"]["generation"]["temperature"] = 0.8
    config_m4_10["evaluation"]["generation"]["top_p"] = 0.95
    config_m4_10["evaluation"]["config_description"] = "Pass@10 for alpha32 model"
    
    # Model 5: 20250301_02_alpha16_dropout0.0_r8_lr0.0002_epochs3 configurations
    
    # First, update base path for model 5
    config_m5_base = deep_copy_config(base_config)
    config_m5_base["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_02_alpha16_dropout0.0_r8_lr0.0002_epochs3"

    ###################### Completed ###########################
    # # High Beam + Medium Temp (Completed)
    config_m5_1 = deep_copy_config(config_m5_base)
    config_m5_1["evaluation"]["generation"]["num_beams"] = 8
    config_m5_1["evaluation"]["generation"]["temperature"] = 0.25
    config_m5_1["evaluation"]["generation"]["top_p"] = 0.95
    config_m5_1["evaluation"]["config_description"] = "High beam + medium temp for no-dropout model"

    ###################### Completed ###########################
    # # Pure Greedy with Beams (Completed)
    config_m5_2 = deep_copy_config(config_m5_base)
    config_m5_2["evaluation"]["generation"]["num_beams"] = 8
    config_m5_2["evaluation"]["generation"]["do_sample"] = False
    config_m5_2["evaluation"]["generation"]["temperature"] = 1.0
    config_m5_2["evaluation"]["config_description"] = "Pure greedy with beams for no-dropout model"
    
    # Low Temperature
    config_m5_3 = deep_copy_config(config_m5_base)
    config_m5_3["evaluation"]["generation"]["num_beams"] = 5
    config_m5_3["evaluation"]["generation"]["temperature"] = 0.1
    config_m5_3["evaluation"]["generation"]["top_p"] = 0.9
    config_m5_3["evaluation"]["config_description"] = "Low temperature for no-dropout model"
    
    # High Temperature
    config_m5_4 = deep_copy_config(config_m5_base)
    config_m5_4["evaluation"]["generation"]["num_beams"] = 3
    config_m5_4["evaluation"]["generation"]["temperature"] = 0.4
    config_m5_4["evaluation"]["generation"]["top_p"] = 0.95
    config_m5_4["evaluation"]["config_description"] = "High temperature for no-dropout model"
    
    # Aggressive Beam Search
    config_m5_5 = deep_copy_config(config_m5_base)
    config_m5_5["evaluation"]["generation"]["num_beams"] = 15
    config_m5_5["evaluation"]["generation"]["temperature"] = 0.2
    config_m5_5["evaluation"]["generation"]["top_p"] = 0.9
    config_m5_5["evaluation"]["config_description"] = "Aggressive beam search for no-dropout model"
    
    # Balanced Parameters
    config_m5_6 = deep_copy_config(config_m5_base)
    config_m5_6["evaluation"]["generation"]["num_beams"] = 5
    config_m5_6["evaluation"]["generation"]["temperature"] = 0.25
    config_m5_6["evaluation"]["generation"]["top_p"] = 0.92
    config_m5_6["evaluation"]["config_description"] = "Balanced parameters for no-dropout model"
    
    # Top-K Filtering
    config_m5_7 = deep_copy_config(config_m5_base)
    config_m5_7["evaluation"]["generation"]["num_beams"] = 5
    config_m5_7["evaluation"]["generation"]["temperature"] = 0.3
    config_m5_7["evaluation"]["generation"]["top_p"] = 0.95
    config_m5_7["evaluation"]["generation"]["top_k"] = 40
    config_m5_7["evaluation"]["config_description"] = "Top-k filtering for no-dropout model"
    
    # Extended Generation
    config_m5_8 = deep_copy_config(config_m5_base)
    config_m5_8["evaluation"]["generation"]["num_beams"] = 5
    config_m5_8["evaluation"]["generation"]["temperature"] = 0.25
    config_m5_8["evaluation"]["generation"]["max_new_tokens"] = 384
    config_m5_8["evaluation"]["config_description"] = "Extended generation for no-dropout model"
    
    # Pure Greedy
    config_m5_9 = deep_copy_config(config_m5_base)
    config_m5_9["evaluation"]["generation"]["num_beams"] = 1
    config_m5_9["evaluation"]["generation"]["do_sample"] = False
    config_m5_9["evaluation"]["generation"]["temperature"] = 1.0
    config_m5_9["evaluation"]["config_description"] = "Pure greedy for no-dropout model"
    
    # Pass@10 Evaluation
    config_m5_10 = deep_copy_config(config_m5_base)
    config_m5_10["evaluation"]["metric"] = "pass@10"
    config_m5_10["evaluation"]["num_samples_per_problem"] = 10
    config_m5_10["evaluation"]["generation"]["temperature"] = 0.8
    config_m5_10["evaluation"]["generation"]["top_p"] = 0.95
    config_m5_10["evaluation"]["config_description"] = "Pass@10 for no-dropout model"

    ###########################################################################################
    # Model 6: Consolidated Optimized Configurations --> Revised list based on 34.76 results 

    ###################### Completed ###########################
    # 60: Ultra-High Beam with Optimal Temperature
    config_m6_1 = deep_copy_config(base_config)
    config_m6_1["evaluation"]["generation"]["num_beams"] = 12
    config_m6_1["evaluation"]["generation"]["temperature"] = 0.25
    config_m6_1["evaluation"]["generation"]["top_p"] = 0.95
    config_m6_1["evaluation"]["config_description"] = "Ultra-High Beam (12) with Optimal Temperature (0.25)"
    ###################### Completed ###########################
    # 61: Hybrid Beam-Greedy
    config_m6_2 = deep_copy_config(base_config)
    config_m6_2["evaluation"]["generation"]["num_beams"] = 10
    config_m6_2["evaluation"]["generation"]["temperature"] = 1.0
    config_m6_2["evaluation"]["generation"]["top_p"] = 1.0
    config_m6_2["evaluation"]["generation"]["do_sample"] = False
    config_m6_2["evaluation"]["generation"]["length_penalty"] = 1.05
    config_m6_2["evaluation"]["config_description"] = "Hybrid Beam-Greedy with Length Penalty"

    ###################### Completed ###########################
    # 62: R16 + Repetition Penalty
    config_m6_3 = deep_copy_config(base_config)
    config_m6_3["evaluation"]["generation"]["num_beams"] = 8
    config_m6_3["evaluation"]["generation"]["temperature"] = 0.25
    config_m6_3["evaluation"]["generation"]["repetition_penalty"] = 1.1
    config_m6_3["evaluation"]["config_description"] = "R16 with Repetition Penalty (1.1)"

    ###################### Completed ###########################
    # 63: R16 + No Dropout + High Beam
    config_m6_4 = deep_copy_config(base_config)
    config_m6_4["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_02_alpha16_dropout0.0_r8_lr0.0002_epochs3"
    config_m6_4["evaluation"]["generation"]["num_beams"] = 10
    config_m6_4["evaluation"]["generation"]["temperature"] = 0.25
    config_m6_4["evaluation"]["config_description"] = "No Dropout Model with High Beam (10)"

    ###################### Completed ###########################
    # 64: Lower Temperature for Precision
    config_m6_5 = deep_copy_config(base_config)
    config_m6_5["evaluation"]["generation"]["num_beams"] = 8
    config_m6_5["evaluation"]["generation"]["temperature"] = 0.2
    config_m6_5["evaluation"]["config_description"] = "Lower Temperature (0.2) for Precision"
    
    ###################### Completed ###########################
    # 65: Precision Focus with Top-K
    config_m6_6 = deep_copy_config(base_config)
    config_m6_6["evaluation"]["generation"]["num_beams"] = 8
    config_m6_6["evaluation"]["generation"]["temperature"] = 0.2
    config_m6_6["evaluation"]["generation"]["top_p"] = 0.92
    config_m6_6["evaluation"]["generation"]["top_k"] = 40
    config_m6_6["evaluation"]["config_description"] = "Precision Focus with Top-K (40)"

    ###################### Completed ###########################
    # 66: Extended Generation with Focused Parameters
    config_m6_7 = deep_copy_config(base_config)
    config_m6_7["evaluation"]["generation"]["num_beams"] = 8
    config_m6_7["evaluation"]["generation"]["max_new_tokens"] = 384
    config_m6_7["evaluation"]["config_description"] = "Extended Generation (384 tokens)"

    ###################### Completed ###########################
    # 67: Alpha32 Model with Optimal Parameters
    config_m6_8 = deep_copy_config(base_config)
    config_m6_8["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_01_alpha32_dropout0.05_r8_lr0.0002_epochs3"
    config_m6_8["evaluation"]["generation"]["num_beams"] = 8
    config_m6_8["evaluation"]["generation"]["temperature"] = 0.25
    config_m6_8["evaluation"]["config_description"] = "Alpha-32 Model with Optimal Parameters"

    # 68: Very Focused Beam Search
    config_m6_9 = deep_copy_config(base_config)
    config_m6_9["evaluation"]["generation"]["num_beams"] = 15
    config_m6_9["evaluation"]["generation"]["temperature"] = 0.15
    config_m6_9["evaluation"]["generation"]["length_penalty"] = 1.0
    config_m6_9["evaluation"]["config_description"] = "Very Focused Beam Search (15 beams, 0.15 temp)"

    # 69: Mixed Strategy
    config_m6_10 = deep_copy_config(base_config)
    config_m6_10["evaluation"]["generation"]["num_beams"] = 9
    config_m6_10["evaluation"]["generation"]["temperature"] = 0.22
    config_m6_10["evaluation"]["generation"]["top_p"] = 0.93
    config_m6_10["evaluation"]["generation"]["no_repeat_ngram_size"] = 3
    config_m6_10["evaluation"]["config_description"] = "Mixed Strategy (9 beams, 0.22 temp, no repeat ngram)"

    # 70: Pass@10 with Diversity Encouragement
    config_m6_11 = deep_copy_config(base_config)
    config_m6_11["evaluation"]["metric"] = "pass@10"
    config_m6_11["evaluation"]["num_samples_per_problem"] = 10
    config_m6_11["evaluation"]["generation"]["temperature"] = 0.7
    config_m6_11["evaluation"]["generation"]["top_p"] = 0.98
    config_m6_11["evaluation"]["generation"]["num_beams"] = 3
    config_m6_11["evaluation"]["config_description"] = "Pass@10 with higher diversity (temp 0.7, top_p 0.98)"

    # 71: Pass@10 with Beam Search
    config_m6_12 = deep_copy_config(base_config)
    config_m6_12["evaluation"]["metric"] = "pass@10"
    config_m6_12["evaluation"]["num_samples_per_problem"] = 10
    config_m6_12["evaluation"]["generation"]["temperature"] = 0.6
    config_m6_12["evaluation"]["generation"]["num_beams"] = 8
    config_m6_12["evaluation"]["config_description"] = "Pass@10 with num_beams (8)"

    ################ Added in final round on 20240302 - Night ######## before calling it a night

    ###################### Completed ########################### 39.02 (QKVO model - High beam (12) + low temp (0.2))
    # Group 7: Focus on the q,k,v,o model (20250301_03_alpha16_dropout0.05_r8_lr0.0002_epochs3)
    config_m7_1 = deep_copy_config(base_config)
    config_m7_1["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_03_alpha16_dropout0.05_r8_lr0.0002_epochs3"
    config_m7_1["evaluation"]["generation"]["num_beams"] = 12
    config_m7_1["evaluation"]["generation"]["temperature"] = 0.2
    config_m7_1["evaluation"]["generation"]["top_p"] = 0.92
    config_m7_1["evaluation"]["config_description"] = "QKVO model - High beam (12) + low temp (0.2)"
    
    ###################### Completed ########################### 38.65 (QKVO model - Greedy with length penalty)
    config_m7_2 = deep_copy_config(base_config)
    config_m7_2["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_03_alpha16_dropout0.05_r8_lr0.0002_epochs3"
    config_m7_2["evaluation"]["generation"]["num_beams"] = 10
    config_m7_2["evaluation"]["generation"]["do_sample"] = False
    config_m7_2["evaluation"]["generation"]["temperature"] = 1.0
    config_m7_2["evaluation"]["generation"]["length_penalty"] = 1.05
    config_m7_2["evaluation"]["config_description"] = "QKVO model - Greedy with length penalty"
    
    ###################### COMPLETE ## Highest 40.1 ######################### 40.49 (QKVO model - Very low temp + repetition penalty)
    config_m7_3 = deep_copy_config(base_config)
    config_m7_3["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_03_alpha16_dropout0.05_r8_lr0.0002_epochs3" 
    config_m7_3["evaluation"]["generation"]["num_beams"] = 8
    config_m7_3["evaluation"]["generation"]["temperature"] = 0.15
    config_m7_3["evaluation"]["generation"]["top_p"] = 0.9
    config_m7_3["evaluation"]["generation"]["repetition_penalty"] = 1.1
    config_m7_3["evaluation"]["config_description"] = "QKVO model - Very low temp + repetition penalty"
    
    config_m7_4 = deep_copy_config(base_config)
    config_m7_4["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_03_alpha16_dropout0.05_r8_lr0.0002_epochs3"
    config_m7_4["evaluation"]["generation"]["num_beams"] = 15
    config_m7_4["evaluation"]["generation"]["temperature"] = 0.25
    config_m7_4["evaluation"]["generation"]["max_new_tokens"] = 384
    config_m7_4["evaluation"]["config_description"] = "QKVO model - Ultra high beam + extended tokens"
    
    config_m7_5 = deep_copy_config(base_config)
    config_m7_5["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_03_alpha16_dropout0.05_r8_lr0.0002_epochs3"
    config_m7_5["evaluation"]["generation"]["num_beams"] = 10
    config_m7_5["evaluation"]["generation"]["temperature"] = 0.2
    config_m7_5["evaluation"]["generation"]["top_k"] = 40
    config_m7_5["evaluation"]["generation"]["no_repeat_ngram_size"] = 3
    config_m7_5["evaluation"]["config_description"] = "QKVO model - Top-k + no_repeat_ngram"
    
    ###################### Completed ########################### 34.76 (QKV model - High beam + low temp)
    # Group 8: Focus on the QKV model (20250301_04_alpha16_dropout0.05_r8_lr0.0002_epochs3)
    config_m8_1 = deep_copy_config(base_config)
    config_m8_1["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_04_alpha16_dropout0.05_r8_lr0.0002_epochs3"
    config_m8_1["evaluation"]["generation"]["num_beams"] = 10
    config_m8_1["evaluation"]["generation"]["temperature"] = 0.2
    config_m8_1["evaluation"]["generation"]["top_p"] = 0.95
    config_m8_1["evaluation"]["config_description"] = "QKV model - High beam + low temp"
    
    ###################### Completed ###########################
    config_m8_2 = deep_copy_config(base_config)
    config_m8_2["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_04_alpha16_dropout0.05_r8_lr0.0002_epochs3"
    config_m8_2["evaluation"]["generation"]["num_beams"] = 12
    config_m8_2["evaluation"]["generation"]["do_sample"] = False
    config_m8_2["evaluation"]["generation"]["temperature"] = 1.0
    config_m8_2["evaluation"]["config_description"] = "QKV model - Pure greedy with high beam"
    
    ###################### Completed ########################### 35.37 (QKV model - Early stopping + optimal params)
    config_m8_3 = deep_copy_config(base_config)
    config_m8_3["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_04_alpha16_dropout0.05_r8_lr0.0002_epochs3"
    config_m8_3["evaluation"]["generation"]["num_beams"] = 8
    config_m8_3["evaluation"]["generation"]["temperature"] = 0.18
    config_m8_3["evaluation"]["generation"]["top_p"] = 0.92
    config_m8_3["evaluation"]["generation"]["early_stopping"] = True
    config_m8_3["evaluation"]["config_description"] = "QKV model - Early stopping + optimal params"
    
    config_m8_4 = deep_copy_config(base_config)
    config_m8_4["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_04_alpha16_dropout0.05_r8_lr0.0002_epochs3"
    config_m8_4["evaluation"]["generation"]["num_beams"] = 15
    config_m8_4["evaluation"]["generation"]["temperature"] = 0.22
    config_m8_4["evaluation"]["generation"]["top_p"] = 0.9
    config_m8_4["evaluation"]["generation"]["max_new_tokens"] = 384
    config_m8_4["evaluation"]["config_description"] = "QKV model - Ultra high beam + more tokens"
    
    config_m8_5 = deep_copy_config(base_config) 35.98 (QKV model - Diversity penalty) 
    config_m8_5["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_04_alpha16_dropout0.05_r8_lr0.0002_epochs3"
    config_m8_5["evaluation"]["generation"]["num_beams"] = 9
    config_m8_5["evaluation"]["generation"]["temperature"] = 0.15
    config_m8_5["evaluation"]["generation"]["diversity_penalty"] = 0.2
    config_m8_5["evaluation"]["config_description"] = "QKV model - Diversity penalty"
    
    ###################### Completed ###########################
    # Group 9: Focus on high-rank PSZ model (20250301_psz_alpha16_dropout0.05_r32_lr0.0002_epochs3)
    config_m9_1 = deep_copy_config(base_config)
    config_m9_1["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_psz_alpha16_dropout0.05_r32_lr0.0002_epochs3"
    config_m9_1["evaluation"]["generation"]["num_beams"] = 10
    config_m9_1["evaluation"]["generation"]["temperature"] = 0.2
    config_m9_1["evaluation"]["generation"]["top_p"] = 0.9
    config_m9_1["evaluation"]["config_description"] = "R32 model - Balanced beam + low temp"
    
    ###################### Completed ###########################
    config_m9_2 = deep_copy_config(base_config)
    config_m9_2["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_psz_alpha16_dropout0.05_r32_lr0.0002_epochs3"
    config_m9_2["evaluation"]["generation"]["num_beams"] = 8
    config_m9_2["evaluation"]["generation"]["do_sample"] = False
    config_m9_2["evaluation"]["generation"]["temperature"] = 1.0
    config_m9_2["evaluation"]["config_description"] = "R32 model - Pure greedy"
    
    ###################### Completed ###########################
    config_m9_3 = deep_copy_config(base_config)
    config_m9_3["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_psz_alpha16_dropout0.05_r32_lr0.0002_epochs3"
    config_m9_3["evaluation"]["generation"]["num_beams"] = 12
    config_m9_3["evaluation"]["generation"]["temperature"] = 0.15
    config_m9_3["evaluation"]["generation"]["top_p"] = 0.95
    config_m9_3["evaluation"]["generation"]["repetition_penalty"] = 1.15
    config_m9_3["evaluation"]["config_description"] = "R32 model - High beam + repetition penalty"
    
    config_m9_4 = deep_copy_config(base_config)
    config_m9_4["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_psz_alpha16_dropout0.05_r32_lr0.0002_epochs3"
    config_m9_4["evaluation"]["generation"]["num_beams"] = 15
    config_m9_4["evaluation"]["generation"]["temperature"] = 0.25
    config_m9_4["evaluation"]["generation"]["max_new_tokens"] = 384
    config_m9_4["evaluation"]["config_description"] = "R32 model - Ultra high beam + extended tokens"
    
    config_m9_5 = deep_copy_config(base_config)
    config_m9_5["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_psz_alpha16_dropout0.05_r32_lr0.0002_epochs3"
    config_m9_5["evaluation"]["generation"]["num_beams"] = 10
    config_m9_5["evaluation"]["generation"]["temperature"] = 0.18
    config_m9_5["evaluation"]["generation"]["top_k"] = 30
    config_m9_5["evaluation"]["generation"]["no_repeat_ngram_size"] = 3
    config_m9_5["evaluation"]["config_description"] = "R32 model - Top-k + no_repeat_ngram"
    
    ###################### Completed ###########################
    # Group 10: Focus on the extended model (20250301_05_alpha16_dropout0.05_r8_lr0.0002_epochs3)
    config_m10_1 = deep_copy_config(base_config)
    config_m10_1["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_05_alpha16_dropout0.05_r8_lr0.0002_epochs3"
    config_m10_1["evaluation"]["generation"]["num_beams"] = 10
    config_m10_1["evaluation"]["generation"]["temperature"] = 0.22
    config_m10_1["evaluation"]["generation"]["top_p"] = 0.92
    config_m10_1["evaluation"]["config_description"] = "Extended model - Balanced params"
    
    ###################### Completed ########################### 35.37 (Extended model - Greedy with length penalty)
    config_m10_2 = deep_copy_config(base_config)
    config_m10_2["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_05_alpha16_dropout0.05_r8_lr0.0002_epochs3"
    config_m10_2["evaluation"]["generation"]["num_beams"] = 12
    config_m10_2["evaluation"]["generation"]["do_sample"] = False
    config_m10_2["evaluation"]["generation"]["temperature"] = 1.0
    config_m10_2["evaluation"]["generation"]["length_penalty"] = 1.05
    config_m10_2["evaluation"]["config_description"] = "Extended model - Greedy with length penalty"
    
    ###################### Completed ###########################
    config_m10_3 = deep_copy_config(base_config)
    config_m10_3["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_05_alpha16_dropout0.05_r8_lr0.0002_epochs3"
    config_m10_3["evaluation"]["generation"]["num_beams"] = 8
    config_m10_3["evaluation"]["generation"]["temperature"] = 0.15
    config_m10_3["evaluation"]["generation"]["top_p"] = 0.9
    config_m10_3["evaluation"]["generation"]["repetition_penalty"] = 1.2
    config_m10_3["evaluation"]["config_description"] = "Extended model - Very low temp + repetition"
    
    config_m10_4 = deep_copy_config(base_config)
    config_m10_4["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_05_alpha16_dropout0.05_r8_lr0.0002_epochs3"
    config_m10_4["evaluation"]["generation"]["num_beams"] = 15
    config_m10_4["evaluation"]["generation"]["temperature"] = 0.2
    config_m10_4["evaluation"]["generation"]["max_new_tokens"] = 384
    config_m10_4["evaluation"]["config_description"] = "Extended model - Ultra high beam + more tokens"
    
    config_m10_5 = deep_copy_config(base_config)
    config_m10_5["lora_model"]["path"] = "/home/ubuntu/01.Stanford/01.CS224N/01.Project/08.Models/20250301_05_alpha16_dropout0.05_r8_lr0.0002_epochs3"
    config_m10_5["evaluation"]["generation"]["num_beams"] = 10
    config_m10_5["evaluation"]["generation"]["temperature"] = 0.25
    config_m10_5["evaluation"]["generation"]["top_k"] = 50
    config_m10_5["evaluation"]["generation"]["diversity_penalty"] = 0.2
    config_m10_5["evaluation"]["config_description"] = "Extended model - Top-k + diversity"


    
    ################################################################################################# 

    # Return all configurations
    return [
        base_config,       # 0: Baseline
        config_temp01,     # 1: Lower temperature
        config_temp03,     # 2: Higher temperature 
        config_beam1,      # 3: Single beam
        config_beam5,      # 4: More beams
        config_topp80,     # 5: Lower top-p
        config_topp95,     # 6: Higher top-p
        config_greedy,     # 7: Greedy decoding
        config_more_tokens,# 8: More tokens
        config_pass10,     # 9: pass@10 configuration

        # Model 1 configs
        config_m1_1,   # 10 ** COMPLETED **
        config_m1_2,   # 11 ** COMPLETED **
        config_m1_3,   # 12 ** COMPLETED **
        config_m1_4,   # 13 ** COMPLETED **
        config_m1_5,   # 14
        config_m1_6,   # 15
        config_m1_7,   # 16
        config_m1_8,   # 17
        config_m1_9,   # 18
        config_m1_10,  # 19
        
        # Model 2 configs 
        config_m2_1,   # 20 ** COMPLETED **
        config_m2_2,   # 21 ** COMPLETED **
        config_m2_3,   # 22 ** COMPLETED **
        config_m2_4,   # 23
        config_m2_5,   # 24
        config_m2_6,   # 25
        config_m2_7,   # 26
        config_m2_8,   # 27
        config_m2_9,   # 28
        config_m2_10,  # 29
        
        # Model 3 configs
        config_m3_1,   # 30 ** COMPLETED **  34.76 (High beam + medium temp for r16 model)
        config_m3_2,   # 31 ** COMPLETED **
        config_m3_3,   # 32 ** COMPLETED **
        config_m3_4,   # 33 ** COMPLETED **
        config_m3_5,   # 34
        config_m3_6,   # 35
        config_m3_7,   # 36
        config_m3_8,   # 37
        config_m3_9,   # 38
        config_m3_10,  # 39
        
        # Model 4 configs
        config_m4_1,   # 40 ** COMPLETED **
        config_m4_2,   # 41 ** COMPLETED **
        config_m4_3,   # 42 ** COMPLETED **
        config_m4_4,   # 43
        config_m4_5,   # 44
        config_m4_6,   # 45
        config_m4_7,   # 46
        config_m4_8,   # 47
        config_m4_9,   # 48
        config_m4_10,  # 49
        
        # Model 5 configs
        config_m5_1,   # 50 ** COMPLETED **
        config_m5_2,   # 51 ** COMPLETED **
        config_m5_3,   # 52
        config_m5_4,   # 53
        config_m5_5,   # 54
        config_m5_6,   # 55
        config_m5_7,   # 56
        config_m5_8,   # 57
        config_m5_9,   # 58
        config_m5_10,  # 59

        ############ Added as consolidated 2nd round of tests ############
        # Model 6 Consolidated Configs --> new set based on improving 34.76%
        config_m6_1,   # 60 ** COMPLETED **
        config_m6_2,   # 61 ** COMPLETED **
        config_m6_3,   # 62 ** COMPLETED **
        config_m6_4,   # 63 ** COMPLETED **
        config_m6_5,   # 64 ** COMPLETED **
        config_m6_6,   # 65 ** COMPLETED **
        config_m6_7,   # 66 ** COMPLETED **
        config_m6_8,   # 67 ** COMPLETED **
        config_m6_9,   # 68
        config_m6_10,  # 69
        config_m6_11,  # 70
        config_m6_12,  # 71

        ############# Added at end of night on 20240302 ############
        # Model 7 configs - QKVO model
        config_m7_1,   # 72 ** COMPLETED ** 39.02 (QKVO model - High beam (12) + low temp (0.2))
        config_m7_2,   # 73 ** COMPLETED ** 38.65 (QKVO model - Greedy with length penalty)
        config_m7_3,   # 74 ** COMPLETED ** 40.49 (QKVO model - Very low temp + repetition penalty)
        config_m7_4,   # 75
        config_m7_5,   # 76
        
        # Model 8 configs - QKV model
        config_m8_1,   # 77 ** COMPLETED ** 34.76 (QKV model - High beam + low temp)
        config_m8_2,   # 78 ** COMPLETED **
        config_m8_3,   # 79 ** COMPLETED ** 35.37 (QKV model - Early stopping + optimal params)
        config_m8_4,   # 80
        config_m8_5,   # 81 ** COMPLETED **  35.98 (QKV model - Diversity penalty)
        
        # Model 9 configs - R32 model
        config_m9_1,   # 82 ** COMPLETED **
        config_m9_2,   # 83 ** COMPLETED **
        config_m9_3,   # 84 ** COMPLETED **
        config_m9_4,   # 85
        config_m9_5,   # 86
        
        # Model 10 configs - Extended model
        config_m10_1,  # 87 ** COMPLETED **
        config_m10_2,  # 88 ** COMPLETED ** 35.37 (Extended model - Greedy with length penalty)
        config_m10_3,  # 89 ** COMPLETED **
        config_m10_4,  # 90
        config_m10_5,  # 91

    ]

def deep_copy_config(config):
    """Create a deep copy of a nested dictionary configuration"""
    import copy
    return copy.deepcopy(config)