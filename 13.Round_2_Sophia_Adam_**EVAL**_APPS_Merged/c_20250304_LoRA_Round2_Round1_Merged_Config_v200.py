def get_evaluation_configurations(model_path=None):
    """
    Returns a list of configurations for LoRA model evaluation experiments
    
    Args:
        model_path (str, optional): Path to the model to evaluate.
            Defaults to the Round 2 merged GOLD model.
    
        Trial_00_Initial_Run_20250305_wrf_MBPP2APPS_adamw_r8_Ir0.0002_epochs3_merged_round2
        Trial_01_3_epochs_Baseline_Competition_20250306_kmm_MBPP2APPS_sophia_r8_Ir0.00016_epochs3
        Trial_01_3_epochs_Baseline_Competition_20250306_kmm_MBPP2APPS_sophia_r8_Ir0.00016_epochs3_merged_round2
        Trial_01_3_epochs_Baseline_Competition_20250306_skm_MBPP2APPS_adamw_r8_Ir0.0002_epochs3
        Trial_01_3_epochs_Baseline_Competition_20250306_skm_MBPP2APPS_adamw_r8_Ir0.0002_epochs3_merged_round2
        Trial_02_6_epochs_Increased_Epochs_Competition_20250307_mfp_MBPP2APPS_adamw_r8_Ir0.0002_epochs6
        Trial_02_6_epochs_Increased_Epochs_Competition_20250307_mfp_MBPP2APPS_adamw_r8_lr0.0002_epochs6_merged_round2
        Trial_02_6_epochs_Increased_Epochs_Competition_20250307_ytj_MBPP2APPS_sophia_r8_Ir0.00016_epochs6
        Trial_02_6_epochs_Increased_Epochs_Competition_20250307_ytj_MBPP2APPS_sophia_r8_Ir0.00016_epochs6_merged_round2
        Trial_03_5_epochs_Reduced_LR_Competition_20250307_arg_MBPP2APPS_adamw_r8_|r0.00015_epochs5
        Trial_03_5_epochs_Reduced_LR_Competition_20250307_arg_MBPP2APPS_adamw_r8_|r0.00015_epochs5_merged_round2
        Trial_03_5_epochs_Reduced_LR_Competition_20250307_sqw_MBPP2APPS_sophia_r8_Ir0.00012_epochs5
        Trial_03_5_epochs_Reduced_LR_Competition_20250307_sqw_MBPP2APPS_sophia_r8_|r0.00012_epochs5_merged_round2
        Trial_04_Batch_Accumulation_competition_20250307_fft_MBPP2APPS_sophia_r8_|r0.00014_epochs4
        Trial_04_Batch_Accumulation_competition_20250307_fft_MBPP2APPS_sophia_r8_|r0.00014_epochs4_merged_round2
        Trial_04_Batch_Accumulation_competition_20250307_yxz_MBPP2APPS_adamw_r8_Ir0.00018_epochs4
        Trial_04_Batch_Accumulation_competition_20250307_yxz_MBPP2APPS_adamw_r8_|r0.00018_epochs4_merged_round2    

    """
    # Use provided model path or default
    if model_path is None:
        # model_path = "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/08.Models/11.Round_2_Sophia_Adam_**TRAINING**_APPS/20250305_wrf_MBPP2APPS_adamw_r8_lr0.0002_epochs3_merged_round2"


        ################# Sophia Model ############### Round 2 ## Done 20250306 #########
        # model_path = r"/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/08.Models/11.Round_2_Sophia_Adam_**TRAINING**_APPS/Trial_01_3_epochs_Baseline_Competition_20250306_kmm_MBPP2APPS_sophia_r8_lr0.00016_epochs3_merged_round2"

        ################# Adam Model ############### Round 2 ###### Started 20250307 ##########
        model_path = r"/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/08.Models/11.Round_2_Sophia_Adam_**TRAINING**_APPS/Trial_01_3_epochs_Baseline_Competition_20250306_skm_MBPP2APPS_adamw_r8_lr0.0002_epochs3_merged_round2"
        
    # Base configuration 
    base_config = {
        "logging": {
            "log_dir": "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/13.Round_2_Sophia_Adam_**EVAL**_APPS/01.Logs",
        },
        "output": {
            "results_dir": "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/13.Round_2_Sophia_Adam_**EVAL**_APPS/results_all",
        },
        "tracking": {
            "file": "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/07.Tracking/13.Round_2_Sophia_Adam_**EVAL**_APPS_Tracking_Tracking.csv",
            "backup_dir": "/home/ubuntu/01.Stanford/efs/Stanford2024/02.CS224N_NLP_w_DeepLearning/20250313_FinalProject/07.Tracking/13.Round_2_Sophia_Adam_**EVAL**_APPS_Tracking_Backup",
        },
        "base_model": {
            "name": "codellama/CodeLlama-7b-hf",
            "dtype": "float16",
            "device_map": "auto",
            "max_position_embeddings": 256,
        },
        "lora_model": {
            "path": model_path,
        },
        ############ NOTE: The evaluation is the baseline condition -- revised in the configs #######
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
    def create_config_variant(description, **kwargs):
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

    # Different temperatures
    config_temp01 = create_config_variant("Lower temperature (0.1)", temperature=0.1)
    config_temp03 = create_config_variant("Higher temperature (0.3)", temperature=0.3)

    # Different beam settings
    config_beam1 = create_config_variant("Single beam", num_beams=1)
    config_beam5 = create_config_variant("More beams (5)", num_beams=5)

    # Top-p variations
    config_topp80 = create_config_variant("Lower top_p (0.8)", top_p=0.8)
    config_topp95 = create_config_variant("Higher top_p (0.95)", top_p=0.95)

    # Greedy sampling (no randomness)
    config_greedy = create_config_variant("Greedy decoding (no sampling)", 
                                         do_sample=False, temperature=1.0, top_p=1.0)

    # More tokens
    config_more_tokens = create_config_variant("More tokens (512)", max_new_tokens=512)

    # pass@10 config
    config_pass10 = create_config_variant("pass@10 configuration", 
                                         metric="pass@10", 
                                         num_samples_per_problem=10,
                                         temperature=0.8, 
                                         top_p=0.95, 
                                         num_beams=1)

    # Model 1 configurations
    config_m1_1 = create_config_variant("High beam (8) + high temp (0.4)", 
                                       num_beams=8, temperature=0.4, top_p=0.95)
    config_m1_2 = create_config_variant("Pure greedy + more beams (10)", 
                                       num_beams=10, temperature=1.0, do_sample=False, top_p=1.0)
    config_m1_3 = create_config_variant("Pure greedy + repetition penalty",
                                       num_beams=1, temperature=1.0, do_sample=False, 
                                       top_p=1.0, repetition_penalty=1.2)
    config_m1_4 = create_config_variant("Balanced temp (0.25) + beams (5)",
                                       num_beams=5, temperature=0.25, top_p=0.95)
    config_m1_5 = create_config_variant("Extended tokens (384) + good params",
                                       num_beams=5, temperature=0.3, top_p=0.95, max_new_tokens=384)
    config_m1_6 = create_config_variant("Top-K (50) + top beams + temp",
                                       num_beams=5, temperature=0.3, top_p=0.95, top_k=50)
    config_m1_7 = create_config_variant("Repetition penalty (1.3) + top params",
                                       num_beams=5, temperature=0.3, top_p=0.95, repetition_penalty=1.3)
    config_m1_8 = create_config_variant("Ultra high beam (15)",
                                       num_beams=15, temperature=0.2, top_p=0.9)
    config_m1_9 = create_config_variant("Standard pass@10 configuration",
                                       metric="pass@10", num_samples_per_problem=10,
                                       temperature=0.8, top_p=0.95, num_beams=1)
    config_m1_10 = create_config_variant("Minimum temperature (0.05) + good params",
                                        num_beams=5, temperature=0.05, top_p=0.95)
    
    # Model 2 configurations
    config_m2_1 = create_config_variant("High beam search (8) for 4-epoch model",
                                       num_beams=8, temperature=0.3, top_p=0.95)
    config_m2_2 = create_config_variant("Greedy with high beam (10) for 4-epoch model",
                                       num_beams=10, do_sample=False, temperature=1.0)
    config_m2_3 = create_config_variant("Medium temperature for 4-epoch model",
                                       num_beams=5, temperature=0.25, top_p=0.95)
    config_m2_4 = create_config_variant("Low temp + high beams for 4-epoch model",
                                       num_beams=8, temperature=0.1, top_p=0.9)
    config_m2_5 = create_config_variant("Very high beam search for 4-epoch model",
                                       num_beams=12, temperature=0.2, top_p=0.9)
    config_m2_6 = create_config_variant("High top-k for 4-epoch model",
                                       num_beams=5, temperature=0.3, top_p=0.95, top_k=30)
    config_m2_7 = create_config_variant("Very high top-p for 4-epoch model",
                                       num_beams=3, temperature=0.3, top_p=0.98)
    config_m2_8 = create_config_variant("Extended tokens for 4-epoch model",
                                       num_beams=5, temperature=0.2, max_new_tokens=512)
    config_m2_9 = create_config_variant("Pure greedy for 4-epoch model",
                                       num_beams=1, do_sample=False, temperature=1.0)
    config_m2_10 = create_config_variant("Pass@10 for 4-epoch model",
                                        metric="pass@10", num_samples_per_problem=10,
                                        temperature=0.8, top_p=0.95)
    
    # Model 3 configurations
    config_m3_1 = create_config_variant("High beam + medium temp for r16 model",
                                       num_beams=8, temperature=0.25, top_p=0.95)
    config_m3_2 = create_config_variant("Pure greedy with beams for r16 model",
                                       num_beams=8, do_sample=False, temperature=1.0)
    config_m3_3 = create_config_variant("Low temperature for r16 model",
                                       num_beams=5, temperature=0.1, top_p=0.9)
    config_m3_4 = create_config_variant("High temperature for r16 model",
                                       num_beams=3, temperature=0.4, top_p=0.95)
    config_m3_5 = create_config_variant("Aggressive beam search for r16 model",
                                       num_beams=15, temperature=0.2, top_p=0.9)
    config_m3_6 = create_config_variant("Balanced parameters for r16 model",
                                       num_beams=5, temperature=0.25, top_p=0.92)
    config_m3_7 = create_config_variant("Top-k filtering for r16 model",
                                       num_beams=5, temperature=0.3, top_p=0.95, top_k=40)
    config_m3_8 = create_config_variant("Extended generation for r16 model",
                                       num_beams=5, temperature=0.25, max_new_tokens=384)
    config_m3_9 = create_config_variant("Pure greedy for r16 model",
                                       num_beams=1, do_sample=False, temperature=1.0)
    config_m3_10 = create_config_variant("Pass@10 for r16 model",
                                        metric="pass@10", num_samples_per_problem=10,
                                        temperature=0.8, top_p=0.95)
    
    # Model 4 configurations
    config_m4_1 = create_config_variant("Balanced beam search for alpha32 model",
                                       num_beams=8, temperature=0.3, top_p=0.95)
    config_m4_2 = create_config_variant("Pure greedy + high beam for alpha32 model",
                                       num_beams=10, do_sample=False, temperature=1.0)
    config_m4_3 = create_config_variant("Lower temperature for alpha32 model",
                                       num_beams=5, temperature=0.15, top_p=0.9)
    config_m4_4 = create_config_variant("Higher temperature for alpha32 model",
                                       num_beams=3, temperature=0.35, top_p=0.95)
    config_m4_5 = create_config_variant("Aggressive beam search for alpha32 model",
                                       num_beams=12, temperature=0.2, top_p=0.9)
    config_m4_6 = create_config_variant("High top-p for alpha32 model",
                                       num_beams=5, temperature=0.25, top_p=0.98)
    config_m4_7 = create_config_variant("Top-k filtering for alpha32 model",
                                       num_beams=5, temperature=0.25, top_k=40, top_p=0.95)
    config_m4_8 = create_config_variant("Extended generation for alpha32 model",
                                       num_beams=5, temperature=0.25, max_new_tokens=384)
    config_m4_9 = create_config_variant("Simple greedy for alpha32 model",
                                       num_beams=1, do_sample=False, temperature=1.0)
    config_m4_10 = create_config_variant("Pass@10 for alpha32 model",
                                        metric="pass@10", num_samples_per_problem=10,
                                        temperature=0.8, top_p=0.95)
    
    # Model 5 configurations
    config_m5_1 = create_config_variant("High beam + medium temp for no-dropout model",
                                       num_beams=8, temperature=0.25, top_p=0.95)
    config_m5_2 = create_config_variant("Pure greedy with beams for no-dropout model",
                                       num_beams=8, do_sample=False, temperature=1.0)
    config_m5_3 = create_config_variant("Low temperature for no-dropout model",
                                       num_beams=5, temperature=0.1, top_p=0.9)
    config_m5_4 = create_config_variant("High temperature for no-dropout model",
                                       num_beams=3, temperature=0.4, top_p=0.95)
    config_m5_5 = create_config_variant("Aggressive beam search for no-dropout model",
                                       num_beams=15, temperature=0.2, top_p=0.9)
    config_m5_6 = create_config_variant("Balanced parameters for no-dropout model",
                                       num_beams=5, temperature=0.25, top_p=0.92)
    config_m5_7 = create_config_variant("Top-k filtering for no-dropout model",
                                       num_beams=5, temperature=0.3, top_p=0.95, top_k=40)
    config_m5_8 = create_config_variant("Extended generation for no-dropout model",
                                       num_beams=5, temperature=0.25, max_new_tokens=384)
    config_m5_9 = create_config_variant("Pure greedy for no-dropout model",
                                       num_beams=1, do_sample=False, temperature=1.0)
    config_m5_10 = create_config_variant("Pass@10 for no-dropout model",
                                        metric="pass@10", num_samples_per_problem=10,
                                        temperature=0.8, top_p=0.95)
    
    # Model 6 consolidated configs
    config_m6_1 = create_config_variant("Ultra-High Beam (12) with Optimal Temperature (0.25)",
                                       num_beams=12, temperature=0.25, top_p=0.95)
    config_m6_2 = create_config_variant("Hybrid Beam-Greedy with Length Penalty",
                                       num_beams=10, temperature=1.0, top_p=1.0, 
                                       do_sample=False, length_penalty=1.05)
    config_m6_3 = create_config_variant("R16 with Repetition Penalty (1.1)",
                                       num_beams=8, temperature=0.25, repetition_penalty=1.1)
    config_m6_4 = create_config_variant("No Dropout Model with High Beam (10)",
                                       num_beams=10, temperature=0.25)
    config_m6_5 = create_config_variant("Lower Temperature (0.2) for Precision",
                                       num_beams=8, temperature=0.2)
    config_m6_6 = create_config_variant("Precision Focus with Top-K (40)",
                                       num_beams=8, temperature=0.2, top_p=0.92, top_k=40)
    config_m6_7 = create_config_variant("Extended Generation (384 tokens)",
                                       num_beams=8, max_new_tokens=384)
    config_m6_8 = create_config_variant("Alpha-32 Model with Optimal Parameters",
                                       num_beams=8, temperature=0.25)
    config_m6_9 = create_config_variant("Very Focused Beam Search (15 beams, 0.15 temp)",
                                       num_beams=15, temperature=0.15, length_penalty=1.0)
    config_m6_10 = create_config_variant("Mixed Strategy (9 beams, 0.22 temp, no repeat ngram)",
                                        num_beams=9, temperature=0.22, top_p=0.93, 
                                        no_repeat_ngram_size=3)
    config_m6_11 = create_config_variant("Pass@10 with higher diversity (temp 0.7, top_p 0.98)",
                                        metric="pass@10", num_samples_per_problem=10,
                                        temperature=0.7, top_p=0.98, num_beams=3)
    config_m6_12 = create_config_variant("Pass@10 with num_beams (8)",
                                        metric="pass@10", num_samples_per_problem=10,
                                        temperature=0.6, num_beams=8)
    
    # Model 7 - QKVO model configs
    config_m7_1 = create_config_variant("QKVO model - High beam (12) + low temp (0.2)",
                                       num_beams=12, temperature=0.2, top_p=0.92)
    config_m7_2 = create_config_variant("QKVO model - Greedy with length penalty",
                                       num_beams=10, do_sample=False, temperature=1.0, 
                                       length_penalty=1.05)
    config_m7_3 = create_config_variant("QKVO model - Very low temp + repetition penalty",
                                       num_beams=8, temperature=0.15, top_p=0.9, 
                                       repetition_penalty=1.1)
    config_m7_4 = create_config_variant("QKVO model - Ultra high beam + extended tokens",
                                       num_beams=15, temperature=0.25, max_new_tokens=384)
    config_m7_5 = create_config_variant("QKVO model - Top-k + no_repeat_ngram",
                                       num_beams=10, temperature=0.2, top_k=40,
                                       no_repeat_ngram_size=3)
    
    # Model 8 - QKV model configs
    config_m8_1 = create_config_variant("QKV model - High beam + low temp",
                                       num_beams=10, temperature=0.2, top_p=0.95)
    config_m8_2 = create_config_variant("QKV model - Pure greedy with high beam",
                                       num_beams=12, do_sample=False, temperature=1.0)
    config_m8_3 = create_config_variant("QKV model - Early stopping + optimal params",
                                       num_beams=8, temperature=0.18, top_p=0.92, 
                                       early_stopping=True)
    config_m8_4 = create_config_variant("QKV model - Ultra high beam + more tokens",
                                       num_beams=15, temperature=0.22, top_p=0.9,
                                       max_new_tokens=384)
    config_m8_5 = create_config_variant("QKV model - Diversity penalty",
                                       num_beams=9, temperature=0.15, diversity_penalty=0.2)
    
    # Model 9 - R32 model configs
    config_m9_1 = create_config_variant("R32 model - Balanced beam + low temp",
                                       num_beams=10, temperature=0.2, top_p=0.9)
    config_m9_2 = create_config_variant("R32 model - Pure greedy",
                                       num_beams=8, do_sample=False, temperature=1.0)
    config_m9_3 = create_config_variant("R32 model - High beam + repetition penalty",
                                       num_beams=12, temperature=0.15, top_p=0.95,
                                       repetition_penalty=1.15)
    config_m9_4 = create_config_variant("R32 model - Ultra high beam + extended tokens",
                                       num_beams=15, temperature=0.25, max_new_tokens=384)
    config_m9_5 = create_config_variant("R32 model - Top-k + no_repeat_ngram",
                                       num_beams=10, temperature=0.18, top_k=30,
                                       no_repeat_ngram_size=3)
    
    # Model 10 - Extended model configs
    config_m10_1 = create_config_variant("Extended model - Balanced params",
                                        num_beams=10, temperature=0.22, top_p=0.92)
    config_m10_2 = create_config_variant("Extended model - Greedy with length penalty",
                                        num_beams=12, do_sample=False, temperature=1.0,
                                        length_penalty=1.05)
    config_m10_3 = create_config_variant("Extended model - Very low temp + repetition",
                                        num_beams=8, temperature=0.15, top_p=0.9,
                                        repetition_penalty=1.2)
    config_m10_4 = create_config_variant("Extended model - Ultra high beam + more tokens",
                                        num_beams=15, temperature=0.2, max_new_tokens=384)
    config_m10_5 = create_config_variant("Extended model - Top-k + diversity",
                                        num_beams=10, temperature=0.25, top_k=50,
                                        diversity_penalty=0.2)

    # Return all configurations with the original order and numbering
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
        config_m3_1,   # 30 ** RETESTED **  34.76 (High beam + medium temp for r16 model) --> NOTE: (**unmerged**) These are results carried over from Round 1 --> got better results 35.98
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
        
        ############# 20250306 --> round 2 after sophia / adam --> new round of evals ####### Ran for 72-76 #########
        config_m7_1,   # 72 ** RETESTED ** 39.02 (QKVO model - High beam (12) + low temp (0.2)) --> NOTE: (**unmerged**)These are results carried over from Round 1 --> Got Lower Result --> 34.76
        config_m7_2,   # 73 ** COMPLETED ** 38.65 (QKVO model - Greedy with length penalty) --> NOTE: (**unmerged**) These are results carried over from Round 1 --> Got Lower Result --> 36.59
        config_m7_3,   # 74 ** COMPLETED ** 40.49 (QKVO model - Very low temp + repetition penalty) --> NOTE: (**unmerged**) These are results carried over from Round 1 --> Got Lower Result --> 33.54
        config_m7_4,   # 75
        config_m7_5,   # 76

        ############# 20250306 --> round 2 after sophia / adam --> new round of evals ####### Ran 77 - 81 #########
        # Model 8 configs - QKV model
        config_m8_1,   # 77 ** COMPLETED ** 34.76 (QKV model - High beam + low temp) --> NOTE: These are results carried over from Round 1
        config_m8_2,   # 78 ** COMPLETED **
        config_m8_3,   # 79 ** COMPLETED ** 35.37 (QKV model - Early stopping + optimal params) --> NOTE: These are results carried over from Round 1
        config_m8_4,   # 80
        config_m8_5,   # 81 ** COMPLETED **  35.98 (QKV model - Diversity penalty) --> NOTE: These are results carried over from Round 1

        ############# 20250306 --> round 2 after sophia / adam --> new round of evals ####### Ran for 82-86 #########
        # Model 9 configs - R32 model
        config_m9_1,   # 82 ** COMPLETED **
        config_m9_2,   # 83 ** COMPLETED **
        config_m9_3,   # 84 ** COMPLETED **
        config_m9_4,   # 85
        config_m9_5,   # 86

        ############# 20250306 --> round 2 after sophia / adam --> new round of evals ####### TBD #########
        # Model 10 configs - Extended model
        config_m10_1,  # 87 ** COMPLETED **
        config_m10_2,  # 88 ** COMPLETED ** 35.37 (Extended model - Greedy with length penalty) --> NOTE: These are results carried over from Round 1
        config_m10_3,  # 89 ** COMPLETED **
        config_m10_4,  # 90
        config_m10_5,  # 91
    ]

def deep_copy_config(config):
    """Create a deep copy of a nested dictionary configuration"""
    import copy
    return copy.deepcopy(config)