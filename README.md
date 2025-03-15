# FLeX: Fourier-based Low-rank EXpansion for Multilingual Transfer

This repository contains the implementation and experimental results for FLeX, a novel approach for enhancing cross-lingual code generation through frequency-domain regularization with Low-Rank Adaptation (LoRA).

## Overview

Cross-lingual code generation is critical in enterprise environments where multiple programming languages coexist. This project, evolved from exploring efficient adaptation techniques, investigates whether parameter-efficient fine-tuning methods and optimizer enhancements can improve cross-lingual transfer from Python to languages like Java.

### Key Results

- **40.1% pass@1** on Python HumanEval using MBPP LoRA fine-tuning (surpassing specialized Code Llama-Python)
- **42.1% pass@1** on Java MultiPL-E using Fourier regularization (exceeding baseline by ~8%)
- **30% faster convergence** with Sophia optimizer compared to AdamW

## Repository Structure

The repository maintains the original folder structure from the project:

- **Round 1**: LoRA fine-tuning on MBPP dataset (unmerged)
  - `04.Round_1_LoRA_MBPP_Model_TRAINING_Unmerged/`
  - `06.Round_1_LoRA_MBPP_EVAL_pass@1_Unmerged/`

- **Round 2**: Optimizer comparison between Adam and Sophia
  - `11.Round_2_Sophia_Adam_TRAINING_APPS_Merged/`
  - `13.Round_2_Sophia_Adam_EVAL_APPS_Merged/`

- **Round 3**: Cross-lingual transfer baseline evaluations
  - `21.Round_3_CrossLingual_TRAINING_CodeSearchNet_UNMERGED_multiPL-E/`
  - `22.Round_3_CrossLingual_TRAINING_MBPP_Merged_multiPL-E/`
  - `23.Round_3_CrossLingual_EVAL_Merged_MBPP_APPS_CodeSearchNet_MultiPL-E/`

- **Round 4**: Fourier-based regularization with merged LoRA
  - `27.Round_4_CrossLingual_TRAINING_FOURIER_MBPP_Merged/`
  - `28.Round_4_CrossLingual_EVAL_Fourier_FirstRun_MBPP_Merged/`
  - `29.Round_4_CrossLingual_EVAL_Fourier_SecondRun_MBPP_Merged/`

- **Round 5**: Fourier-based regularization with unmerged LoRA
  - `31.Round_5_CrossLingual_TRAINING_Fourier_MBPP_UNMERGED/`
  - `32.Round_5_CrossLingual_EVAL_Fourier_MBPP_UNMERGED/`

- **Tracking**: `07.Tracking/`

## Method

FLeX introduces a novel Fourier-based regularization technique that applies frequency domain analysis to LoRA parameter updates:

1. **Low-Rank Adaptation (LoRA)**: We fine-tune Code Llama-7B using LoRA, focusing on a small subset of parameters to efficiently adapt the model.

2. **Fourier Transform Regularization**: We decompose parameter updates into frequency components and apply regularization to preserve low-frequency (generalizable) components while penalizing high-frequency (language-specific) ones:

LFourier(w) = Σ ρ(k, n, T) · |F(w)k|²

3. **Unmerged vs. Merged LoRA**: We demonstrate that keeping LoRA weights unmerged with the base model significantly improves cross-lingual performance.

## Results

| Model Variant | Python HumanEval | Java MultiPL-E |
|---------------|------------------|----------------|
| Code Llama-7B (base) | 34.2% | 33.3% |
| Code Llama-Python-7B | 38.4% | 35.4% |
| LoRA MBPP (unmerged) | 40.1% | 31.5% |
| FLeX (merged) | 36.6% | 32.9% |
| FLeX (unmerged) | 39.8% | 42.1% |

## Paper

See the attached paper for full details on the methodology and results.

## Citation

```bibtex
@article{narasimhan2025flex,
title={FLeX: Fourier-based Low-rank EXpansion for multilingual transfer},
author={Narasimhan, Gaurav},
journal={Stanford CS224N Custom Project},
year={2025}
}
