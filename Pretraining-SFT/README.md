Generate README.md for Qwen3-14B Full-Parameter SFT Training Script (English version)
"""

readme_content = r"""# Qwen3-14B Full-Parameter SFT Training Script

This project provides a training script for **full-parameter supervised fine-tuning (SFT)** of the **Qwen3-14B** model. The script is built on Hugging Face `transformers` and `DeepSpeed ZeRO-3` to support distributed training, multi-GPU parallelism, gradient accumulation, periodic evaluation, and sample generation visualization during training (logged via SwanLab).

## Features

- Full-parameter fine-tuning (not LoRA/Adapter)
- Integrated DeepSpeed ZeRO-3 to optimize memory usage and support large-scale model training
- Automatic data preprocessing: splits raw JSON/JSONL data into training and validation sets (default 5% validation ratio)
- Periodic evaluation during training (evaluation steps)
- Custom callback (`PredictionCallback`): after each evaluation step, generates responses using samples from the validation set, prints them to the console, and uploads them to SwanLab
- Experiment tracking: logs hyperparameters, loss curves, generated samples, etc., via SwanLab
- Supports single-node multi-GPU / multi-node multi-GPU training (requires correct distributed environment setup)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- DeepSpeed 0.12+
- SwanLab (optional; can remove related code if not needed)
- Other dependencies: `pandas`, `datasets`, `accelerate`, etc.

We recommend creating a conda environment and installing the dependencies:

```bash
conda create -n sft python=3.10
conda activate sft
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Choose according to your CUDA version
pip install transformers datasets accelerate deepspeed swanlab pandas
