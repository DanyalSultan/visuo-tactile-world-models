#!/bin/bash
#SBATCH --job-name=world_model_train
#SBATCH --output=training_logs_%j.txt
#SBATCH --error=training_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu        # Tells Stanage that I need a NVIDIA A100 node
#SBATCH --qos=gpu             # Required QoS flag for GPUs on Stanage
#SBATCH --gres=gpu:1          # Explicitly request 1 GPU
#SBATCH --mem=82G             # Request 82GB of CPU RAM (Stanage A100 standard)
#SBATCH --time=48:00:00       # 48-hour maximum run time

echo "--- Initializing Compute Node ---"
hostname
nvidia-smi

echo "--- Loading Modules ---"
# Stanage requires you to load Anaconda before you can use it
module load Anaconda3/2024.02-1

echo "--- Activating Environment ---"
# Source the conda setup script for bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate worldmodel

echo "--- Starting Training ---"
python train.py

echo "--- Job Complete ---"