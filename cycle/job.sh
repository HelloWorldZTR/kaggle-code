#!/bin/bash

#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=cyop
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=20
#SBATCH --time 12:00:00


python physics_engine.py