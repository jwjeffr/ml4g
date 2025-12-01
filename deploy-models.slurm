#!/bin/bash

#SBATCH --job-name models
#SBATCH --nodes 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --mem 64gb
#SBATCH --gpus h100:1
#SBATCH --time 72:00:00

srun venv/bin/python deploy-models.py
