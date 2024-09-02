#!/bin/bash

#SBATCH --job-name=acas-xu-DQN
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time 1-0:00
#SBATCH --qos=co_long_gpu
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err
#SBATCH --constraint rtx6000 

source /stck/wabouir/.bashrc 

python /stck/wabouir/acas-xu/acas_xu_train.py
