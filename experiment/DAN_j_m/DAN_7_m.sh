#!/bin/sh
#SBATCH -J torch
#SBATCH -p xhhgnormal   #修改队列名称，whichpartition查看队列名称
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
set -x
source ~/.bashrc
module load nvidia/cuda/11.6
conda activate RL-torch

n_j=5
n_m_options="5 8 11 14 17 20 23 25"

for n_m in $n_m_options; do
    python train/DANEC.py --n_j $n_j --n_m $n_m 
done
