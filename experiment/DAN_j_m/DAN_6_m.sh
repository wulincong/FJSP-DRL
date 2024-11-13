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

exp="DAN_j_m"

n_j=6
n_m_options="30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5"
op_per_job=10

logdir=./runs/$exp
max_updates=500

# Train DAN models with different parameters

for n_m in $n_m_options; do
    python train/DAN.py --n_j $n_j --n_m $n_m --op_per_job $op_per_job --data_source SD2 --logdir $logdir/DAN/train_model/${n_j}x${n_m}x${op_per_job} --max_updates $max_updates 
done
