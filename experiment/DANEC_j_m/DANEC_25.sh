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

python train/DANEC.py --n_j 25 --n_m 5
python train/DANEC.py --n_j 25 --n_m 8
python train/DANEC.py --n_j 25 --n_m 11
python train/DANEC.py --n_j 25 --n_m 14
python train/DANEC.py --n_j 25 --n_m 17
python train/DANEC.py --n_j 25 --n_m 20
python train/DANEC.py --n_j 25 --n_m 23
python train/DANEC.py --n_j 25 --n_m 25