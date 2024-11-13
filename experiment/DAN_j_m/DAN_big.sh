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

python train/DAN.py --n_j 30 --n_m 10  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/30x10 --max_updates 500
python train/DAN.py --n_j 30 --n_m 15  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/30x15 --max_updates 500
python train/DAN.py --n_j 30 --n_m 20  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/30x20 --max_updates 500
python train/DAN.py --n_j 40 --n_m 10  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/40x10 --max_updates 500
python train/DAN.py --n_j 40 --n_m 15  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/40x15 --max_updates 500
python train/DAN.py --n_j 40 --n_m 20  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/40x20 --max_updates 500