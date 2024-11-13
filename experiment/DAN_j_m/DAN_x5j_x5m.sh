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

python train/DAN.py --n_j 30 --n_m 30  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/30x30 --max_updates 500
python train/DAN.py --n_j 30 --n_m 25  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/30x25 --max_updates 500
python train/DAN.py --n_j 30 --n_m 20  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/30x20 --max_updates 500
python train/DAN.py --n_j 30 --n_m 15  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/30x15 --max_updates 500
python train/DAN.py --n_j 30 --n_m 10  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/30x10 --max_updates 500
python train/DAN.py --n_j 30 --n_m 5  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/30x5 --max_updates 500
python train/DAN.py --n_j 25 --n_m 30  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/25x30 --max_updates 500
python train/DAN.py --n_j 25 --n_m 25  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/25x25 --max_updates 500
python train/DAN.py --n_j 25 --n_m 20  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/25x20 --max_updates 500
python train/DAN.py --n_j 25 --n_m 15  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/25x15 --max_updates 500
python train/DAN.py --n_j 25 --n_m 10  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/25x10 --max_updates 500
python train/DAN.py --n_j 25 --n_m 5  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/25x5 --max_updates 500
python train/DAN.py --n_j 20 --n_m 30  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/20x30 --max_updates 500
python train/DAN.py --n_j 20 --n_m 25  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/20x25 --max_updates 500
python train/DAN.py --n_j 20 --n_m 20  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/20x20 --max_updates 500
python train/DAN.py --n_j 20 --n_m 15  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/20x15 --max_updates 500
python train/DAN.py --n_j 20 --n_m 10  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/20x10 --max_updates 500
python train/DAN.py --n_j 20 --n_m 5  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/20x5 --max_updates 500
python train/DAN.py --n_j 15 --n_m 30  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/15x30 --max_updates 500
python train/DAN.py --n_j 15 --n_m 25  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/15x25 --max_updates 500
python train/DAN.py --n_j 15 --n_m 20  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/15x20 --max_updates 500
python train/DAN.py --n_j 15 --n_m 15  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/15x15 --max_updates 500
python train/DAN.py --n_j 15 --n_m 10  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/15x10 --max_updates 500
python train/DAN.py --n_j 15 --n_m 5  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/15x5 --max_updates 500
python train/DAN.py --n_j 10 --n_m 30  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/10x30 --max_updates 500
python train/DAN.py --n_j 10 --n_m 25  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/10x25 --max_updates 500
python train/DAN.py --n_j 10 --n_m 20  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/10x20 --max_updates 500
python train/DAN.py --n_j 10 --n_m 15  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/10x15 --max_updates 500
python train/DAN.py --n_j 10 --n_m 10  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/10x10 --max_updates 500
python train/DAN.py --n_j 10 --n_m 5  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/10x5 --max_updates 500
python train/DAN.py --n_j 5 --n_m 30  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/5x30 --max_updates 500
python train/DAN.py --n_j 5 --n_m 25  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/5x25 --max_updates 500
python train/DAN.py --n_j 5 --n_m 20  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/5x20 --max_updates 500
python train/DAN.py --n_j 5 --n_m 15  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/5x15 --max_updates 500
python train/DAN.py --n_j 5 --n_m 10  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/5x10 --max_updates 500
python train/DAN.py --n_j 5 --n_m 5  --data_source SD2 --logdir ./runs/DAN_j_m/DAN/train_model/5x5 --max_updates 500