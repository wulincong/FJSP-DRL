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
python ./train/multi_task_maml_exp18.py --logdir ./runs/exp18/exp18-10/maml/train_model --model_suffix exp18-10 --maml_model True --meta_iterations 210 --num_tasks 4 --hidden_dim_actor 512 --hidden_dim_critic 512 --n_j_options 10 25 20 15 --n_m_options 5 5 10 15 --op_per_job_options 5 5 10 15
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-10/DAN/finetuning/maml+exp18-10/10x5 --model_suffix free --finetuning_model maml+exp18-10 --max_updates 11 --n_j 10 --n_m 5 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-10/DAN/finetuning/maml+exp18-10/10x10 --model_suffix free --finetuning_model maml+exp18-10 --max_updates 11 --n_j 10 --n_m 10 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-10/DAN/finetuning/maml+exp18-10/10x15 --model_suffix free --finetuning_model maml+exp18-10 --max_updates 11 --n_j 10 --n_m 15 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-10/DAN/finetuning/maml+exp18-10/25x5 --model_suffix free --finetuning_model maml+exp18-10 --max_updates 11 --n_j 25 --n_m 5 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-10/DAN/finetuning/maml+exp18-10/25x10 --model_suffix free --finetuning_model maml+exp18-10 --max_updates 11 --n_j 25 --n_m 10 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-10/DAN/finetuning/maml+exp18-10/25x15 --model_suffix free --finetuning_model maml+exp18-10 --max_updates 11 --n_j 25 --n_m 15 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-10/DAN/finetuning/maml+exp18-10/20x5 --model_suffix free --finetuning_model maml+exp18-10 --max_updates 11 --n_j 20 --n_m 5 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-10/DAN/finetuning/maml+exp18-10/20x10 --model_suffix free --finetuning_model maml+exp18-10 --max_updates 11 --n_j 20 --n_m 10 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-10/DAN/finetuning/maml+exp18-10/20x15 --model_suffix free --finetuning_model maml+exp18-10 --max_updates 11 --n_j 20 --n_m 15 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-10/DAN/finetuning/maml+exp18-10/15x5 --model_suffix free --finetuning_model maml+exp18-10 --max_updates 11 --n_j 15 --n_m 5 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-10/DAN/finetuning/maml+exp18-10/15x10 --model_suffix free --finetuning_model maml+exp18-10 --max_updates 11 --n_j 15 --n_m 10 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-10/DAN/finetuning/maml+exp18-10/15x15 --model_suffix free --finetuning_model maml+exp18-10 --max_updates 11 --n_j 15 --n_m 15 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
