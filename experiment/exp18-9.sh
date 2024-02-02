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
python ./train/multi_task_maml_exp18.py --logdir ./runs/exp18/exp18-9/maml/train_model --model_suffix exp18-9 --maml_model True --meta_iterations 209 --num_tasks 6 --hidden_dim_actor 512 --hidden_dim_critic 512 --n_j_options 5 9 13 17 21 25 --n_m_options 25 21 17 13 9 5 --op_per_job_options 10 10 10 10 10 10
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/5x25 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 5 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/5x21 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 5 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/5x17 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 5 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/5x13 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 5 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/5x9 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 5 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/5x5 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 5 --n_m 5 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/9x25 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 9 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/9x21 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 9 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/9x17 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 9 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/9x13 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 9 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/9x9 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 9 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/9x5 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 9 --n_m 5 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/13x25 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 13 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/13x21 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 13 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/13x17 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 13 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/13x13 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 13 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/13x9 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 13 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/13x5 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 13 --n_m 5 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/17x25 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 17 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/17x21 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 17 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/17x17 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 17 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/17x13 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 17 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/17x9 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 17 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/17x5 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 17 --n_m 5 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/21x25 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 21 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/21x21 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 21 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/21x17 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 21 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/21x13 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 21 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/21x9 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 21 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/21x5 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 21 --n_m 5 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/25x25 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 25 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/25x21 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 25 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/25x17 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 25 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/25x13 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 25 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/25x9 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 25 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/30x30x10+mix/25x5 --model_suffix free --finetuning_model 30x30x10+mix --max_updates 11 --n_j 25 --n_m 5 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/5x25 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 5 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/5x21 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 5 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/5x17 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 5 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/5x13 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 5 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/5x9 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 5 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/5x5 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 5 --n_m 5 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/9x25 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 9 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/9x21 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 9 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/9x17 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 9 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/9x13 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 9 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/9x9 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 9 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/9x5 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 9 --n_m 5 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/13x25 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 13 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/13x21 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 13 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/13x17 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 13 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/13x13 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 13 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/13x9 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 13 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/13x5 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 13 --n_m 5 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/17x25 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 17 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/17x21 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 17 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/17x17 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 17 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/17x13 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 17 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/17x9 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 17 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/17x5 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 17 --n_m 5 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/21x25 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 21 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/21x21 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 21 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/21x17 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 21 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/21x13 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 21 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/21x9 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 21 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/21x5 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 21 --n_m 5 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/25x25 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 25 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/25x21 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 25 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/25x17 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 25 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/25x13 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 25 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/25x9 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 25 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/5x5x10+mix/25x5 --model_suffix free --finetuning_model 5x5x10+mix --max_updates 11 --n_j 25 --n_m 5 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/5x25 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 5 --n_m 25 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/5x21 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 5 --n_m 21 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/5x17 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 5 --n_m 17 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/5x13 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 5 --n_m 13 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/5x9 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 5 --n_m 9 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/5x5 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 5 --n_m 5 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/9x25 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 9 --n_m 25 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/9x21 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 9 --n_m 21 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/9x17 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 9 --n_m 17 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/9x13 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 9 --n_m 13 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/9x9 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 9 --n_m 9 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/9x5 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 9 --n_m 5 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/13x25 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 13 --n_m 25 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/13x21 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 13 --n_m 21 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/13x17 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 13 --n_m 17 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/13x13 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 13 --n_m 13 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/13x9 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 13 --n_m 9 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/13x5 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 13 --n_m 5 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/17x25 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 17 --n_m 25 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/17x21 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 17 --n_m 21 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/17x17 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 17 --n_m 17 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/17x13 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 17 --n_m 13 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/17x9 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 17 --n_m 9 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/17x5 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 17 --n_m 5 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/21x25 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 21 --n_m 25 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/21x21 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 21 --n_m 21 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/21x17 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 21 --n_m 17 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/21x13 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 21 --n_m 13 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/21x9 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 21 --n_m 9 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/21x5 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 21 --n_m 5 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/25x25 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 25 --n_m 25 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/25x21 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 25 --n_m 21 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/25x17 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 25 --n_m 17 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/25x13 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 25 --n_m 13 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/25x9 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 25 --n_m 9 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/maml+exp18-9/25x5 --model_suffix free --finetuning_model maml+exp18-9 --max_updates 11 --n_j 25 --n_m 5 --exp_dim jxm --hidden_dim_actor 512 --hidden_dim_critic 512
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
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/5x25 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 5 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/5x21 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 5 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/5x17 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 5 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/5x13 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 5 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/5x9 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 5 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/5x5 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 5 --n_m 5 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/9x25 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 9 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/9x21 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 9 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/9x17 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 9 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/9x13 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 9 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/9x9 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 9 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/9x5 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 9 --n_m 5 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/13x25 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 13 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/13x21 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 13 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/13x17 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 13 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/13x13 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 13 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/13x9 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 13 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/13x5 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 13 --n_m 5 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/17x25 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 17 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/17x21 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 17 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/17x17 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 17 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/17x13 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 17 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/17x9 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 17 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/17x5 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 17 --n_m 5 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/21x25 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 21 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/21x21 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 21 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/21x17 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 21 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/21x13 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 21 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/21x9 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 21 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/21x5 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 21 --n_m 5 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/25x25 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 25 --n_m 25 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/25x21 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 25 --n_m 21 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/25x17 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 25 --n_m 17 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/25x13 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 25 --n_m 13 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/25x9 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 25 --n_m 9 --exp_dim jxm 
python ./train/DAN_finetuning.py --logdir ./runs/exp18/exp18-9/DAN/finetuning/25x25x10+mix/25x5 --model_suffix free --finetuning_model 25x25x10+mix --max_updates 11 --n_j 25 --n_m 5 --exp_dim jxm 
