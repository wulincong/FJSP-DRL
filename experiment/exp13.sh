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

exp=exp13

echo $exp


# 定义通用参数
logdir_maml="./runs/${exp}_maml"
logdir="./runs/$exp"
hidden_dim_actor=64
hidden_dim_critic=64
num_mlp_layers_actor=3
num_mlp_layers_critic=3
num_envs=4
# multi_task_maml_exp11.py 脚本的特定参数
meta_iterations=1000
max_updates_maml=1000
model_suffix=${exp}_${meta_iterations}_${hidden_dim_actor}_${num_mlp_layers_actor}
n_j_options="15 15 15 15"
n_m_options="5  7  9  10"
num_tasks=4
# DAN_finetuning.py 脚本的特定参数
max_updates_finetune=100
lr=0.003

python ${t}/multi_task_maml_exp13.py --logdir ./runs/exp13_maml \
                                        --model_suffix $model_suffix \
                                        --maml_model True \
                                        --meta_iterations $meta_iterations \
                                        --num_tasks ${num_tasks} \
                                        --max_updates 1000 \
                                        --num_envs $num_envs \
                                        --hidden_dim_actor $hidden_dim_actor \
                                        --hidden_dim_critic $hidden_dim_critic \
                                        --num_mlp_layers_actor $num_mlp_layers_actor \
                                        --num_mlp_layers_critic $num_mlp_layers_critic \
                                        --n_j_options $n_j_options \
                                        --n_m_options $n_m_options



# echo 试验13测试

# data="10,5 15,10 20,5 20,10"

model=maml+$model_suffix

echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
    python "${t}/DAN_finetuning_freeze.py" --logdir ./runs/transfer_${model}_${n_j}x${n_m}_${lr}_freeze \
                                    --model_suffix exp13_${model}_${n_j}x${n_m}\
                                    --finetuning_model $model \
                                    --max_updates $max_updates_finetune \
                                    --n_j $n_j \
                                    --n_m $n_m \
                                    --num_envs $num_envs \
                                    --hidden_dim_actor $hidden_dim_actor \
                                    --hidden_dim_critic $hidden_dim_critic \
                                    --num_mlp_layers_actor $num_mlp_layers_actor \
                                    --num_mlp_layers_critic $num_mlp_layers_critic \
                                    --lr $lr
done
