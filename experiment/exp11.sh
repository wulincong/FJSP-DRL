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


t=train
ADAPT_NUMS=5
TEST_DIR=./test_script

exp=exp11
echo $exp
echo 寻找最好的MAML训练

# 定义通用参数
logdir_maml="./runs/exp11_maml"
logdir="./runs/exp11"
hidden_dim_actor=64
hidden_dim_critic=64
num_mlp_layers_actor=3
num_mlp_layers_critic=3
num_envs=5
# multi_task_maml_exp11.py 脚本的特定参数
meta_iterations=1000
max_updates_maml=1000
model_suffix=${exp}_${meta_iterations}_${hidden_dim_actor}_${num_mlp_layers_actor}
n_j_options="15 15 15"
n_m_options="5  7  9  10"
num_tasks=6
# DAN_finetuning.py 脚本的特定参数
max_updates_finetune=100
lr=0.003

# 需要迭代的数据
data="15,5 15,7 15,9 15,10"

echo exp2 无maml训练15x5  SD2

# python ${t}/DAN.py                 --n_j 15 \
#                                 --n_m 5 \
#                                 --data_source SD2 \
#                                 --model_suffix SD2 \
#                                 --logdir ./runs/exp2/train/15x5_SD2 \
#                                 --max_updates 500




# 执行 multi_task_maml_exp11.py
python ${t}/multi_task_maml_exp11.py --logdir $logdir_maml \
                                     --model_suffix $model_suffix \
                                     --maml_model True \
                                     --meta_iterations $meta_iterations \
                                     --num_tasks ${num_tasks} \
                                     --max_updates $max_updates_maml \
                                     --num_envs $num_envs \
                                     --hidden_dim_actor $hidden_dim_actor \
                                     --hidden_dim_critic $hidden_dim_critic \
                                     --num_mlp_layers_actor $num_mlp_layers_actor \
                                     --num_mlp_layers_critic $num_mlp_layers_critic \
                                     --n_j_options $n_j_options \
                                     --n_m_options $n_m_options

# 执行 DAN_finetuning.py
for model in maml+$model_suffix
do
    echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
        python "${t}/DAN_finetuning.py" --logdir $logdir/transfer_${model}_${n_j}x${n_m} \
                                        --model_suffix exp11_${model}_${n_j}x${n_m} \
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
done

# for model in 15x5+mix+SD2
# do
#     echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
#         python "${t}/DAN_finetuning.py" --logdir ./runs/exp2/finetuning/transfer_${model}_${n_j}x${n_m} \
#                                         --model_suffix exp2_${model}_${n_j}x${n_m} \
#                                         --finetuning_model $model \
#                                         --max_updates 100 \
#                                         --n_j $n_j \
#                                         --n_m $n_m \
#                                         --num_envs 5 \
#                                         --lr 0.003
#     done
# done

