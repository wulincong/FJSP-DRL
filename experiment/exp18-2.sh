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

exp=exp18
echo $exp
cat << EOL
机器数和工件数同时改变
机器数多->少 结合 op_per_job 少->多
增加训练的
EOL

# 本试验特殊参数
n_j_options="15 15 15 15"
n_m_options="13 10 7 5"
op_per_job_options="4 7 10 12"

# 定义通用参数
logdir=./runs/$exp
hidden_dim_actor=512
hidden_dim_critic=512
num_mlp_layers_actor=3
num_mlp_layers_critic=3
num_envs=10
num_tasks=4

# multi_task_maml_exp.py 脚本的特定参数
meta_iterations=201
max_updates_maml=500

# DAN_finetuning.py 脚本的特定参数
max_updates_finetune=21
lr=0.003

# 需要迭代的数据
data="13,4 10,7 7,10 5,12"

logdir_dan=$logdir/DAN


# python ${t}/DAN.py              --n_j 15 \
#                                 --n_m 13 \
#                                 --op_per_job 4 \
#                                 --data_source SD2 \
#                                 --model_suffix SD2_4 \
#                                 --logdir $logdir_dan/train_model/15x13x4 \
#                                 --max_updates 500

# python ${t}/DAN.py              --n_j 15 \
#                                 --n_m 5 \
#                                 --op_per_job 12 \
#                                 --data_source SD2 \
#                                 --model_suffix SD2_12 \
#                                 --logdir $logdir_dan/train_model/15x5x12 \
#                                 --max_updates 500

# for n_j in "${n_j_options[@]}";do
#     echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
#         python ${t}/DAN.py                 --n_j $n_j \
#                                         --n_m $n_m \
#                                         --data_source SD2 \
#                                         --model_suffix SD2 \
#                                         --logdir $logdir_dan/train_model/${n_j}X$n_m \
#                                         --max_updates 500
#     done
# done

# n_j=15
# for model in 15x13+mix+SD2_4 15x5+mix+SD2_12
# do
#     for n_m in $n_m_options; do
#     for op_per_job in $op_per_job_options; do
#         python "${t}/DAN_finetuning.py" --logdir $logdir_dan/finetuning/${model}/${n_j}x${n_m}_${op_per_job} \
#                                 --model_suffix free \
#                                 --finetuning_model $model \
#                                 --max_updates $max_updates_finetune \
#                                 --n_j $n_j \
#                                 --n_m $n_m \
#                                 --op_per_job $op_per_job
#                                 --num_envs $num_envs \
#                                 --lr 0.003
#     done
#     done
# done



model_suffix=${exp}_${meta_iterations}_${hidden_dim_actor}_${num_mlp_layers_actor}
logdir_maml=$logdir/maml

python ${t}/multi_task_maml_exp18.py --logdir $logdir_maml/train_model/model_suffix \
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
                                     --n_m_options $n_m_options \
                                     --op_per_job_options $op_per_job_options

# 执行 DAN_finetuning.py

n_j=15
for model in maml+$model_suffix
do
    for n_m in $n_m_options; do
    for op_per_job in $op_per_job_options; do
        python "${t}/DAN_finetuning.py" --logdir $logdir_maml/finetuning/${model}/${n_j}x${n_m}_${op_per_job} \
                                        --model_suffix free \
                                        --finetuning_model $model \
                                        --max_updates $max_updates_finetune \
                                        --n_j $n_j \
                                        --n_m $n_m \
                                        --op_per_job $op_per_job \
                                        --num_envs $num_envs \
                                        --hidden_dim_actor $hidden_dim_actor \
                                        --hidden_dim_critic $hidden_dim_critic \
                                        --num_mlp_layers_actor $num_mlp_layers_actor \
                                        --num_mlp_layers_critic $num_mlp_layers_critic \
                                        --lr $lr
    done
    done
done

