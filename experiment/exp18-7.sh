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
subexp=exp18-7
echo $exp/$subexp
cat << EOL
exp18-7用新的元学习算法做之前的实验 改变seed
EOL

# 本试验特殊参数
n_j_options="5  7  10 13 15 17 20 23 25"
n_m_options="10 10 10 10 10 10 10 10 10"
op_per_job_options="5 5 5 5 5 5 5 5 5"

# 定义通用参数
logdir=./runs/$exp/$subexp
hidden_dim_actor=512
hidden_dim_critic=512
num_mlp_layers_actor=3
num_mlp_layers_critic=3
num_envs=10
num_tasks=9
seed_train=23333

# multi_task_maml_exp.py 脚本的特定参数
meta_iterations=207
max_updates_maml=500

# DAN_finetuning.py 脚本的特定参数
max_updates_finetune=11
lr=0.003
exp_dim=j

# 需要迭代的数据

logdir_dan=$logdir/DAN

# python ${t}/DAN.py              --n_j 30 \
#                                 --n_m 10 \
#                                 --op_per_job 5 \
#                                 --data_source SD2 \
#                                 --model_suffix $subexp \
#                                 --logdir $logdir_dan/train_model/30x10x5 \
#                                 --max_updates 500 \
#                                 --exp_dim $exp_dim

# python ${t}/DAN.py              --n_j 5 \
#                                 --n_m 10 \
#                                 --op_per_job 5 \
#                                 --data_source SD2 \
#                                 --model_suffix $subexp \
#                                 --logdir $logdir_dan/train_model/5x10x5 \
#                                 --max_updates 500 \
#                                 --exp_dim $exp_dim


n_m=10
op_per_job=5


for model in 30x10x5+mix 5x10x5+mix
do
    for n_j in $n_j_options; do
    # for op_per_job in $op_per_job_options; do
        python "${t}/DAN_finetuning.py" --logdir $logdir_dan/finetuning/${model}/${n_j}x${n_m} \
                                --model_suffix free \
                                --finetuning_model $model \
                                --max_updates $max_updates_finetune \
                                --n_j $n_j \
                                --n_m $n_m \
                                --op_per_job $op_per_job \
                                --num_envs $num_envs \
                                --lr $lr \
                                --exp_dim $exp_dim \
                                --seed_train $seed_train
    # done
    done
done



model_suffix=${subexp}_${meta_iterations}_${hidden_dim_actor}_${num_mlp_layers_actor}
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
                                     --op_per_job_options $op_per_job_options \
                                    --seed_train $seed_train
# 执行 DAN_finetuning.py


for model in maml+$model_suffix
do
    # for n_j in $n_j_options; do
    for op_per_job in $op_per_job_options; do
        python "${t}/DAN_finetuning.py" --logdir $logdir_maml/finetuning/${model}/${n_j}x${n_m}x${op_per_job} \
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
                                        --lr $lr \
                                        --exp_dim $exp_dim \
                                        --seed_train $seed_train
    # done
    done
done

