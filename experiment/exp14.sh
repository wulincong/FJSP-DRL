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

exp=exp14
echo $exp
cat << EOL
寻找最好的MAML训练
用固定的几个问题训练MAML
EOL

# 本试验特殊参数
n_j_options="15 15 15 15"
n_m_options="5  7  9  10"

# 定义通用参数
logdir=./runs/$exp
hidden_dim_actor=512
hidden_dim_critic=512
num_mlp_layers_actor=3
num_mlp_layers_critic=3
num_envs=4

# multi_task_maml_exp11.py 脚本的特定参数
meta_iterations=1000
max_updates_maml=1000

num_tasks=4
# DAN_finetuning.py 脚本的特定参数
max_updates_finetune=50
lr=0.003

# 需要迭代的数据
data="15,5 15,7 15,9 15,10"

logdir_dan=$logdir/DAN

# echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
#     python ${t}/DAN.py                 --n_j $n_j \
#                                     --n_m $n_m \
#                                     --data_source SD2 \
#                                     --model_suffix SD2 \
#                                     --logdir $logdir_dan/train_model/${n_j}X$n_m \
#                                     --max_updates 500
# done


for model in 15x5+mix+SD2 15x10+mix+SD2
do
    echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
        python "${t}/DAN_finetuning.py" --logdir $logdir_dan/finetuning/${model}/${n_j}x${n_m} \
                                        --model_suffix free \
                                        --finetuning_model $model \
                                        --max_updates $max_updates_finetune \
                                        --n_j $n_j \
                                        --n_m $n_m \
                                        --num_envs $num_envs \
                                        --lr 0.003
    done
done



model_suffix=${exp}_${meta_iterations}_${hidden_dim_actor}_${num_mlp_layers_actor}
logdir_maml=$logdir/maml

python ${t}/multi_task_maml_$exp.py --logdir $logdir_maml/train_model \
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
        python "${t}/DAN_finetuning.py" --logdir $logdir_maml/finetuning/${model}/${n_j}x${n_m} \
                                        --model_suffix free \
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

