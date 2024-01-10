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
TEST_DIR=./test_script

exp=exp15
echo $exp

cat << EOL
不同的op_per_job+MAML
试验步骤：用常规办法DAN.py训练op_per_job_options中的每个，得到几个模型,作为baseline
常规方法得到的模型进行迁移学习 从第一个问题模型迁移到其他，以及从最后一个问题的模型迁移到其他问题。
MAML训练
MAML迁移
对比试验结果
EOL

# 本试验特殊参数
op_per_job_options="4 6 8 10 12"

# 定义通用参数
logdir="./runs/${exp}"
hidden_dim_actor=64
hidden_dim_critic=64
num_mlp_layers_actor=3
num_mlp_layers_critic=3
num_envs=4

# DAN.py参数
n_j=15
n_m=5
data_source="SD2"
logdir_dan="${logdir}/DAN"
options=($op_per_job_options)

# for op_per_job in "${options[@]}";do
op_per_job=4
python ./train/DAN.py   --n_j $n_j \
                --n_m $n_m \
                --data_source $data_source \
                --model_suffix SD2_operjob${op_per_job} \
                --logdir $logdir_dan/train_model/$op_per_job \
                --op_per_job $op_per_job \
                --max_updates 500
# done


# DAN_finetuning.py 脚本的特定参数
max_updates_finetune=50
lr=0.003
first_model_name="${n_j}x${n_m}+mix+${data_source}_operjob${options[0]}"
last_model_name="${n_j}x${n_m}+mix+${data_source}_operjob${options[-1]}"

for model in $first_model_name $last_model_name; do
    for op_per_job in "${options[@]}";do
        python "./train/DAN_finetuning.py" --logdir $logdir_dan/finetuning/${model}/op_per_job${op_per_job} \
                                    --model_suffix free \
                                    --finetuning_model $model \
                                    --max_updates $max_updates_finetune \
                                    --n_j $n_j \
                                    --n_m $n_m \
                                    --num_envs $num_envs \
                                    --op_per_job $op_per_job \
                                    --lr $lr
    done
done


# # multi_task_maml_exp**.py 脚本的特定参数
# meta_iterations=1000
# max_updates_maml=1000
# model_suffix=${exp}_${meta_iterations}_${hidden_dim_actor}_${num_mlp_layers_actor}
# num_tasks=4
# logdir_maml="${logdir}/maml"

# python ${t}/multi_task_maml_${exp}.py --logdir $logdir_maml \
#                                         --model_suffix $model_suffix \
#                                         --maml_model True \
#                                         --meta_iterations $meta_iterations \
#                                         --num_envs $num_envs \
#                                         --hidden_dim_actor $hidden_dim_actor \
#                                         --hidden_dim_critic $hidden_dim_critic \
#                                         --num_mlp_layers_actor $num_mlp_layers_actor \
#                                         --num_mlp_layers_critic $num_mlp_layers_critic \
#                                         --op_per_job_options $op_per_job_options 


# logdir_maml_finetuning=$logdir/maml_finetuning
# model=maml+$model_suffix

# for op_per_job in "${options[@]}";do
#     python "${t}/DAN_finetuning.py" --logdir $logdir_maml_finetuning/transfer${model}/operjob$op_per_job\
#                                     --model_suffix exp13_${model}_${n_j}x${n_m}\
#                                     --finetuning_model $model \
#                                     --max_updates $max_updates_finetune \
#                                     --num_envs $num_envs \
#                                     --hidden_dim_actor $hidden_dim_actor \
#                                     --hidden_dim_critic $hidden_dim_critic \
#                                     --num_mlp_layers_actor $num_mlp_layers_actor \
#                                     --num_mlp_layers_critic $num_mlp_layers_critic \
#                                     --lr $lr

# done

