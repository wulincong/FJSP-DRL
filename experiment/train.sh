#!/bin/sh
#SBATCH -J torch
#SBATCH -p xhhgnormal   #修改队列名称，whichpartition查看队列名称
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1

source ~/.bashrc
module load nvidia/cuda/11.6
conda activate RL-torch


t=train
meta_iterations=3000
num_tasks=5
max_updates=$((meta_iterations * num_tasks))
ADAPT_NUMS=5
TEST_DIR=./test_script


echo exp1 无maml训练

# data="10,5 15,10 20,5 20,10"
# data="20,15"

# echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do

#     python ${t}/DAN.py            --n_j $n_j \
#                                     --n_m $n_m \
#                                     --data_source SD2 \
#                                     --model_suffix SD2 \
#                                     --logdir ./runs/exp1/${n_j}x${n_m}_SD2 \
#                                     --max_updates 1000 
# done

# echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do

#     python ${t}/DAN.py            --n_j $n_j \
#                                     --n_m $n_m \
#                                     --data_source SD1 \
#                                     --model_suffix SD1 \
#                                     --logdir ./runs/exp1/${n_j}x${n_m}_SD1 \
#                                     --max_updates 500 
# done

# python ${TEST_DIR}/test_trained_model.py 	--data_source SD1	\
# 				                --model_source SD1	\
#     				            --test_data 10x5    \
#         			            --test_model 10x5	\
#             			        --test_mode False	\
#                 		        --sample_times 100	\
#                             >> train_log/exp1_1_10x5_SD1.log

# # 实验1-2： 无maml训练10x5  SD2
# echo exp2 无maml训练10x5  SD2
# python ${t}/DAN.py                 --n_j 10 \
#                                 --n_m 5 \
#                                 --data_source SD2 \
#                                 --model_suffix SD2 \
#                                 --logdir ./runs/exp1_2_10x5_SD2 \
#                                 --max_updates 500
# #                             > train_log/exp1_2_10x5_SD2.log

# python ${TEST_DIR}/test_trained_model.py 	--data_source SD2	\
# 				                --model_source SD2	\
#     				            --test_data 10x5+mix    \
#         			            --test_model 10x5+mix	\
#             			        --test_mode False	\
#                 		        --sample_times 100	\
#                             >> train_log/exp1_2_10x5_SD2.log

# 实验2： maml训练10x5 SD1
# python ${t}/maml.py            --n_j 10 \
#                                 --n_m 5 \
#                                 --data_source SD1 \
#                                 --model_suffix maml_sd1 \
#                                 --logdir ./runs/exp2_maml_10x5_SD1 \
#                                 --meta_iterations ${meta_iterations} \
#                                 --num_tasks ${num_tasks} \
#                             > train_log/exp2_maml_10x5_SD1.log

# # python ${TEST_DIR}/test_trained_model.py 	--data_source SD1	\
# # 				                --model_source SD1	\
# #     				            --test_data 10x5    \
# #         			            --test_model 10x5+maml_sd1	\
# #             			        --test_mode False	\
# #                 		        --sample_times 100	\
# #                             >> train_log/exp2_maml_10x5_SD1.log

# 实验3： 对比maml训练相同env 10x5 SD2
# echo exp3
# python ${t}/same_env.py    --data_source SD2 \
#                             --model_suffix maml_sd2_same \
#                             --logdir ./runs/exp3_same_env_10x5_SD2 \
#                             --meta_iterations ${meta_iterations} \
#                             --num_tasks ${num_tasks} \
#                             --adapt_nums ${ADAPT_NUMS} \
#                         > train_log/exp3_same_env_10x5_SD2.log

# echo 实验4： maml模型泛化性测试 10x5 扩展到11x5 SD2
# python ./test_script/trained_model.py --data_source=10x6 \
#                                         --test_data 10x6+mix 11x5+mix 11x6+mix \
#                                         --model_source=SD2 \
#                                         --test_model 10x5+mix+maml_sd2_same 10x5+mix+maml_sd2_same_

# echo 试验5： 在不同规模的FJSP任务上进行MAML学习\(8~12\)x\(4~6\)
# python ${t}/multi_task_maml.py  --logdir ./runs/exp5_multi_task_maml \
#                                 --meta_iterations ${meta_iterations} \
#                                 --num_tasks ${num_tasks} \
#                                 --reset_env_timestep 50

# echo 试验6： 在不同规模的FJSP任务上进行MAML学习,并计算收敛的步数\(8~12\)x\(4~6\)
# python ${t}/multi_task_maml_exp6.py  --logdir ./runs/exp6_multi_task_maml \
#                                 --meta_iterations ${meta_iterations} \
#                                 --model_suffix exp6 \
#                                 --num_tasks ${num_tasks} \
#                                 > train_log/exp6.log

# echo 实验7： 无MAML迁移学习
# python ${t}/multi_task_transfer_learn.py --logdir ./runs/exp7_multi_task_transfer_learn \
#                                         --model_suffix exp7 \
#                                         --meta_iterations ${meta_iterations} \
#                                         --num_tasks ${num_tasks} \
#                                         --max_updates ${max_updates} \
#                                         > train_log/exp7.log


# echo 试验8：epoch内MAML
# python ${t}/multi_task_maml_exp8.py --logdir ./runs/exp8_multi_task_maml \
#                                         --model_suffix exp8 \
#                                         --meta_iterations ${meta_iterations} \
#                                         --num_tasks ${num_tasks} \
#                                         --max_updates ${max_updates} 
#                                         # > train_log/exp8.log

# echo 试验9：epoch内MAML 使用新的meta_loss函数
# python ${t}/multi_task_maml_exp9.py --logdir ./runs/exp9_multi_task_maml \
#                                         --model_suffix exp9_1w \
#                                         --meta_iterations ${meta_iterations} \
#                                         --num_tasks ${num_tasks} \
#                                         --max_updates ${max_updates} \
#                                         --num_envs 5


echo 试验10：用原论文的模型模型进行finetuning


data="10,5 15,10 20,5 20,10"

# for model in 10x5+SD1 15x10+SD1 20x5+SD1 20x10+SD1 

# do
#     echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
#         python "${t}/DAN_finetuning.py" --logdir ./runs/exp10/transfer_${model}_${n_j}x${n_m} \
#                                         --model_suffix exp10_${model}_${n_j}x${n_m} \
#                                         --finetuning_model $model \
#                                         --max_updates 100 \
#                                         --n_j $n_j \
#                                         --n_m $n_m \
#                                         --num_envs 3 \
#                                         --data_source SD1 \
#                                         --model_source SD1

#     done
# done




# python ${t}/multi_task_maml_exp11.py --logdir ./runs/exp11_maml \
#                                         --model_suffix exp11_1k_opt \
#                                         --maml_model True \
#                                         --meta_iterations 1000 \
#                                         --num_tasks ${num_tasks} \
#                                         --max_updates 1000 \
#                                         --num_envs 5 \
#                                         --hidden_dim_actor 128 \
#                                         --hidden_dim_critic 128 \
#                                         --num_mlp_layers_actor 5 \
#                                         --num_mlp_layers_critic 5 \
#                                         --n_j_options 8 10 11 13 15 16 17 20 21 \
#                                         --n_m_options 4 5 8 10 12




exp=exp11
echo $exp

# 定义通用参数
logdir="./runs/exp11_maml"

hidden_dim_actor=256
hidden_dim_critic=256
num_mlp_layers_actor=3
num_mlp_layers_critic=3
num_envs=5
# multi_task_maml_exp11.py 脚本的特定参数
meta_iterations=2000
max_updates_maml=2000
model_suffix=${exp}_${meta_iterations}_${hidden_dim_actor}_${num_mlp_layers_actor}
n_j_options="20 20 20 20 20 20"
n_m_options="5  7  9  10 11 13"
num_tasks=6
# n_j_options="20 20 20"
# n_m_options="15 15 15"
# DAN_finetuning.py 脚本的特定参数
max_updates_finetune=100
lr=0.003

# 需要迭代的数据
data="20,5 20,7 20,9 20,10 20,11 20,13 20,15"

# 执行 multi_task_maml_exp11.py
python ${t}/multi_task_maml_exp11.py --logdir $logdir \
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
                                     --minibatch_size 500

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


# for model in 20x15+mix+SD2
# do
#     echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
#         python "${t}/DAN_finetuning.py" --logdir ./runs/exp11/transfer_${model}_${n_j}x${n_m} \
#                                         --model_suffix exp11_${model}_${n_j}x${n_m} \
#                                         --finetuning_model $model \
#                                         --max_updates 100 \
#                                         --n_j $n_j \
#                                         --n_m $n_m \
#                                         --num_envs 5 \
#                                         --lr 0.003
#     done
# done


# echo 试验12：冻结注意力机制层

# data="10,5 15,10 20,5 20,10"

# model=maml+exp11_1k_opt_dim128_layer4

# echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
#     python "${t}/DAN_finetuning_freeze.py" --logdir ./runs/transfer_${model}_${n_j}x${n_m}_${lr}_freeze \
#                                     --model_suffix exp12_${model}_${n_j}x${n_m}_freeze \
#                                     --finetuning_model $model \
#                                     --max_updates 100 \
#                                     --n_j $n_j \
#                                     --n_m $n_m \
#                                     --num_envs 5 \
#                                     --hidden_dim_actor 128 \
#                                     --hidden_dim_critic 128 \
#                                     --num_mlp_layers_actor 4 \
#                                     --num_mlp_layers_critic 4 \
#                                     --lr 0.003
# done

exp=exp13

echo $exp
# python ${t}/multi_task_maml_exp13.py --logdir ./runs/exp13_maml \
#                                         --model_suffix exp13_1k_opt_dim512_layer3 \
#                                         --maml_model True \
#                                         --meta_iterations 300 \
#                                         --num_tasks ${num_tasks} \
#                                         --max_updates 1000 \
#                                         --num_envs 5 \
#                                         --hidden_dim_actor 512 \
#                                         --hidden_dim_critic 512 \
#                                         --num_mlp_layers_actor 3 \
#                                         --num_mlp_layers_critic 3 \
#                                         --n_j_options 8 10 11 13 15 16 17 20 21 \
#                                         --n_m_options 4 5 8 10 12 15



# echo 试验13测试

# data="10,5 15,10 20,5 20,10"

# model=maml+exp13_1k_opt_dim512_layer3

# echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
#     python "${t}/DAN_finetuning_freeze.py" --logdir ./runs/transfer_${model}_${n_j}x${n_m}_${lr}_freeze \
#                                     --model_suffix exp12_${model}_${n_j}x${n_m}_freeze \
#                                     --finetuning_model $model \
#                                     --max_updates 100 \
#                                     --n_j $n_j \
#                                     --n_m $n_m \
#                                     --num_envs 5 \
#                                     --hidden_dim_actor 512 \
#                                     --hidden_dim_critic 512 \
#                                     --num_mlp_layers_actor 3 \
#                                     --num_mlp_layers_critic 3 \
#                                     --lr 0.003
# done

# echo exp13 fast_adapt5
# python ${t}/multi_task_maml_exp13.py --logdir ./runs/exp13_maml \
#                                         --model_suffix exp13_1k_opt_dim512_layer3_ap5 \
#                                         --maml_model True \
#                                         --meta_iterations 300 \
#                                         --num_tasks ${num_tasks} \
#                                         --max_updates 1000 \
#                                         --num_envs 5 \
#                                         --hidden_dim_actor 512 \
#                                         --hidden_dim_critic 512 \
#                                         --num_mlp_layers_actor 3 \
#                                         --num_mlp_layers_critic 3 \
#                                         --n_j_options 8 10 11 13 15 16 17 20 21 \
#                                         --n_m_options 4 5 8 10 12

# data="10,5 15,10 20,5 20,10"

# model=maml+exp13_1k_opt_dim512_layer3_ap5

# echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
#     python "${t}/DAN_finetuning.py" --logdir ./runs/$exp/transfer_${model}_${n_j}x${n_m}_${lr} \
#                                     --model_suffix exp12_${model}_${n_j}x${n_m} \
#                                     --finetuning_model $model \
#                                     --max_updates 100 \
#                                     --n_j $n_j \
#                                     --n_m $n_m \
#                                     --num_envs 5 \
#                                     --hidden_dim_actor 512 \
#                                     --hidden_dim_critic 512 \
#                                     --num_mlp_layers_actor 3 \
#                                     --num_mlp_layers_critic 3 \
#                                     --lr 0.003
# done


echo 测试同步
