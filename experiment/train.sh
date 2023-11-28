# python maml_train.py > train_log/maml_train.log

# 实验1： 无maml训练10x5  SD1

t=./train
meta_iterations=3000
num_tasks=5
max_updates=$((meta_iterations * num_tasks))
ADAPT_NUMS=5
TEST_DIR=./test_script

# echo exp1

# python ${t}/DAN.py            --n_j 10 \
#                                 --n_m 5 \
#                                 --data_source SD1 \
#                                 --model_suffix SD1 \
#                                 --logdir ./runs/exp1_1_10x5_SD1 \
#                                 --max_updates ${max_updates} \
#                             > train_log/exp1_1_10x5_SD1.log

# python ${TEST_DIR}/test_trained_model.py 	--data_sourc`e SD1	\
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


# echo 试验10：用原论文的训练方法加上元学习的模型进行finetuning


# data="10,5 15,10 20,5 20,10"

# # data="15,10 20,5 20,10"
# for model in 10x5+mix+exp9 10x5+mix+exp9_1w 15x10+mix+SD2 10x5+mix+SD2 
# # for model in 10x5+mix+exp11_3k 
# do
#     echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
#         python "${t}/DAN_finetuning.py" --logdir ./runs/transfer_${model}_${n_j}x${n_m} \
#                                         --model_suffix exp10_${model}_${n_j}x${n_m} \
#                                         --finetuning_model $model \
#                                         --max_updates 500 \
#                                         --n_j $n_j \
#                                         --n_m $n_m \
#                                         --num_envs 3 \
#                                         --hidden_dim_actor 128 \
#                                         --hidden_dim_critic 128 
#     done
# done


# echo 试验11：epoch内MAML 使用相同类型问题训练，增加模型复杂度
# python ${t}/multi_task_maml_exp11.py --logdir ./runs/exp11_maml \
#                                         --model_suffix exp11_1k_add_hidden_dim \
#                                         --meta_iterations 1000 \
#                                         --num_tasks ${num_tasks} \
#                                         --max_updates ${max_updates} \
#                                         --num_envs 5 \
#                                         --hidden_dim_actor 128 \
#                                         --hidden_dim_critic 128 \
#                                         --num_mlp_layers_actor 3 \
#                                         --num_mlp_layers_critic 3 

# python ${t}/multi_task_maml_exp11.py --logdir ./runs/exp11_maml \
#                                         --model_suffix exp11_1k_add_dim256 \
#                                         --meta_iterations 1000 \
#                                         --num_tasks ${num_tasks} \
#                                         --max_updates ${max_updates} \
#                                         --num_envs 5 \
#                                         --hidden_dim_actor 256 \
#                                         --hidden_dim_critic 256 \
#                                         --num_mlp_layers_actor 3 \
#                                         --num_mlp_layers_critic 3 


# python ${t}/multi_task_maml_exp11.py --logdir ./runs/exp11_maml \
#                                         --model_suffix exp11_1k_add_layers5 \
#                                         --meta_iterations 1000 \
#                                         --num_tasks ${num_tasks} \
#                                         --max_updates ${max_updates} \
#                                         --num_envs 5 \
#                                         --hidden_dim_actor 64 \
#                                         --hidden_dim_critic 64 \
#                                         --num_mlp_layers_actor 5 \
#                                         --num_mlp_layers_critic 5 

# python ${t}/multi_task_maml_exp11.py --logdir ./runs/exp11_maml \
#                                         --model_suffix exp11_1k_add_layers7 \
#                                         --meta_iterations 1000 \
#                                         --num_tasks ${num_tasks} \
#                                         --max_updates ${max_updates} \
#                                         --num_envs 5 \
#                                         --hidden_dim_actor 64 \
#                                         --hidden_dim_critic 64 \
#                                         --num_mlp_layers_actor 7 \
#                                         --num_mlp_layers_critic 7 


# python ${t}/multi_task_maml_exp11.py --logdir ./runs/exp11_maml \
#                                         --model_suffix exp11_1k_add_layers_add_dim \
#                                         --meta_iterations 1000 \
#                                         --num_tasks ${num_tasks} \
#                                         --max_updates ${max_updates} \
#                                         --num_envs 5 \
#                                         --hidden_dim_actor 128 \
#                                         --hidden_dim_critic 128 \
#                                         --num_mlp_layers_actor 5 \
#                                         --num_mlp_layers_critic 5 



# python ${t}/multi_task_maml_exp11.py --logdir ./runs/exp11_maml \
#                                         --model_suffix exp11_1k_j_10_15_20_m_5_10 \
#                                         --maml_model True \
#                                         --meta_iterations 1000 \
#                                         --num_tasks ${num_tasks} \
#                                         --max_updates 1000 \
#                                         --num_envs 5 \
#                                         --hidden_dim_actor 256 \
#                                         --hidden_dim_critic 256 \
#                                         --num_mlp_layers_actor 5 \
#                                         --num_mlp_layers_critic 5 \
#                                         --n_j_options 10 15 20 \
#                                         --n_m_options 5 10

# data="10,5 15,10 20,5 20,10"

# for model in maml+exp11_1k_j_10_15_20_m_5_10
# do
#     echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
#         python "${t}/DAN_finetuning.py" --logdir ./runs/transfer_${model}_${n_j}x${n_m} \
#                                         --model_suffix exp11_${model}_${n_j}x${n_m} \
#                                         --finetuning_model $model \
#                                         --max_updates 500 \
#                                         --n_j $n_j \
#                                         --n_m $n_m \
#                                         --num_envs 5 \
#                                         --hidden_dim_actor 256 \
#                                         --hidden_dim_critic 256 \
#                                         --num_mlp_layers_actor 5 \
#                                         --num_mlp_layers_critic 5 
#     done
# done

# python ${t}/multi_task_maml_exp11.py --logdir ./runs/exp11_maml \
#                                         --model_suffix exp11_1k_j_8_21_m_4_12 \
#                                         --maml_model True \
#                                         --meta_iterations 1000 \
#                                         --num_tasks ${num_tasks} \
#                                         --max_updates 1000 \
#                                         --num_envs 5 \
#                                         --hidden_dim_actor 256 \
#                                         --hidden_dim_critic 256 \
#                                         --num_mlp_layers_actor 5 \
#                                         --num_mlp_layers_critic 5 \
#                                         --n_j_options 8 11 13 16 17 21 \
#                                         --n_m_options 4 8 12

# data="10,5 15,10 20,5 20,10"

# for model in maml+exp11_1k_j_8_21_m_4_12
# do
#     echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
#         python "${t}/DAN_finetuning.py" --logdir ./runs/transfer_${model}_${n_j}x${n_m} \
#                                         --model_suffix exp11_${model}_${n_j}x${n_m} \
#                                         --finetuning_model $model \
#                                         --max_updates 500 \
#                                         --n_j $n_j \
#                                         --n_m $n_m \
#                                         --num_envs 5 \
#                                         --hidden_dim_actor 256 \
#                                         --hidden_dim_critic 256 \
#                                         --num_mlp_layers_actor 5 \
#                                         --num_mlp_layers_critic 5 
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

# data="10,5 15,10 20,5 20,10"

# model=maml+exp11_1k_opt
# for lr in 0.0003 0.003 0.03 0.1
# do
#     echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
#         python "${t}/DAN_finetuning.py" --logdir ./runs/transfer_${model}_${n_j}x${n_m}_${lr} \
#                                         --model_suffix exp11_${model}_${n_j}x${n_m} \
#                                         --finetuning_model $model \
#                                         --max_updates 100 \
#                                         --n_j $n_j \
#                                         --n_m $n_m \
#                                         --num_envs 5 \
#                                         --hidden_dim_actor 256 \
#                                         --hidden_dim_critic 256 \
#                                         --num_mlp_layers_actor 5 \
#                                         --num_mlp_layers_critic 5 \
#                                         --lr $lr
#     done
# done




# data="10,5 15,10 20,5 20,10"

# model=maml+exp11_1k_opt

# echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
#     python "${t}/DAN_finetuning.py" --logdir ./runs/transfer_${model}_${n_j}x${n_m}_${lr} \
#                                     --model_suffix exp11_${model}_${n_j}x${n_m} \
#                                     --finetuning_model $model \
#                                     --max_updates 100 \
#                                     --n_j $n_j \
#                                     --n_m $n_m \
#                                     --num_envs 5 \
#                                     --hidden_dim_actor 128 \
#                                     --hidden_dim_critic 128 \
#                                     --num_mlp_layers_actor 5 \
#                                     --num_mlp_layers_critic 5 \
#                                     --lr 0.003
# done


# python ${t}/multi_task_maml_exp11.py --logdir ./runs/exp11_maml \
#                                         --model_suffix exp11_1k_opt_dim256_layer5 \
#                                         --maml_model True \
#                                         --meta_iterations 1000 \
#                                         --num_tasks ${num_tasks} \
#                                         --max_updates 1000 \
#                                         --num_envs 5 \
#                                         --hidden_dim_actor 256 \
#                                         --hidden_dim_critic 256 \
#                                         --num_mlp_layers_actor 5 \
#                                         --num_mlp_layers_critic 5 \
#                                         --n_j_options 8 10 11 13 15 16 17 20 21 \
#                                         --n_m_options 4 5 8 10 12

# data="10,5 15,10 20,5 20,10"

# model=maml+exp11_1k_opt_dim256_layer5

# echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
#     python "${t}/DAN_finetuning.py" --logdir ./runs/transfer_${model}_${n_j}x${n_m}_${lr} \
#                                     --model_suffix exp11_${model}_${n_j}x${n_m} \
#                                     --finetuning_model $model \
#                                     --max_updates 100 \
#                                     --n_j $n_j \
#                                     --n_m $n_m \
#                                     --num_envs 5 \
#                                     --hidden_dim_actor 256 \
#                                     --hidden_dim_critic 256 \
#                                     --num_mlp_layers_actor 5 \
#                                     --num_mlp_layers_critic 5 \
#                                     --lr 0.003
# done




# python ${t}/multi_task_maml_exp11.py --logdir ./runs/exp11_maml \
#                                         --model_suffix exp11_1k_opt_dim512_layer3 \
#                                         --maml_model True \
#                                         --meta_iterations 1000 \
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

# model=maml+exp11_1k_opt_dim512_layer3

# echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
#     python "${t}/DAN_finetuning.py" --logdir ./runs/transfer_${model}_${n_j}x${n_m}_${lr} \
#                                     --model_suffix exp11_${model}_${n_j}x${n_m} \
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

# python ${t}/multi_task_maml_exp11.py --logdir ./runs/exp11_maml \
#                                         --model_suffix exp11_1k_opt_dim128_layer4 \
#                                         --maml_model True \
#                                         --meta_iterations 1000 \
#                                         --num_tasks ${num_tasks} \
#                                         --max_updates 1000 \
#                                         --num_envs 5 \
#                                         --hidden_dim_actor 128 \
#                                         --hidden_dim_critic 128 \
#                                         --num_mlp_layers_actor 4 \
#                                         --num_mlp_layers_critic 4 \
#                                         --n_j_options 8 10 11 13 15 16 17 20 21 \
#                                         --n_m_options 4 5 8 10 12


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


echo exp13
python ${t}/multi_task_maml_exp13.py --logdir ./runs/exp13_maml \
                                        --model_suffix exp13_1k_opt_dim512_layer3 \
                                        --maml_model True \
                                        --meta_iterations 300 \
                                        --num_tasks ${num_tasks} \
                                        --max_updates 1000 \
                                        --num_envs 5 \
                                        --hidden_dim_actor 512 \
                                        --hidden_dim_critic 512 \
                                        --num_mlp_layers_actor 3 \
                                        --num_mlp_layers_critic 3 \
                                        --n_j_options 8 10 11 13 15 16 17 20 21 \
                                        --n_m_options 4 5 8 10 12



echo 试验13测试

data="10,5 15,10 20,5 20,10"

model=maml+exp13_1k_opt_dim512_layer3

echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
    python "${t}/DAN_finetuning_freeze.py" --logdir ./runs/transfer_${model}_${n_j}x${n_m}_${lr}_freeze \
                                    --model_suffix exp12_${model}_${n_j}x${n_m}_freeze \
                                    --finetuning_model $model \
                                    --max_updates 100 \
                                    --n_j $n_j \
                                    --n_m $n_m \
                                    --num_envs 5 \
                                    --hidden_dim_actor 512 \
                                    --hidden_dim_critic 512 \
                                    --num_mlp_layers_actor 3 \
                                    --num_mlp_layers_critic 3 \
                                    --lr 0.003
done

echo exp13 fast_adapt5
python ${t}/multi_task_maml_exp13.py --logdir ./runs/exp13_maml \
                                        --model_suffix exp13_1k_opt_dim512_layer3_ap5 \
                                        --maml_model True \
                                        --meta_iterations 300 \
                                        --num_tasks ${num_tasks} \
                                        --max_updates 1000 \
                                        --num_envs 5 \
                                        --hidden_dim_actor 512 \
                                        --hidden_dim_critic 512 \
                                        --num_mlp_layers_actor 3 \
                                        --num_mlp_layers_critic 3 \
                                        --n_j_options 8 10 11 13 15 16 17 20 21 \
                                        --n_m_options 4 5 8 10 12

