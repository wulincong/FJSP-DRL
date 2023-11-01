# python maml_train.py > train_log/maml_train.log

# 实验1： 无maml训练10x5  SD1

t=./train
meta_iterations=3000
num_tasks=7
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

# data="15,10 20,5 20,10"
# # for model in 10x5+mix+exp9 10x5+mix+exp9_1w 15x10+mix+SD2 10x5+mix+SD2 
# for model in 10x5+mix+exp9_1w 10x5+mix+SD2 
# do
#     echo $data | tr ' ' '\n' | while IFS=, read n_j n_m; do
#         python "${t}/DAN_finetuning.py" --logdir ./runs/transfer_${model}_${n_j}x${n_m} \
#                                         --model_suffix exp10_${model}_${n_j}x${n_m} \
#                                         --finetuning_model $model \
#                                         --max_updates 500 \
#                                         --n_j $n_j \
#                                         --n_m $n_m \
#                                         --num_envs 3
#     done
# done


echo 试验11：epoch内MAML 使用相同类型问题训练，
python ${t}/multi_task_maml_exp9.py --logdir ./runs/exp9_multi_task_maml \
                                        --model_suffix exp9_1w \
                                        --meta_iterations ${meta_iterations} \
                                        --num_tasks ${num_tasks} \
                                        --max_updates ${max_updates} \
                                        --num_envs 5 \
                                        --hidden_dim_actor 128 \
                                        --hidden_dim_critic 128 


