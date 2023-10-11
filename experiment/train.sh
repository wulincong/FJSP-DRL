# python maml_train.py > train_log/maml_train.log

# 实验1： 无maml训练10x5  SD1

t=./train
meta_iterations=50
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

# # python ${TEST_DIR}/test_trained_model.py 	--data_source SD1	\
# # 				                --model_source SD1	\
# #     				            --test_data 10x5    \
# #         			            --test_model 10x5	\
# #             			        --test_mode False	\
# #                 		        --sample_times 100	\
# #                             >> train_log/exp1_1_10x5_SD1.log

# # 实验1-2： 无maml训练10x5  SD2
# echo exp2
# python ${t}/DAN.py                 --n_j 10 \
#                                 --n_m 5 \
#                                 --data_source SD2 \
#                                 --model_suffix SD2 \
#                                 --logdir ./runs/exp1_2_10x5_SD2 \
#                                 --max_updates ${max_updates} \
#                             > train_log/exp1_2_10x5_SD2.log

# # python ${TEST_DIR}/test_trained_model.py 	--data_source SD2	\
# # 				                --model_source SD2	\
# #     				            --test_data 10x5+mix    \
# #         			            --test_model 10x5+mix	\
# #             			        --test_mode False	\
# #                 		        --sample_times 100	\
# #                             >> train_log/exp1_2_10x5_SD2.log


# # 实验2： maml训练10x5 SD1
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

# # 实验3： 对比maml训练相同env 10x5 SD2
# echo exp3
# python ${t}/same_env.py    --data_source SD2 \
#                             --model_suffix maml_sd2_same \
#                             --logdir ./runs/exp3_same_env_10x5_SD2 \
#                             --meta_iterations ${meta_iterations} \
#                             --num_tasks ${num_tasks} \
#                             --adapt_nums ${ADAPT_NUMS} \
#                         > train_log/exp3_same_env_10x5_SD2.log


# # 实验4： maml模型泛化性测试 10x5 扩展到11x5 SD2
python ./test_script/trained_model.py --data_source=10x6 \
                                        --test_data 10x6+mix 11x5+mix 11x6+mix \
                                        --model_source=SD2 \
                                        --test_model 10x5+mix+maml_sd2_same 10x5+mix+maml_sd2_same_



