# python maml_train.py > train_log/maml_train.log

# 实验1： 无maml训练10x5  SD1
python train.py                 --n_j 10 \
                                --n_m 5 \
                                --data_source SD1 \
                                --model_suffix SD1 \
                                --logdir ./runs/exp1_10x5_SD1 \
                                --max_updates 1000 \
                            > train_log/exp1_10x5_SD1.log

python test_trained_model.py 	--data_source SD1	\
				                --model_source SD1	\
    				            --test_data 10x5    \
        			            --test_model 10x5	\
            			        --test_mode False	\
                		        --sample_times 100	\
                            >> train_log/exp1_10x5_SD1.log



# 实验2： maml训练10x5 SD1
python maml_train.py            --n_j 10 \
                                --n_m 5 \
                                --data_source SD1 \
                                --model_suffix maml_sd1 \
                                --logdir ./runs/exp2_maml_10x5_SD1 \
                                --meta_iterations 200 \
                                --num_tasks 5 \
                            > train_log/exp2_maml_10x5_SD1.log

python test_trained_model.py 	--data_source SD1	\
				                --model_source SD1	\
    				            --test_data 10x5    \
        			            --test_model 10x5+maml_sd1	\
            			        --test_mode False	\
                		        --sample_times 100	\
                            >> train_log/exp2_maml_10x5_SD1.log
