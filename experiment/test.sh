TEST_DIR=./test_script

echo 测试1： maml模型泛化性测试\(few-shot\)
python ${TEST_DIR}/trained_model.py --data_source=10x6 \
                                    --test_data  10x5+mix 11x6+mix 10x5+test 10x6+mix 11x5+mix 11x6+mix 15x10+mix 20x5+mix 20x10+mix 30x10+mix \
                                    --model_source=SD2 \
                                    --test_model 10x5+mix+SD2 10x5+mix+exp8 10x5+mix+exp7 \
                                    --adapt_nums 50 
                                    # > test1.log

# python ${TEST_DIR}/heuristic.py --data_source 10x6 \
#                                     --test_data  10x5+mix 11x6+mix 10x5+test 10x6+mix 11x5+mix 11x6+mix 15x10+mix 20x5+mix 20x10+mix 30x10+mix \
#                                     --model_source SD2 \
#                                     --test_model 10x5+mix+exp6 10x5+mix+exp7

