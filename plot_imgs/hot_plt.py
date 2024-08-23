import os, sys
from params import parser
from data_utils import pack_data_from_config
import numpy as np
from test_script.base import Test
from datetime import datetime
import torch
from model.PPO import PPO_initialize

log_timestamp = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
# MAML finetuning
# 获取finetuning的每个过程
def MAML_finetuning_test(args):
    configs = parser.parse_args(args=args)
    
    print(configs.hidden_dim_actor, configs.hidden_dim_critic)
    test_model = []

    for model_name in configs.test_model:
        test_model.append((f'./trained_network/{configs.model_source}/{model_name}.pth', model_name))
    print(test_model)

    model = test_model[0]
    if model == "test_random":
        ppo = PPO_initialize(configs)
        torch.save(ppo.policy.state_dict(), f'./trained_network/SD2/test_random.pth')

    test_data = pack_data_from_config(configs.data_source, configs.test_data)
    makespans = []
    finetuning_makespans = []
    for data in test_data:
        print("datta[1]: ",data[1])
        print("-" * 25 + "Test Learned Model" + "-" * 25)
        print(f"test data name: {data[1]}")
        if model[1].startswith("maml"): finetuning = True
        print(f"Model name : {model[1]}")
        result_5_times = []
        test = Test(configs, data[0], model[0])
        result = test.finetuning(times=4)
        finetuning_makespans.append(np.mean(result))
        print(result)
        # result = test_greedy_strategy(data[0], model[0], config.seed_test)
        # print(result)
        for j in range(2):
            result = test.greedy_strategy()
            result_5_times.append(result)
        result_5_times = np.array(result_5_times)

        save_result = np.min(result_5_times, axis=0)
        print("testing results:")
        print(f"makespan(greedy): ", save_result[:, 0].mean())
        makespans.append(save_result[:, 0].min())
        print(f"time: ", save_result[:, 1].mean())
        # print(f"Max fast_adapt cnt:", save_result[:, 2].max())
        # print(f"Average fast_adapt time:", save_result[:, 3].mean())
        print("="*100)

    return makespans, finetuning_makespans



def MAML_finetuning(model, dim=64):
    n_j_list = [5, 8, 11, 14, 17, 20, 23, 25]
    n_m_list = n_j_list
    
    test_data = []
    for n_j in n_j_list:
        for n_m in n_m_list:
            test_data.append(f"{n_j}x{n_m}+mix")
    
    args = ["--test_data", *test_data, 
        "--hidden_dim_actor", f"{dim}", "--hidden_dim_critic", f"{dim}",
         "--test_model", model]
    
    model_res, _ = MAML_finetuning_test(args = args)
    with open(f"./plot_imgs/hot_plt_data/model_res{model}.txt", "w") as file:
        print(model_res, file = file)

if __name__ == "__main__":
    # random
    MAML_finetuning("test_random", dim=64)

