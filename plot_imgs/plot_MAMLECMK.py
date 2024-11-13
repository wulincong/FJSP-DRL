import os, sys
os.environ['ON_PY']="1"
from params import parser
from common_utils import setup_seed
from data_utils import pack_data_from_config
import numpy as np
from test_script.base import Test
# import matplotlib.pyplot as plt
import numpy as np

notebook_dir = os.getcwd()
print(notebook_dir)
# 将工作目录更改为上一级目录
os.chdir("/work/home/lxx_hzau/project/FJSP-DRL-main")

instances = [ "10x5EC+ECMK", "20x5EC+ECMK", "15x10EC+ECMK", "20x10EC+ECMK", ]
test_data_list = [ "10x5+mix", "20x5+mix", "15x10+mix", "20x10+mix",]



args = ["--test_data", *test_data_list,
        "--test_model", *instances]

ec_args = ["--fea_j_input_dim", "16", 
    "--fea_m_input_dim", "11",
    '--factor_Mk', "0.8",
    '--factor_Ec', "0.2", 
    "--model_source", "SD2EC0",
    "--data_source", "SD2EC0",
    "--lr", "3e-4",
    ]

args = [*ec_args, *args]

print(args)

plot_dict = {}
for key in instances:
    plot_dict[key] = [np.array(11) for _ in range(4)]


# DAN 解
configs = parser.parse_args(args=args)

setup_seed(configs.seed_test)

test_model = []

for model_name in configs.test_model:
    test_model.append((f'./trained_network/{configs.model_source}/{model_name}.pth', model_name))
print(test_model)
test_data = pack_data_from_config(configs.data_source, configs.test_data)


baseline_makespans = []
baseline_EC = []
baseline_R = []
for i in range(len(test_model)):
    model = test_model[i]
    data = test_data[i]
    print("datta[1]: ",data[1])
    print("-" * 25 + "Test Learned Model" + "-" * 25)
    print(f"test data name: {data[1]}")
    finetuning = True if model[1].startswith("maml") else False
    print(f"Model name : {model[1]}")
    result_5_times = []
    for j in range(2):
        test = Test(configs, data[0], model[0])     
        result = test.greedy_strategy(finetuning=finetuning)
        result_5_times.append(result)
    result_5_times = np.array(result_5_times)

    save_result = np.min(result_5_times, axis=0)
    print("testing results:")
    print(f"makespan(greedy): ", save_result[:, 0].mean())
    baseline_makespans.append(save_result[:, 0].mean())
    baseline_EC.append(save_result[:, 1].mean())
    baseline_R.append(np.mean(save_result[:, 3]))
    # print(f"time: ", save_result[:, 2].mean())
    # print(f"Max fast_adapt cnt:", save_result[:, 2].max())
    # print(f"Average fast_adapt time:", save_result[:, 3].mean())
    print("="*100)

baseline_makespans = np.array(baseline_makespans)
baseline_EC = np.array(baseline_EC)
baseline_R = np.array(baseline_R)
# print(baseline_makespans)
# print(baseline_EC)
print(baseline_R)
# MAML finetuning
# 获取finetuning的每个过程

def MAML_finetuning_test(args):
    configs = parser.parse_args(args=args)
    # print(configs.hidden_dim_actor, configs.hidden_dim_critic)
    test_model = []
    for model_name in configs.test_model:
        test_model.append((f'./trained_network/{configs.model_source}/{model_name}.pth', model_name))
    print(test_model)

    model = test_model[0]

    test_data = pack_data_from_config(configs.data_source, configs.test_data)

    makespans = []
    finetuning_makespans = []
    finetuning_ecs = []
    finetuning_R = []
    for data in test_data:
        ### 对每个实例
        print("datta[1]: ",data[1])
        print("-" * 25 + "Test Learned Model" + "-" * 25)
        print(f"test data name: {data[1]}")
        if model[1].startswith("maml"): finetuning = True
        print(f"Model name : {model[1]}")
        result_5_times = []
        test = Test(configs, data[0], model[0])
        ep_reward_list = test.finetuning(times=5)
        finetuning_R.append(ep_reward_list)

    return finetuning_R

model = "maml+MAMLEC1723516249"

args = [*ec_args, "--test_data", *test_data_list,  "--test_model", model, 
                # "--hidden_dim_actor", "512", "--hidden_dim_critic", "512",
        ]

maml_finetuning_R = MAML_finetuning_test(args)

print(maml_finetuning_R)


# pretrain finetuning

args = [*ec_args, "--test_data", *test_data_list, 
        # "--hidden_dim_actor", "512", "--hidden_dim_critic", "512",
        "--test_model", "PreTrain"]

pretrain_finetuning_R = MAML_finetuning_test(args)



print(pretrain_finetuning_R)


# random
import torch
from model.PPO import PPO_initialize

ppo = PPO_initialize(configs)

torch.save(ppo.policy.state_dict(), f'./trained_network/SD2EC0/test_random.pth')

args = [*ec_args, "--test_data", *test_data_list, 
        "--test_model", "test_random"]

random_finetuning_R = MAML_finetuning_test(args)

print(random_finetuning_R)
for idx, key in enumerate(instances):
    # plot_dict[key][0] = [baseline_makespans[idx]  for _ in range(configs.adapt_nums)]
    plot_dict[key][0] = np.full(5, baseline_R[idx])
    plot_dict[key][1] = maml_finetuning_R[idx][:5]
    plot_dict[key][2] = pretrain_finetuning_R[idx][:5]
    plot_dict[key][3] = random_finetuning_R[idx][:5]

print(plot_dict)

