import os, sys
os.environ['ON_PY']="1"
from params import parser
from common_utils import setup_seed
from data_utils import pack_data_from_config
import numpy as np
from test_script.base import Test
import matplotlib.pyplot as plt
import numpy as np


# 将工作目录更改为上一级目录
os.chdir("/work/home/lxx_hzau/project/FJSP-DRL-main")
instances = [ "10x5EC+ECMK", "20x5EC+ECMK", "15x10EC+ECMK", "20x10EC+ECMK", ]
test_data_list = [ "10x5+mix", "20x5+mix", "15x10+mix", "20x10+mix",]

plot_dict = {}
for key in instances:
    plot_dict[key] = [np.array(11) for _ in range(4)]

args = ["--test_data", *test_data_list,
        "--test_model", *instances]

ec_args = ["--fea_j_input_dim", "12", 
    "--fea_m_input_dim", "9",
    '--factor_Mk', "0.0",
    '--factor_Ec', "1.0", 
    "--model_source", "SD2EC",
    "--data_source", "SD2EC0",
    "--lr", "1e-4",
    ]

args = [*ec_args, *args]

print(args)

# DAN 解
configs = parser.parse_args(args=args)

setup_seed(configs.seed_test)

test_model = []

for model_name in configs.test_model:
    test_model.append((f'./trained_network/{configs.model_source}/{model_name}.pth', model_name))
print(test_model)
test_data = pack_data_from_config(configs.data_source, configs.test_data)

print(os.getcwd())
baseline_makespans = []
baseline_EC = []
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
        # test.env.render()
    result_5_times = np.array(result_5_times)

    save_result = np.min(result_5_times, axis=0)
    print("testing results:")
    print(f"makespan(greedy): ", save_result[:, 0].mean())
    baseline_makespans.append(save_result[:, 0].mean())
    baseline_EC.append(save_result[:, 1].mean())
    # print(f"time: ", save_result[:, 2].mean())
    # print(f"Max fast_adapt cnt:", save_result[:, 2].max())
    # print(f"Average fast_adapt time:", save_result[:, 3].mean())
    print("="*100)

baseline_makespans = np.array(baseline_makespans)
baseline_EC = np.array(baseline_EC)
print(baseline_makespans)
print(baseline_EC)


