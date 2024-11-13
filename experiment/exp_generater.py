import numpy as np

shell_head = """#!/bin/sh
#SBATCH -J torch
#SBATCH -p xhhgnormal   #修改队列名称，whichpartition查看队列名称
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
set -x
source ~/.bashrc
module load nvidia/cuda/11.6
conda activate RL-torch
"""

list_j = [5, 8, 11, 14, 17, 20, 23, 25]
list_m = [5, 8, 11, 14, 17, 20, 23, 25]


exp="DAN_j_m"

for n_j in list_j:

    commd_list = []
    for n_m in list_m:
        commd = f'python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model {n_j}x{n_m}EC+ECMK --test_data {n_j}x{n_m}+mix'
        commd_list.append(commd)
    commds = "\n".join(commd_list)
    print(commds)
    print("#"*100)

# exp="DAN_j_m"

# for n_j in list_j:

#     commd_list = []
#     for n_m in list_m:
#         commd = f'python train/DANEC.py --n_j {n_j} --n_m {n_m}'
#         commd_list.append(commd)

#     commds = "\n".join(commd_list)
#     with open(f"./DANEC_j_m/DANEC_{n_j}.sh", "w") as f:
#         s = f'''{shell_head}\n{commds}'''
#         f.write(s)
