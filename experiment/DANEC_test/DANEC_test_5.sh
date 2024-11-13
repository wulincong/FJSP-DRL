#!/bin/sh
#SBATCH -J torch
#SBATCH -p xhhgnormal   #修改队列名称，whichpartition查看队列名称
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
source ~/.bashrc
module load nvidia/cuda/11.6
conda activate RL-torch

python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 5x5EC+ECMK --test_data 5x5+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 5x8EC+ECMK --test_data 5x8+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 5x11EC+ECMK --test_data 5x11+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 5x14EC+ECMK --test_data 5x14+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 5x17EC+ECMK --test_data 5x17+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 5x20EC+ECMK --test_data 5x20+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 5x23EC+ECMK --test_data 5x23+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 5x25EC+ECMK --test_data 5x25+mix
####################################################################################################