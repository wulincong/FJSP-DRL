#!/bin/sh
#SBATCH -J torch
#SBATCH -p xhhgnormal   #修改队列名称，whichpartition查看队列名称
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
source ~/.bashrc
module load nvidia/cuda/11.6
conda activate RL-torch


python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 14x5EC+ECMK --test_data 14x5+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 14x8EC+ECMK --test_data 14x8+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 14x11EC+ECMK --test_data 14x11+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 14x14EC+ECMK --test_data 14x14+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 14x17EC+ECMK --test_data 14x17+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 14x20EC+ECMK --test_data 14x20+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 14x23EC+ECMK --test_data 14x23+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 14x25EC+ECMK --test_data 14x25+mix
####################################################################################################