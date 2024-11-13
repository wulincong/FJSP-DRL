#!/bin/sh
#SBATCH -J torch
#SBATCH -p xhhgnormal   #修改队列名称，whichpartition查看队列名称
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1
source ~/.bashrc
module load nvidia/cuda/11.6
conda activate RL-torch


python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 20x5EC+ECMK --test_data 20x5+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 20x8EC+ECMK --test_data 20x8+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 20x11EC+ECMK --test_data 20x11+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 20x14EC+ECMK --test_data 20x14+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 20x17EC+ECMK --test_data 20x17+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 20x20EC+ECMK --test_data 20x20+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 20x23EC+ECMK --test_data 20x23+mix
python ./test_script/trained_model_ec.py --fea_j_input_dim 12 --fea_m_input_dim 9 --model_source SD2EC0 --data_source SD2EC0 --seed_train 234 --test_model 20x25EC+ECMK --test_data 20x25+mix
####################################################################################################
