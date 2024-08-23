import os, sys
os.environ['ON_PY']="1"
from params import parser
from train.Trainer import *

EXP="MAMLEC"
print(EXP)
hidden_dim=64

# 本试验特殊参数
n_j_options = [ "23", "13", "15"]
n_m_options = [ "17", "12", "5"]
op_per_job_options=n_m_options
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
print(n_j_options, n_m_options, op_per_job_options)

LOG_DIR="./runs/"+EXP

num_tasks=len(n_j_options)

meta_iterations=1500


def train_model():
    args = [
        "--logdir", f"./runs/{EXP}/{TIMESTAMP}",
        "--model_suffix", EXP,
        "--meta_iterations", f"{meta_iterations}",
        "--maml_model", "True", 
        "--num_tasks", f"{num_tasks}",
        "--hidden_dim_actor", f"{hidden_dim}",
        "--hidden_dim_critic",  f"{hidden_dim}",
        "--n_j_options", *n_j_options, 
        "--n_m_options", *n_m_options, 
        "--op_per_job_options", *op_per_job_options, 
        "--fea_j_input_dim", "12", 
        "--fea_m_input_dim", "9",
        "--model_source", "SD2EC",
        "--data_source", "SD2EC",
        "--num_envs", "20"
        ,"--seed_train", "234"
        ,"--reset_env_timestep", "50"
        ,"--low", "5"
        ,'--lr', "3e-4",
        
        ]
    print(args)
    configs = parser.parse_args(args=args)
    
    trainer = MultiTaskTrainerEc(configs)

    trainer.train()

train_model()

