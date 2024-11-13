from datetime import datetime
import numpy as np
from params import parser
from Trainer import MultiTaskTrainer
n_j = 10
n_m = 5

exp = "MAMLMK"
hidden_dim=64
n_j_options=[str(int(_)) for _ in np.linspace(5, 25, 6)]
n_m_options=n_j_options[::-1]

n_j_options = ["25", "25", "15"]
n_m_options = ["20", "15", "5"]
op_per_job_options=n_m_options
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
num_tasks=len(n_j_options)

def train_model():
    args = [
        "--logdir", f"./runs/{exp}/{TIMESTAMP}",
        "--model_suffix", exp,
        "--meta_iterations", "1500",
        "--maml_model", "True", 
        "--num_tasks", f"{num_tasks}",
        "--hidden_dim_actor", f"{hidden_dim}",
        "--hidden_dim_critic",  f"{hidden_dim}",
        "--n_j_options", *n_j_options, 
        "--n_m_options", *n_m_options, 
        "--op_per_job_options", *op_per_job_options, 
        "--reset_env_timestep", "50", 
        ]
    print(args)
    configs = parser.parse_args(args=args)
    
    trainer = MultiTaskTrainer(configs)

    trainer.train()

train_model()
