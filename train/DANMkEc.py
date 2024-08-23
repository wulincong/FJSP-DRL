from datetime import datetime

from params import parser
from Trainer import DANEcMktrainer
n_j = 10
n_m = 5

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

def train_model(n_j, n_m, model_suffix, factor_Mk, factor_Ec):
    configs = parser.parse_args(args=[
        "--logdir", f"./runs/EcMk{n_j}-{n_m}/{factor_Mk}-{factor_Ec}/{TIMESTAMP}",
        "--model_suffix", model_suffix,
        "--max_updates", "1500",
        "--n_j", f"{n_j}",
        "--n_m", f"{n_m}",
        "--fea_j_input_dim", "16", 
        "--fea_m_input_dim", "11",
        '--factor_Mk', f"{factor_Mk}",
        '--factor_Ec', f"{factor_Ec}",
        "--model_source", "SD2EC0",
        "--data_source", "SD2EC0",
        "--num_envs", "20"
        ,"--reset_env_timestep", "50"
        ,"--seed_train", "234"
        ,"--low", "5"
        ,'--lr', "3e-4",
        ])
    trainer = DANEcMktrainer(configs)

    trainer.train()


for n_j, n_m in [(10,5), (20,5), (15, 10),(20, 10)]:
    train_model(n_j, n_m, "ECMK", 0.8, 0.2)

# for i in range(10):
#     energy_weight= i / 10
#     makespan_weight=(10-i) / 10

#     train_model(10, 5, "ECMK", makespan_weight, energy_weight)
