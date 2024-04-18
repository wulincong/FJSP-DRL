from datetime import datetime

from params import parser
from Trainer import DANMkEcTrainer
n_j = 10
n_m = 5

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

def train_model(n_j, n_m, model_suffix, factor_Mk, factor_Ec):
    configs = parser.parse_args(args=[
        "--logdir", f"./runs/EcMk{n_j}-{n_m}/{factor_Mk}-{factor_Ec}/{TIMESTAMP}",
        "--model_suffix", model_suffix,
        "--max_updates", "1000",
        "--n_j", f"{n_j}",
        "--n_m", f"{n_m}",
        "--fea_j_input_dim", "12", 
        "--fea_m_input_dim", "9",
        '--factor_Mk', f"{factor_Mk}",
        '--factor_Ec', f"{factor_Ec}",
        "--model_source", "SD2EC",
        "--data_source", "SD2EC",
        # "--num_envs", "10"
        ])
    trainer = DANMkEcTrainer(configs)

    trainer.train()


# for n_j, n_m in [(10,5), (20,5), (15, 10), (20,10)]:
#     train_model(n_j, n_m, "EC", 0.0, 1.0)

# for i in range(0, 10):
#     factor_Mk = i / 10.0
#     factor_Ec = 1 - factor_Mk
#     print(i / 10.0)
#     train_model(10, 5, "ECMK", factor_Mk, factor_Ec)

# train_model(15, 5, "ECMK", 0.0, 1.0)
train_model(20, 5, "ECMK", 0.0, 1.0)
train_model(10, 10, "ECMK", 0.0, 1.0)