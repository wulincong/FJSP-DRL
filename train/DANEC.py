from params import parser
from Trainer import DANMkEcTrainer
n_j = 10
n_m = 5

configs = parser.parse_args(args=[
    "--logdir", f"./runs/EcMk{n_j}-{n_m}/0-1",
    "--model_suffix", "free",
    "--max_updates", "1000",
    "--n_j", f"{n_j}",
    "--n_m", f"{n_m}",
    "--num_envs", "10",
    "--fea_j_input_dim", "12", 
    "--fea_m_input_dim", "9",
    '--factor_Mk', "0.0",
    '--factor_Ec', "1.0"
    # "--num_envs", "10",
    # '--high', '10'
    ])
trainer = DANMkEcTrainer(configs)

trainer.train()

configs = parser.parse_args(args=[
    "--logdir", f"./runs/EcMk{n_j}-{n_m}/1-0",
    "--model_suffix", "free",
    "--max_updates", "1000",
    "--n_j", f"{n_j}",
    "--n_m", f"{n_m}",
    "--fea_j_input_dim", "12", 
    "--fea_m_input_dim", "9",
    '--factor_Mk', "1.0",
    '--factor_Ec', "0.0"
    # "--num_envs", "10",
    # '--high', '10'
    ])
trainer = DANMkEcTrainer(configs)

trainer.train()

configs = parser.parse_args(args=[
    "--logdir", f"./runs/EcMk{n_j}-{n_m}/1-1",
    "--model_suffix", "free",
    "--max_updates", "1000",
    "--n_j", f"{n_j}",
    "--n_m", f"{n_m}",
    "--fea_j_input_dim", "12", 
    "--fea_m_input_dim", "9",
    '--factor_Mk', "0.5",
    '--factor_Ec', "0.5"
    # "--num_envs", "10",
    # '--high', '10'
    ])

trainer = DANMkEcTrainer(configs)

trainer.train()
