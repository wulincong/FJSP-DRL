import os, sys
os.environ['ON_PY']="1"
from params import parser
from train.Trainer import *

n_j_options="5  7  10 13 15 17 20 23 25"
n_m_options="10 10 10 10 10 10 10 10 10"
op_per_job_options="5 5 5 5 5 5 5 5 5"

EXP="EXP19"
print(EXP)
hidden_dim=512


# 本试验特殊参数
n_j_options=[int(_) for _ in np.linspace(5, 25, 6)]
n_m_options=n_j_options[::-1]
op_per_job_options=[10 for _ in n_m_options]
op_per_job_options=n_m_options
n_j_options=[13, 17, 21, 25, 25]
n_m_options=[5,  13, 9,  5, 17]
op_per_job_options=n_m_options
print(n_j_options, n_m_options, op_per_job_options)

LOG_DIR="./runs/"+EXP

num_tasks=len(n_j_options)

# multi_task_maml_exp.py 脚本的特定参数
meta_iterations=200


def train_maml(file=None, model_suffix=None, train_model = "maml"):
    if model_suffix is None: model_suffix = EXP+train_model
    logdir_maml=LOG_DIR+ f"/{train_model}/train_model"
    commd = f'python ./train/multi_task_maml_exp18.py --logdir {logdir_maml} --model_suffix {model_suffix} --maml_model True \
--meta_iterations {meta_iterations} --num_tasks {num_tasks} \
--hidden_dim_actor {hidden_dim} --hidden_dim_critic {hidden_dim} \
--n_j_options {" ".join(map(str, n_j_options))} --n_m_options {" ".join(map(str, n_m_options))} --op_per_job_options {" ".join(map(str, op_per_job_options))}'
    print(commd, file=file)
    args = commd.split()
    args = args[2:]
    # print(args)
    return f"maml+{model_suffix}", commd, args




model_name, commd, args = train_maml()
print(args)
configs = parser.parse_args(args=args)
trainer = MultiTaskECTrainer(configs)
# trainer = MultiTaskTrainer(configs)
trainer.train()

# trainer = MultiTaskTrainerCustomize(configs)
# trainer.train(inner_update_params_startswith=[])
# trainer.train(inner_update_params_startswith=['feature_exact'])

# trainer = MultiTaskTrainerCustomize(configs)
# trainer.train(inner_update_params_startswith=['critic'])

# trainer = MultiTaskTrainerCustomize(configs)
# trainer.train(inner_update_params_startswith=['actor'])

# trainer = MultiTaskTrainerCustomize(configs)
# trainer.train(inner_update_params_startswith=['feature_exact', 'critic'])


# _, _, args = train_maml(train_model="pre_train")
# print(args)
# configs = parser.parse_args(args=args)
# pretrainer = PretrainTrainer(configs)
# pretrainer.train()





