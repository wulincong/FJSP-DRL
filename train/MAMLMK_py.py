from datetime import datetime
import numpy as np
from params import parser
from Trainer import MultiTaskTrainer
import matplotlib.pyplot as plt
n_j = 10
n_m = 5

exp = "MAMLMK"
hidden_dim=64
n_j_options=[str(int(_)) for _ in np.linspace(5, 25, 6)]
n_m_options=n_j_options[::-1]

n_j_options = ["15", "10",]
n_m_options = ["5", "5",]
op_per_job_options=n_m_options
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
num_tasks=len(n_j_options)


args = [
    "--logdir", f"./runs/{exp}/{TIMESTAMP}",
    "--model_suffix", exp,
    "--meta_iterations", "2000",
    "--maml_model", "True", 
    "--num_tasks", f"{num_tasks}",
    "--hidden_dim_actor", f"{hidden_dim}",
    "--hidden_dim_critic",  f"{hidden_dim}",
    "--n_j_options", *n_j_options, 
    "--n_m_options", *n_m_options, 
    "--op_per_job_options", *op_per_job_options, 
    ]
print(args)
configs = parser.parse_args(args=args)

trainer = MultiTaskTrainer(configs)

trainer.train()

from sklearn.decomposition import PCA
from model.PPO import PPO_initialize
import torch
# 初始化 PPO 和参数列表
models = [f"{n_j}x{n_m}x0+mix" for n_j in range(5, 26, 5) for n_m in range(5, 26, 5) if n_j > n_m]
models.append("10x5+mix+SD2")
args = ["--test_data", "10x5+mix", "20x5+mix", "15x10+mix", "20x10+mix", "--test_model", *models]
configs = parser.parse_args(args=args)
test_model = [(f'./trained_network/{configs.model_source}/{model_name}.pth', model_name) for model_name in configs.test_model]

param_list = []

# 加载模型参数
for model_path, model_name in test_model:
    ppo = PPO_initialize(configs)
    # print(model_path)
    ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))
    parameters = list(ppo.policy.actor.parameters())
    
    param_list.append(parameters)

train_param = trainer.train_param_list


collation_param = [*param_list, *train_param]
param_matrix = np.array([np.concatenate([p.data.cpu().numpy().flatten() for p in params]).flatten() for params in collation_param])

# 使用 PCA 进行降维
pca = PCA(n_components=2)
params_reduced = pca.fit_transform(param_matrix)

# 绘制结果，为每个点添加模型名称标签
plt.figure(figsize=(10, 8))  # 可以调整大小以更好地适应所有标签
plt.scatter(params_reduced[:, 0], params_reduced[:, 1], marker='o')

print(params_reduced)
# 为每个点添加文本标签
end_i = 0
for i, label in enumerate([name for _, name in test_model]):
    plt.annotate(label[:-6], (params_reduced[i, 0], params_reduced[i, 1]))
    end_i = i

while end_i < len(params_reduced):
    plt.annotate(f"{end_i - len(test_model)}", (params_reduced[end_i, 0], params_reduced[end_i, 1]))
    end_i += 1

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Model Parameters')
plt.savefig("./training_params.png")
