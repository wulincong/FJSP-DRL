from stable_baselines3.common.env_util import SubprocVecEnv, DummyVecEnv
from stable_baselines3 import PPO
import numpy as np
import random
from data_utils import CaseGenerator
from gym_jobshop.envs.fjsp_env import FJSPEnv  # 假设 FJSPEnv 已正确定义
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from model.main_model import DualAttentionNetwork

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        # 获取输入维度
        input_dim = observation_space.shape[0]
        # 定义自定义网络
        self.network = DualAttentionNetwork()

    def forward(self, observations):
        return self.network(observations)


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(
            *args,
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            **kwargs,
        )


class MaskedPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(MaskedPolicy, self).__init__(*args, **kwargs)

    def predict(self, observation, action_mask=None, deterministic=False):
        """
        选择一个动作，考虑掩码。
        :param observation: 环境状态
        :param action_mask: 动作掩码，形状为 (action_space,) 的布尔数组
        :param deterministic: 是否选择最优动作
        :return: 动作和动作的概率分布
        """
        # 1. 获取原始动作分布
        actions, values, log_prob = self.forward(observation)


# 定义环境生成函数，支持初始化参数
def make_custom_vec_env(env_class, n_envs, env_kwargs_list, seed=None):
    """
    创建自定义向量化环境，并支持传递不同的初始化参数。
    :param env_class: 环境类
    :param n_envs: 并行环境数量
    :param env_kwargs_list: 每个环境的初始化参数列表 (每个元素是一个字典)
    :param seed: 随机种子
    :return: 向量化环境
    """
    def make_env(env_idx):
        def _init():
            env = env_class(**env_kwargs_list[env_idx])
            env.set_initial_data(env_kwargs_list[env_idx]["job_length"], env_kwargs_list[env_idx]["op_pt"])
            if seed is not None:
                env.seed(seed + env_idx)
            return env
        return _init

    # 确保 n_envs 与 env_kwargs_list 的长度一致
    assert len(env_kwargs_list) == n_envs, "env_kwargs_list 的长度必须等于 n_envs"
    return DummyVecEnv([make_env(i) for i in range(n_envs)])

def generate_env_and_train(n_j, n_m, op_per_job_min=10, op_per_job_max=20, n_envs=4, total_timesteps=100000, reset_interval=5000):
    """
    根据输入参数生成调度实例，创建向量化环境并训练模型。
    :param n_j: 作业数量
    :param n_m: 机器数量
    :param op_per_job_min: 每个作业的最少工序数量
    :param op_per_job_max: 每个作业的最多工序数量
    :param n_envs: 并行环境数量
    :param total_timesteps: 总训练步数
    :param reset_interval: 环境重新生成的时间步间隔
    """
    # 初始化随机种子
    current_seed = 0

    def create_new_env():
        nonlocal current_seed
        # 随机生成作业和工序数据
        case = CaseGenerator(n_j, n_m, op_per_job_min, op_per_job_max, flag_same_opes=False)
        job_length, op_pt, _ = case.get_case(0)

        # 构造每个环境的初始化参数
        env_kwargs_list = [{"num_jobs": n_j, "num_machines": n_m, "job_length": job_length, "op_pt": op_pt} for _ in range(n_envs)]

        # 创建新的向量化环境
        env = make_custom_vec_env(FJSPEnv, n_envs, env_kwargs_list, seed=current_seed)
        current_seed += n_envs  # 更新随机种子
        return env

    # 初始化环境和模型
    env = create_new_env()
    model = PPO(
        CustomPolicy,
        env,
        verbose=1,
        tensorboard_log="./ppo_fjsp_tensorboard/"
    )

    # 训练模型
    steps_remaining = total_timesteps
    while steps_remaining > 0:
        steps_to_train = min(reset_interval, steps_remaining)
        model.learn(total_timesteps=steps_to_train)
        steps_remaining -= steps_to_train

        if steps_remaining > 0:  # 如果还有剩余步数，重新生成环境
            print(f"重置环境数据，剩余训练步数: {steps_remaining}")
            env = create_new_env()
            model.set_env(env)

    # 保存模型
    model.save("ppo_fjsp")
    print("模型已保存为 'ppo_fjsp'")

# 使用实例
if __name__ == "__main__":
    n_j = 14  # 作业数量
    n_m = 11  # 机器数量
    generate_env_and_train(n_j, n_m)
