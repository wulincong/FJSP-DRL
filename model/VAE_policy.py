import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomFeatureExtractor, self).__init__(observation_space, features_dim)
        # 获取输入维度
        input_dim = observation_space.shape[0]
        # 定义自定义网络
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim)
        )

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
