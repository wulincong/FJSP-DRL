import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from common_utils import nonzero_averaging, get_subdict, sample_action
from model.main_model import DualAttentionNetwork
from params import configs
from model.sub_layers import *
from stable_baselines3.common.distributions import Categorical

class DualAttentionNetwork_(DualAttentionNetwork):
    def __init__(self, observation_space, config=configs):
        super().__init__(config)
        
        # embedding_dim 是网络输出的嵌入维度，需从配置中获取
        embedding_dim = config.layer_fea_output_dim[-1]
        self.pair_input_dim = observation_space['pair_features'].shape[-1]
        # Calculate features_dim
        self.features_dim = 4 * embedding_dim  # 2 * embedding_dim (global) + 2 * embedding_dim (aggregated)

    def forward(self, observations, fast_weights=None) -> th.Tensor:
        # 调用父类的 forward 方法提取特征
        fea_j = observations['job_features']  # [sz_b, N, job_input_dim]
        fea_m = observations['mch_features']  # [sz_b, M, mch_input_dim]
        op_mask = observations['op_mask']     # [sz_b, N, 3]
        mch_mask = observations['mch_mask']   # [sz_b, M, M]
        comp_idx = observations['comp_idx']   # [sz_b, M, M, J]
        candidate = observations['candidate']

        fea_j, fea_m, fea_j_global, fea_m_global = super().forward(fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, fast_weights)
        sz_b, M, _, J = comp_idx.size() 
        d = fea_j.size(-1)

        candidate_idx = observations['candidate'].unsqueeze(-1).repeat(1, 1, d).type(torch.int64)
        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)  # [sz_b, J, embedding_dim]

        Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        # Combine global and pair features
        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        fea_pairs = observations['pair_features'].view(sz_b, -1, self.pair_input_dim)

        candidate_features = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, 
                                        Fea_Gj_input, Fea_Gm_input, fea_pairs), dim=-1)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)

        # Combine global features
        combined_global_features = torch.cat((fea_j_global, fea_m_global), dim=-1)  # [sz_b, global_dim]

        # Optionally include aggregated job and machine features
        fea_j_agg = fea_j.mean(dim=1)  # Average job features, [sz_b, embedding_dim]
        fea_m_agg = fea_m.mean(dim=1)  # Average machine features, [sz_b, embedding_dim]

        combined_features = torch.cat((combined_global_features, fea_j_agg, fea_m_agg), dim=-1)  # [sz_b, fixed_dim]

        return candidate_features, global_feature, combined_features


class DANIELPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, configs=configs, **kwargs):
        """
        DANIEL-based custom policy for Stable-Baselines3
        :param observation_space: Observation space (from gymnasium)
        :param action_space: Action space (from gymnasium)
        :param lr_schedule: Learning rate schedule
        """
        super(DANIELPolicy, self).__init__(observation_space, action_space, lr_schedule,
                                            **kwargs)
        self.features_extractor_class = DualAttentionNetwork_
        self.features_extractor_kwargs = {"config":configs}

        device = torch.device(configs.device)
        # Define dimensions from observation space
        self.pair_input_dim = observation_space["pair_features"].shape[-1]
        self.embedding_output_dim = kwargs.get("layer_fea_output_dim", [32, 8])[-1]

        # Initialize DualAttentionNetwork (feature extractor)
        self.features_extractor = DualAttentionNetwork_(observation_space, configs).to(device)
        self.pi_features_extractor = self.features_extractor
        self.vf_features_extractor = self.features_extractor
        # Actor and critic networks
        self.actor = Actor(configs.num_mlp_layers_actor, 4 * self.embedding_output_dim + self.pair_input_dim,
                           configs.hidden_dim_actor, 1).to(device)
        self.critic = Critic(configs.num_mlp_layers_critic, 2 * self.embedding_output_dim, configs.hidden_dim_critic,
                             1).to(device)

    def forward(self, observations, deterministic=False):
        """
        Forward pass to compute action probabilities and value estimates
        :param observations: Observations from the environment
        :param deterministic: Whether to return deterministic actions
        :return: actions, values, log_probs
        """
        # Extract features from observations
        comp_idx = observations['comp_idx']   # [sz_b, M, M, J]
        candidate_features, global_feature, combined_features = self.features_extractor(observations)
        sz_b, M, _, J = comp_idx.size()

        # Compute actor outputs
        
        candidate_scores = self.actor(candidate_features).squeeze(-1)  # [sz_b, J * M]
        dynamic_pair_mask = observations['dynamic_pair_mask'].reshape(sz_b, -1).bool()
        candidate_scores[dynamic_pair_mask] = float('-inf')
        pi = F.softmax(candidate_scores, dim=1)

        # Compute critic outputs
        v = self.critic(global_feature)

        # Select actions
        if deterministic:
            actions = torch.argmax(pi, dim=-1)  # [sz_b]
            log_probs = torch.log(pi.gather(1, actions.unsqueeze(-1)).squeeze(-1))  # [sz_b]
        else:
            actions, log_probs = sample_action(pi)

        # Compute log_probs
        return actions, v, log_probs

    def evaluate_actions(self, observations, actions):
        """
        Evaluate actions given observations
        :param observations: Observations from the environment
        :param actions: Actions to evaluate
        :return: Values, log probabilities, and entropy
        """

        candidate_features, global_feature, combined_features = self.features_extractor(observations)

        # Process features for the decision-making network
        comp_idx = observations['comp_idx']   # [sz_b, M, M, J]
        sz_b, M, _, J = comp_idx.size()

        # Compute actor outputs
        candidate_scores = self.actor(candidate_features).squeeze(-1)  # [sz_b, J * M]
        dynamic_pair_mask = observations['dynamic_pair_mask'].reshape(sz_b, -1).bool()
        candidate_scores[dynamic_pair_mask] = float('-inf')
        pi = F.softmax(candidate_scores, dim=1)  # [sz_b, J * M]

        # Compute critic outputs
        values = self.critic(global_feature)  # [sz_b, 1]

        # Compute log probabilities for the given actions
        log_probs = torch.log(pi.gather(1, actions.unsqueeze(-1)).squeeze(-1))  # [sz_b]

        # Compute entropy of the policy
        entropy = -(pi * torch.log(pi + 1e-8)).sum(dim=1)  # [sz_b]

        return values, log_probs, entropy

    def predict_values(self, observations):
        candidate_features, global_feature, combined_features = self.features_extractor(observations)
        v = self.critic(global_feature)
        return v
    
    def _predict(self, observations, deterministic=False):
        """
        Predict actions given observations
        :param observations: Observations from the environment
        :param deterministic: Whether to use deterministic actions
        :return: Predicted actions and state values
        """
        # Extract features from observations
        candidate_features, global_feature, combined_features = self.features_extractor(observations)
        sz_b, M, _, J = observations['comp_idx'].size()
        # Compute actor outputs
        candidate_scores = self.actor(candidate_features).squeeze(-1)  # [sz_b, J * M]

        dynamic_pair_mask = observations['dynamic_pair_mask'].reshape(sz_b, -1).bool()
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float('-inf')

        pi = F.softmax(candidate_scores, dim=1)

        # Select actions
        if deterministic:
            actions = torch.argmax(pi, dim=-1)  # [sz_b]
        else:
            actions = torch.multinomial(pi, num_samples=1).squeeze(-1)  # [sz_b]

        return actions

