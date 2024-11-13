from common_utils import nonzero_averaging, get_subdict
from model.attention_layer import *
from model.sub_layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualAttentionNetwork(nn.Module):
    def __init__(self, config):
        """
            The implementation of dual attention network (DAN)
        :param config: a package of parameters
        """
        super(DualAttentionNetwork, self).__init__()

        self.fea_j_input_dim = config.fea_j_input_dim
        self.fea_m_input_dim = config.fea_m_input_dim
        self.output_dim_per_layer = config.layer_fea_output_dim
        self.num_heads_OAB = config.num_heads_OAB
        self.num_heads_MAB = config.num_heads_MAB
        self.last_layer_activate = nn.ELU()

        self.num_dan_layers = len(self.num_heads_OAB)
        assert len(config.num_heads_MAB) == self.num_dan_layers
        assert len(self.output_dim_per_layer) == self.num_dan_layers
        self.alpha = 0.2
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout_prob = config.dropout_prob

        num_heads_OAB_per_layer = [1] + self.num_heads_OAB
        num_heads_MAB_per_layer = [1] + self.num_heads_MAB

        # mid_dim = [self.embedding_output_dim] * (self.num_dan_layers - 1)
        mid_dim = self.output_dim_per_layer[:-1]

        j_input_dim_per_layer = [self.fea_j_input_dim] + mid_dim

        m_input_dim_per_layer = [self.fea_m_input_dim] + mid_dim

        self.op_attention_blocks = torch.nn.ModuleList()
        self.mch_attention_blocks = torch.nn.ModuleList()

        for i in range(self.num_dan_layers):
            self.op_attention_blocks.append(
                MultiHeadOpAttnBlock(
                    input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_OAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )

        for i in range(self.num_dan_layers):
            self.mch_attention_blocks.append(
                MultiHeadMchAttnBlock(
                    node_input_dim=num_heads_MAB_per_layer[i] * m_input_dim_per_layer[i],
                    edge_input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_MAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, fast_weights=None):
        """
        :param candidate: the index of candidates  [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :return:
            fea_j.shape = [sz_b, N, output_dim]
            fea_m.shape = [sz_b, M, output_dim]
            fea_j_global.shape = [sz_b, output_dim]
            fea_m_global.shape = [sz_b, output_dim]
        """
        sz_b, M, _, J = comp_idx.size()

        comp_idx_for_mul = comp_idx.reshape(sz_b, -1, J)

        for layer in range(self.num_dan_layers):
            candidate_idx = candidate.unsqueeze(-1). \
                repeat(1, 1, fea_j.shape[-1]).type(torch.int64)

            # fea_j_jc: candidate features with shape [sz_b, N, J]
            fea_j_jc = torch.gather(fea_j, 1, candidate_idx).type(torch.float32)
            comp_val_layer = torch.matmul(comp_idx_for_mul,
                                     fea_j_jc).reshape(sz_b, M, M, -1)
            fea_j = self.op_attention_blocks[layer](fea_j, op_mask, get_subdict(fast_weights, f"op_attention_blocks.{layer}"))
            fea_m = self.mch_attention_blocks[layer](fea_m, mch_mask, comp_val_layer, get_subdict(fast_weights, f"mch_attention_blocks.{layer}"))

        fea_j_global = nonzero_averaging(fea_j)
        fea_m_global = nonzero_averaging(fea_m)

        return fea_j, fea_m, fea_j_global, fea_m_global


class DANIEL(nn.Module):
    def __init__(self, config):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(DANIEL, self).__init__()
        device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        self.feature_exact = DualAttentionNetwork(config).to(
            device)
        
        self.actor = Actor(config.num_mlp_layers_actor, 4 * self.embedding_output_dim + self.pair_input_dim,
                           config.hidden_dim_actor, 1).to(device)

        self.critic = Critic(config.num_mlp_layers_critic, 2 * self.embedding_output_dim, config.hidden_dim_critic,
                             1).to(device)


    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs, params = None):
        """
            :param candidate: 候选操作的索引，形状为[sz_b, J]
            :param fea_j: 输入操作特征向量，形状为[sz_b, N, 8]
            :param op_mask: 用于屏蔽不存在的前驱/后继操作，形状为[sz_b, N, 3]
            :param fea_m: 输入操作特征向量，形状为[sz_b, M, 6]
            :param mch_mask: 用于屏蔽注意力系数的掩码，形状为[sz_b, M, M]
            :param comp_idx: 形状为[sz_b, M, M, J]的张量，用于计算 T_E。
            comp_idx[i, k, q, j] (对于任意 i) 的值表示机器 M_k 和 M_q 是否竞争候选操作[i,j]
            :param dynamic_pair_mask: 形状为[sz_b, J, M]的张量，用于屏蔽不兼容的操作-机器对
            :param fea_pairs: 包含对特征的形状为[sz_b, J, M, 8]的张量
            :return:
            pi: 调度策略，形状为[sz_b, J*M]
            v: 状态值，形状为[sz_b, 1]
        """
        if params is None:
            params = {}
            for name, param in self.named_parameters():
                params[name] = param

        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(fea_j, op_mask, candidate, fea_m, mch_mask,
                                                                      comp_idx, fast_weights=self.get_subdict(params, 'feature_exact')) # 调用self.feature_exact函数，计算操作特征向量的一些中间结果。

        sz_b, M, _, J = comp_idx.size()  # 获取comp_idx张量的形状信息。
        d = fea_j.size(-1)

        # collect the input of decision-making network
        # 根据候选操作的索引，生成一个新的张量candidate_idx，并重复它以匹配操作特征向量的维度。
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)
        
        # 对操作特征向量进行索引操作，得到候选操作的特征向量。
        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        # 对Fea_j_JC和fea_m进行扩展和重塑操作，使得它们的形状匹配后续计算的需求。
        Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        # 将全局操作特征向量扩展成与Fea_j_JC_serialized相同的形状
        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        
        # 重塑fea_pairs张量的形状以匹配接下来拼接操作的需求。
        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        # candidate_feature.shape = [sz_b, J*M, 4*output_dim + 8]
        # 将多个特征张量按照最后一个维度拼接在一起，得到候选操作的特征张量。
        candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, Fea_Gj_input,
                                       Fea_Gm_input, fea_pairs), dim=-1)
        
        # h0 = torch.zeros(1, candidate_feature.size(0), 64).to(candidate_feature.device)
        candidate_scores = self.actor(candidate_feature, params=self.get_subdict(params, 'actor')) # 20, 50, 1
        # print(candidate_scores[0])
        # candidate_scores, h0 = self.actor(candidate_feature, h0)
        candidate_scores = candidate_scores.squeeze(-1) 

        # masking incompatible op-mch pairs
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float('-inf')
        pi = F.softmax(candidate_scores, dim=1)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature, params=self.get_subdict(params, 'critic'))  # 20,1 
        return pi, v

    def get_subdict(self, params_dict, prefix):
        """
        获取params_dict中以prefix为前缀的子字典
        """
        subdict = {k[len(prefix) + 1:]: v for k, v in params_dict.items() if k.startswith(prefix)}
        return subdict

    def get_named_parameters(self):
        for name, param in self.named_parameters():
            print(name, param.size())


class MetaActorDANIEL(nn.Module):
    def __init__(self, config, actor):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(DANIEL, self).__init__()
        device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        self.feature_exact = DualAttentionNetwork(config).to(
            device)
        
        self.actor = actor.to(device)

        # self.actor = ActorRNN(config.num_mlp_layers_actor, 4 * self.embedding_output_dim + self.pair_input_dim,
        #             config.hidden_dim_actor, 1).to(device)
        self.critic = Critic(config.num_mlp_layers_critic, 2 * self.embedding_output_dim, config.hidden_dim_critic,
                             1).to(device)
        # self.critic = CriticRNN(config.num_mlp_layers_critic, 2 * self.embedding_output_dim, config.hidden_dim_critic,
        #                      1).to(device)

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs):
        """
            :param candidate: 候选操作的索引，形状为[sz_b, J]
            :param fea_j: 输入操作特征向量，形状为[sz_b, N, 8]
            :param op_mask: 用于屏蔽不存在的前驱/后继操作，形状为[sz_b, N, 3]
            :param fea_m: 输入操作特征向量，形状为[sz_b, M, 6]
            :param mch_mask: 用于屏蔽注意力系数的掩码，形状为[sz_b, M, M]
            :param comp_idx: 形状为[sz_b, M, M, J]的张量，用于计算 T_E。
            comp_idx[i, k, q, j] (对于任意 i) 的值表示机器 M_k 和 M_q 是否竞争候选操作[i,j]
            :param dynamic_pair_mask: 形状为[sz_b, J, M]的张量，用于屏蔽不兼容的操作-机器对
            :param fea_pairs: 包含对特征的形状为[sz_b, J, M, 8]的张量
            :return:
            pi: 调度策略，形状为[sz_b, J*M]
            v: 状态值，形状为[sz_b, 1]
        """

        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(fea_j, op_mask, candidate, fea_m, mch_mask,
                                                                      comp_idx, ) # 调用self.feature_exact函数，计算操作特征向量的一些中间结果。

        sz_b, M, _, J = comp_idx.size()  # 获取comp_idx张量的形状信息。
        d = fea_j.size(-1)

        # collect the input of decision-making network
        # 根据候选操作的索引，生成一个新的张量candidate_idx，并重复它以匹配操作特征向量的维度。
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)
        
        # 对操作特征向量进行索引操作，得到候选操作的特征向量。
        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        # 对Fea_j_JC和fea_m进行扩展和重塑操作，使得它们的形状匹配后续计算的需求。
        Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        # 将全局操作特征向量扩展成与Fea_j_JC_serialized相同的形状
        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        
        # 重塑fea_pairs张量的形状以匹配接下来拼接操作的需求。
        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        # candidate_feature.shape = [sz_b, J*M, 4*output_dim + 8]
        # 将多个特征张量按照最后一个维度拼接在一起，得到候选操作的特征张量。
        candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, Fea_Gj_input,
                                       Fea_Gm_input, fea_pairs), dim=-1)
        
        # h0 = torch.zeros(1, candidate_feature.size(0), 64).to(candidate_feature.device)
        candidate_scores = self.actor(candidate_feature) # 20, 50, 1
        # candidate_scores, h0 = self.actor(candidate_feature, h0)
        candidate_scores = candidate_scores.squeeze(-1) 

        # masking incompatible op-mch pairs
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float('-inf')
        pi = F.softmax(candidate_scores, dim=1)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        v = self.critic(global_feature)  # 20,1 
        return pi, v

    def get_named_parameters(self):
        for name, param in self.named_parameters():
            print(name, param.size())


