from model.main_model import *
from common_utils import eval_actions
import torch.nn as nn
import torch
from copy import deepcopy
# from params import configs
import numpy as np
import learn2learn as l2l
from torch import autograd
from collections import OrderedDict

from torch.utils.tensorboard import SummaryWriter

class Memory:
    def __init__(self, gamma, gae_lambda):
        """
            the memory used for collect trajectories for PPO training
        :param gamma: discount factor
        :param gae_lambda: GAE parameter for PPO algorithm
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        # input variables of DANIEL
        self.fea_j_seq = []  # [N, tensor[sz_b, N, 8]]
        self.op_mask_seq = []  # [N, tensor[sz_b, N, 3]]
        self.fea_m_seq = []  # [N, tensor[sz_b, M, 6]]
        self.mch_mask_seq = []  # [N, tensor[sz_b, M, M]]
        self.dynamic_pair_mask_seq = []  # [N, tensor[sz_b, J, M]]
        self.comp_idx_seq = []  # [N, tensor[sz_b, M, M, J]]
        self.candidate_seq = []  # [N, tensor[sz_b, J]]
        self.fea_pairs_seq = []  # [N, tensor[sz_b, J]]

        # other variables
        self.action_seq = []  # action index with shape [N, tensor[sz_b]]
        self.reward_seq = []  # reward value with shape [N, tensor[sz_b]]
        self.val_seq = []  # state value with shape [N, tensor[sz_b]]
        self.done_seq = []  # done flag with shape [N, tensor[sz_b]]
        self.log_probs = []  # log(p_{\theta_old}(a_t|s_t)) with shape [N, tensor[sz_b]]

    def clear_memory(self):
        # print(f"清除了{len(self.action_seq)}个action！")
        self.clear_state()
        del self.action_seq[:]
        del self.reward_seq[:]
        del self.val_seq[:]
        del self.done_seq[:]
        del self.log_probs[:]

    def clear_state(self):
        del self.fea_j_seq[:]
        del self.op_mask_seq[:]
        del self.fea_m_seq[:]
        del self.mch_mask_seq[:]
        del self.dynamic_pair_mask_seq[:]
        del self.comp_idx_seq[:]
        del self.candidate_seq[:]
        del self.fea_pairs_seq[:]

    def push(self, state):
        """
            push a state into the memory 
            将一个状态数据（state）推入内存，
            存储环境状态的各种特征和信息。
        :param state: the MDP state
        :return:
        """
        # print(state.fea_j_tensor)
        # print(state.fea_j_tensor.shape)
        self.fea_j_seq.append(state.fea_j_tensor)
        self.op_mask_seq.append(state.op_mask_tensor)
        self.fea_m_seq.append(state.fea_m_tensor)
        self.mch_mask_seq.append(state.mch_mask_tensor)
        self.dynamic_pair_mask_seq.append(state.dynamic_pair_mask_tensor)
        self.comp_idx_seq.append(state.comp_idx_tensor)
        self.candidate_seq.append(state.candidate_tensor)
        self.fea_pairs_seq.append(state.fea_pairs_tensor)
    
    def get_batch(self, batch_size):
        # 假设所有数据都有相同的批次大小（sz_b）
        indices = torch.randperm(len(self.action_seq))[:batch_size]
        batch = {
            'actions': torch.cat([self.action_seq[idx] for idx in indices], dim=0),
            'rewards': torch.cat([self.reward_seq[idx] for idx in indices], dim=0),
            # 添加其他必要的字段
        }
        return batch

    def transpose_data(self):
        """
            将收集的数据从不同维度进行转置，以便后续处理和计算。例如，将状态序列中
            的时间步和环境序列进行转置，使得状态在时间步上连续，而环境在序列上连续。
            transpose the first and second dimension of collected variables
        """
        # 14
        t_Fea_j_seq = torch.stack(self.fea_j_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_op_mask_seq = torch.stack(self.op_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_Fea_m_seq = torch.stack(self.fea_m_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_mch_mask_seq = torch.stack(self.mch_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_dynamicMask_seq = torch.stack(self.dynamic_pair_mask_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_Compete_m_seq = torch.stack(self.comp_idx_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_candidate_seq = torch.stack(self.candidate_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_pairMessage_seq = torch.stack(self.fea_pairs_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_action_seq = torch.stack(self.action_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_reward_seq = torch.stack(self.reward_seq, dim=0).transpose(0, 1).flatten(0, 1)
        self.t_old_val_seq = torch.stack(self.val_seq, dim=0).transpose(0, 1)
        t_val_seq = self.t_old_val_seq.flatten(0, 1)
        t_done_seq = torch.stack(self.done_seq, dim=0).transpose(0, 1).flatten(0, 1)
        t_logprobs_seq = torch.stack(self.log_probs, dim=0).transpose(0, 1).flatten(0, 1)

        return t_Fea_j_seq, t_op_mask_seq, t_Fea_m_seq, t_mch_mask_seq, t_dynamicMask_seq, \
               t_Compete_m_seq, t_candidate_seq, t_pairMessage_seq, \
               t_action_seq, t_reward_seq, t_val_seq, t_done_seq, t_logprobs_seq

    def get_gae_advantages(self):
        """
        计算广义优势估计（Generalized Advantage Estimate, GAE）。
        这是计算优势估计的方法，用于评估策略相对于值函数的性能，从而指导策略的更新。
        方法中首先计算每个时间步的优势估计，然后进行转置和标准化，最终返回计算得到的
        优势估计和目标价值。
        Compute the generalized advantage estimates
        :return: advantage sequences, state value sequence
        """

        reward_arr = torch.stack(self.reward_seq, dim=0)
        values = self.t_old_val_seq.transpose(0, 1)
        len_trajectory, len_envs = reward_arr.shape

        advantage = torch.zeros(len_envs, device=values.device)
        advantage_seq = []
        for i in reversed(range(len_trajectory)):

            if i == len_trajectory - 1:
                delta_t = reward_arr[i] - values[i]
            else:
                delta_t = reward_arr[i] + self.gamma * values[i + 1] - values[i]
            advantage = delta_t + self.gamma * self.gae_lambda * advantage
            advantage_seq.insert(0, advantage)

        # [sz_b, N]
        t_advantage_seq = torch.stack(advantage_seq, dim=0).transpose(0, 1).to(torch.float32)

        # [sz_b, N]
        v_target_seq = (t_advantage_seq + self.t_old_val_seq).flatten(0, 1)

        # normalization
        t_advantage_seq = (t_advantage_seq - t_advantage_seq.mean(dim=1, keepdim=True)) \
                          / (t_advantage_seq.std(dim=1, keepdim=True) + 1e-8)

        return t_advantage_seq.flatten(0, 1), v_target_seq


class PPO:
    def __init__(self, config):
        """
            The implementation of PPO algorithm
        :param config: a package of parameters
        """
        self.lr = config.lr  # 学习率
        self.meta_lr = config.meta_lr
        self.gamma = config.gamma  # 折扣因子
        self.adapt_lr = config.adapt_lr
        self.gae_lambda = config.gae_lambda  # GAE广义优势估计参数
        self.eps_clip = config.eps_clip  # PPO算法中的剪切范围
        self.k_epochs = config.k_epochs  # PPO算法中的迭代次数
        self.tau = config.tau  # 软更新时的更新权重
        self.ppo_steps = 1

        self.ploss_coef = config.ploss_coef  # 策略损失系数
        self.vloss_coef = config.vloss_coef  # 价值损失系数
        self.entloss_coef = config.entloss_coef  # 交叉熵损失系数
        self.minibatch_size = config.minibatch_size  # 批次大小

        self.policy = DANIEL(config)
        self.policy_old = deepcopy(self.policy)

        self.policy_old.load_state_dict(self.policy.state_dict())
        # 创建优化器和值函数损失函数
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.meta_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.meta_lr)
        self.feature_exact_optimizer = torch.optim.Adam(self.policy.feature_exact.parameters(), lr=self.lr)
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=self.lr)

        self.V_loss_2 = nn.MSELoss()
        self.device = torch.device(config.device)

    def update(self, memory):
        '''
        :param memory: data used for PPO training
        :return: total_loss and critic_loss
        '''

        # 获取转置后的训练数据，用于策略更新
        t_data = memory.transpose_data()  # Tensor len 13  pre torch.Size([1000, 50, 10])
        # 计算广义优势估计（GAE）和目标价值  A_t, G_t
        t_advantage_seq, v_target_seq = memory.get_gae_advantages()

        full_batch_size = len(t_data[-1])  # 获取完整批次大小 # 1000
        num_batch = np.ceil(full_batch_size / self.minibatch_size)  # 计算小批次数 1.0

        loss_epochs = 0
        v_loss_epochs = 0

        for _ in range(self.k_epochs):  # 4
            # 对每个迭代进行小批次的策略更新
            # Split into multiple batches of updates due to memory limitations
            
            for i in range(int(num_batch)):
                if i + 1 < num_batch:
                    start_idx = i * self.minibatch_size
                    end_idx = (i + 1) * self.minibatch_size
                else:
                    # the last batch  处理最后一个小批次
                    start_idx = i * self.minibatch_size
                    end_idx = full_batch_size

                # 通过策略网络获取动作分布和值函数估计
                pis, vals = self.policy(fea_j=t_data[0][start_idx:end_idx],
                                        op_mask=t_data[1][start_idx:end_idx],
                                        candidate=t_data[6][start_idx:end_idx],
                                        fea_m=t_data[2][start_idx:end_idx],
                                        mch_mask=t_data[3][start_idx:end_idx],
                                        comp_idx=t_data[5][start_idx:end_idx],
                                        dynamic_pair_mask=t_data[4][start_idx:end_idx],
                                        fea_pairs=t_data[7][start_idx:end_idx])

                action_batch = t_data[8][start_idx: end_idx]  # 获取动作序列
                logprobs, ent_loss = eval_actions(pis, action_batch)  # 计算动作的概率和熵损失
                ratios = torch.exp(logprobs - t_data[12][start_idx: end_idx].detach())  # 计算重要性采样比率

                advantages = t_advantage_seq[start_idx: end_idx]  # 获取优势估计
                surr1 = ratios * advantages  # 计算第一个损失项
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages  # 计算第二个损失项

                v_loss = self.V_loss_2(vals.squeeze(1), v_target_seq[start_idx: end_idx])  # 计算价值损失
                p_loss = - torch.min(surr1, surr2)  # 计算策略损失   L^PPO-clip(pi_theta)
                ent_loss = - ent_loss.clone()  # 计算熵损失
                loss = self.vloss_coef * v_loss + self.ploss_coef * p_loss + self.entloss_coef * ent_loss  # 计算总损失
                # 梯度清零，进行反向传播和优化
                self.optimizer.zero_grad()  
                loss_epochs += loss.mean().detach()
                v_loss_epochs += v_loss.mean().detach()
                loss.mean().backward()
                self.optimizer.step()
        # soft update 进行软更新
        for policy_old_params, policy_params in zip(self.policy_old.parameters(), self.policy.parameters()):
            policy_old_params.data.copy_(self.tau * policy_old_params.data + (1 - self.tau) * policy_params.data)

        return loss_epochs.item() / self.k_epochs, v_loss_epochs.item() / self.k_epochs


    def inner_update(self, memory, num_steps=4, inner_lr=0.001, params=None):
        '''
        :param memory: data used for PPO training
        :param num_steps: number of gradient updates for inner loop
        :param inner_lr: learning rate for inner loop updates
        :return: Updated policy parameters after inner loop
        '''
        # 获取转置后的训练数据，用于策略更新
        t_data = memory.transpose_data()
        t_advantage_seq, v_target_seq = memory.get_gae_advantages()

        full_batch_size = len(t_data[-1])
        num_batch = np.ceil(full_batch_size / self.minibatch_size)
        if params is None:
            updated_params = {name: param.clone() for name, param in self.policy.named_parameters()}
        else:
            updated_params = {name: param.clone() for name, param in params.items()}
        
        for _ in range(num_steps):
            for i in range(int(num_batch)):
                start_idx = i * self.minibatch_size
                end_idx = min((i + 1) * self.minibatch_size, full_batch_size)

                pis, vals = self.policy(fea_j=t_data[0][start_idx:end_idx],
                                        op_mask=t_data[1][start_idx:end_idx],
                                        candidate=t_data[6][start_idx:end_idx],
                                        fea_m=t_data[2][start_idx:end_idx],
                                        mch_mask=t_data[3][start_idx:end_idx],
                                        comp_idx=t_data[5][start_idx:end_idx],
                                        dynamic_pair_mask=t_data[4][start_idx:end_idx],
                                        fea_pairs=t_data[7][start_idx:end_idx],
                                        params=updated_params)

                action_batch = t_data[8][start_idx: end_idx]
                logprobs, ent_loss = eval_actions(pis, action_batch)
                ratios = torch.exp(logprobs - t_data[12][start_idx: end_idx].detach())

                advantages = t_advantage_seq[start_idx: end_idx]
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                v_loss = self.V_loss_2(vals.squeeze(1), v_target_seq[start_idx: end_idx])
                p_loss = - torch.min(surr1, surr2)
                ent_loss = - ent_loss.clone()
                loss = self.vloss_coef * v_loss + self.ploss_coef * p_loss + self.entloss_coef * ent_loss
                
                # Compute gradients w.r.t. inner loop parameters
                grads = torch.autograd.grad(loss.mean(), updated_params.values(),  allow_unused=True)

                updated_params = {name: param - inner_lr * grad for (name, param), grad in zip(updated_params.items(), grads) if grad is not None}

        return updated_params


    def fast_adapt(self, memory: Memory, clone: DANIEL, freeze_feature_exact=True):
        '''
        memory: 
        clone: 
        '''
        if freeze_feature_exact:
            for name, param in self.policy.named_parameters():
                if name.startswith('feature_exact'):
                    param.requires_grad = False
        optimizer_clone = torch.optim.Adam(clone.actor.parameters(), lr=0.001) 
        # 获取转置后的训练数据，用于策略更新
        t_data = memory.transpose_data()  # Tensor len 13  pre torch.Size([1000, 50, 10])
        # 计算广义优势估计（GAE）和目标价值  A_t, G_t
        t_advantage_seq, v_target_seq = memory.get_gae_advantages()

        end_idx = len(t_data[-1])  # 获取完整批次大小 # 1000
        start_idx = 0
        # 通过策略网络获取动作分布和值函数估计
        pis, vals = clone(fea_j=t_data[0][start_idx:end_idx],
                            op_mask=t_data[1][start_idx:end_idx],
                            candidate=t_data[6][start_idx:end_idx],
                            fea_m=t_data[2][start_idx:end_idx],
                            mch_mask=t_data[3][start_idx:end_idx],
                            comp_idx=t_data[5][start_idx:end_idx],
                            dynamic_pair_mask=t_data[4][start_idx:end_idx],
                            fea_pairs=t_data[7][start_idx:end_idx])

        action_batch = t_data[8][start_idx: end_idx]  # 获取动作序列
        logprobs, ent_loss = eval_actions(pis, action_batch)  # 计算动作的概率和熵损失
        ratios = torch.exp(logprobs - t_data[12][start_idx: end_idx].detach())  # 计算重要性采样比率

        advantages = t_advantage_seq[start_idx: end_idx]  # 获取优势估计
        surr1 = ratios * advantages  # 计算第一个损失项
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages  # 计算第二个损失项

        v_loss = self.V_loss_2(vals.squeeze(1), v_target_seq[start_idx: end_idx])  # 计算价值损失
        p_loss = - torch.min(surr1, surr2)  # 计算策略损失   L^PPO-clip(pi_theta)
        ent_loss = - ent_loss.clone()  # 计算熵损失

        loss = self.vloss_coef * v_loss + self.ploss_coef * p_loss + self.entloss_coef * ent_loss  # 计算总损失
        # 梯度清零，进行反向传播和优化
        optimizer_clone.zero_grad()  
        loss_epochs = loss.mean().detach()
        v_loss_epochs = v_loss.mean().detach()
        loss.mean().backward()
        # gradients = autograd.grad(loss.mean(), clone.parameters())
        # 查看哪些参数受到loss的影响
        # for name, param in clone.named_parameters():
        #     if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
        #         print(name, "受到了loss的影响")
        #     else:
        #         print(name, "没有受到loss的影响")
        optimizer_clone.step()
        for policy_old_params, policy_params in zip(clone.parameters(), self.policy.parameters()):
            policy_old_params.data.copy_(self.tau * policy_old_params.data + (1 - self.tau) * policy_params.data)

        return loss_epochs.item(), v_loss_epochs.item() / self.k_epochs, clone


    def compute_loss(self, memory):
        '''
        :param memory: data used for PPO training
        :return: total_loss and critic_loss
        '''

        # 获取转置后的训练数据，用于策略更新
        t_data = memory.transpose_data()  
        # 计算广义优势估计（GAE）和目标价值  A_t, G_t
        t_advantage_seq, v_target_seq = memory.get_gae_advantages()

        full_batch_size = len(t_data[-1])  # 获取完整批次大小
        num_batch = np.ceil(full_batch_size / self.minibatch_size)  # 计算小批次数

        loss_epochs = 0
        v_loss_epochs = 0
        loss_list = []
        v_loss_list = []


        for i in range(int(num_batch)):
            if i + 1 < num_batch:
                start_idx = i * self.minibatch_size
                end_idx = (i + 1) * self.minibatch_size
            else:
                # the last batch  处理最后一个小批次
                start_idx = i * self.minibatch_size
                end_idx = full_batch_size

            # 通过策略网络获取动作分布和值函数估计
            pis, vals = self.policy(fea_j=t_data[0][start_idx:end_idx],
                                    op_mask=t_data[1][start_idx:end_idx],
                                    candidate=t_data[6][start_idx:end_idx],
                                    fea_m=t_data[2][start_idx:end_idx],
                                    mch_mask=t_data[3][start_idx:end_idx],
                                    comp_idx=t_data[5][start_idx:end_idx],
                                    dynamic_pair_mask=t_data[4][start_idx:end_idx],
                                    fea_pairs=t_data[7][start_idx:end_idx])

            action_batch = t_data[8][start_idx: end_idx]  # 获取动作序列
            logprobs, ent_loss = eval_actions(pis, action_batch)  # 计算动作的概率和熵损失
            ratios = torch.exp(logprobs - t_data[12][start_idx: end_idx].detach())  # 计算重要性采样比率

            advantages = t_advantage_seq[start_idx: end_idx]  # 获取优势估计
            surr1 = ratios * advantages  # 计算第一个损失项
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages  # 计算第二个损失项

            v_loss = self.V_loss_2(vals.squeeze(1), v_target_seq[start_idx: end_idx])  # 计算价值损失
            p_loss = - torch.min(surr1, surr2)  # 计算策略损失
            ent_loss = - ent_loss.clone()  # 计算熵损失
            loss = self.vloss_coef * v_loss + self.ploss_coef * p_loss + self.entloss_coef * ent_loss  # 计算总损失

            loss_epochs += loss.mean().detach()
            v_loss_epochs += v_loss.mean().detach()

            loss_list.append(loss.mean())
            v_loss_list.append(v_loss)
            
        total_loss = torch.stack(loss_list).mean()
        total_v_loss = torch.stack(v_loss_list).mean()
        return total_loss, total_v_loss



    def outer_ppo_loss(self, valid_replays, valid_advantage_seq, valid_v_target_seq, old_policy, new_policy):

        pis_old, vals_old = self.policy(fea_j=valid_replays[0][:],
                                    op_mask=valid_replays[1][:],
                                    candidate=valid_replays[6][:],
                                    fea_m=valid_replays[2][:],
                                    mch_mask=valid_replays[3][:],
                                    comp_idx=valid_replays[5][:],
                                    dynamic_pair_mask=valid_replays[4][:],
                                    fea_pairs=valid_replays[7][:],
                                    params=old_policy
                                    )

        action_batch = valid_replays[8]
        logprobs_old, ent_loss = eval_actions(pis_old, action_batch)

        pis, vals = self.policy(fea_j=valid_replays[0][:],
                                    op_mask=valid_replays[1][:],
                                    candidate=valid_replays[6][:],
                                    fea_m=valid_replays[2][:],
                                    mch_mask=valid_replays[3][:],
                                    comp_idx=valid_replays[5][:],
                                    dynamic_pair_mask=valid_replays[4][:],
                                    fea_pairs=valid_replays[7][:],
                                    params=new_policy
                                    )

        logprobs, ent_loss = eval_actions(pis, action_batch)
        #ratio
        ratio = torch.exp(logprobs - valid_replays[12].detach())  # 计算重要性采样比率
        # print("ratio:", ratio)
        # valid_advantage_seq = (valid_advantage_seq - valid_advantage_seq.mean()) / (valid_advantage_seq.std() + 1e-8)
        surr1 = ratio * valid_advantage_seq  # 计算第一个损失项

        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * valid_advantage_seq  # 计算第二个损失项
        v_loss = self.V_loss_2(vals.squeeze(1), valid_v_target_seq)  # 计算价值损失
        p_loss = - torch.min(surr1, surr2)  # 计算策略损失

        ent_loss = - ent_loss.clone()  # 计算熵损失
        loss = self.vloss_coef * v_loss + self.ploss_coef * p_loss + self.entloss_coef * ent_loss  # 计算总损失
        loss = loss / (valid_replays[0].shape[0] )
        # print(valid_replays[0].shape[0])
        return loss

    def meta_loss(self, iteration_replays, iteration_policies):
        '''
        iteration_replays: 任务回放
        iteration_policies: 每个任务对应的策略
        '''
        mean_loss = 0.00
        for _ in range(len(iteration_replays)):

            task_replays:Memory = iteration_replays[_]
            old_policy = iteration_policies[_]
            t_data = task_replays.transpose_data()
            t_advantage_seq, v_target_seq = task_replays.get_gae_advantages()
            # print(t_advantage_seq.shape)
            # 确定训练集和验证集的分割比例
            train_ratio = 0.9

            # 计算分割点
            # 假设所有张量的第一维长度是一样的，我们可以用任何一个张量来计算分割点
            split_idx = int(len(t_data[0]) * train_ratio)

            # 分割数据
            train_replays = tuple(data[:split_idx] for data in t_data)
            valid_replays = tuple(data[split_idx:] for data in t_data)
            train_advantage_seq = t_advantage_seq[:split_idx]
            valid_advantage_seq = t_advantage_seq[split_idx:]
            train_v_target_seq = v_target_seq[:split_idx]
            valid_v_target_seq = v_target_seq[split_idx:]
            
            full_batch_size = len(train_replays[-1])
            num_batch = np.ceil(full_batch_size / self.minibatch_size)
            # print(" len(train_replays[0])", len(train_replays[0]))

            #fast_adapt
            # cloned_model = l2l.clone_module(self.policy)
            updated_params = dict(self.policy.named_parameters())
            # with torch.no_grad():
            new_policy = {name: param.clone() for name, param in self.policy.named_parameters() if param.requires_grad}

            for i in range(int(num_batch)):
                start_idx = i * self.minibatch_size
                end_idx = min((i + 1) * self.minibatch_size, full_batch_size)
                pis, vals = self.policy(fea_j=train_replays[0][start_idx:end_idx],
                        op_mask=train_replays[1][start_idx:end_idx],
                        candidate=train_replays[6][start_idx:end_idx],
                        fea_m=train_replays[2][start_idx:end_idx],
                        mch_mask=train_replays[3][start_idx:end_idx],
                        comp_idx=train_replays[5][start_idx:end_idx],
                        dynamic_pair_mask=train_replays[4][start_idx:end_idx],
                        fea_pairs=train_replays[7][start_idx:end_idx],
                        params=new_policy)

                action_batch = train_replays[8][start_idx:end_idx]
                logprobs, ent_loss = eval_actions(pis, action_batch)
                ratios = torch.exp(logprobs - train_replays[12][start_idx:end_idx].detach())  # 计算重要性采样比率
                advantages = train_advantage_seq[start_idx: end_idx]
                surr1 = ratios * advantages  # 计算第一个损失项
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages  # 计算第二个损失项
                v_loss = self.V_loss_2(vals.squeeze(1), train_v_target_seq[start_idx:end_idx])  # 计算价值损失
                p_loss = - torch.min(surr1, surr2)  # 计算策略损失
                ent_loss = - ent_loss.clone()  # 计算熵损失
            
                loss = self.vloss_coef * v_loss + self.ploss_coef * p_loss + self.entloss_coef * ent_loss  # 计算总损失
                # print(loss)
                grads = torch.autograd.grad(loss.mean(), new_policy.values(),  allow_unused=True)

                new_policy = {name: param - self.adapt_lr * grad if grad is not None else param for (name, param), grad in zip(new_policy.items(), grads)}
                # print(new_policy)
            
            loss = self.outer_ppo_loss(valid_replays, valid_advantage_seq, valid_v_target_seq, old_policy, new_policy)
            mean_loss += loss.mean()
        
        mean_loss /= len(iteration_replays)

        return mean_loss

    def meta_optimize(self, iteration_replays, iteration_policies):
        for _ in range(self.ppo_steps):
            loss = self.meta_loss(iteration_replays, iteration_policies)
            loss.backward()
            self.meta_optimizer.step()
            self.meta_optimizer.zero_grad()

            for policy_old_params, policy_params in zip(self.policy_old.parameters(), self.policy.parameters()):
                policy_old_params.data.copy_(self.tau * policy_old_params.data + (1 - self.tau) * policy_params.data)

        return loss

def PPO_initialize(configs):
    ppo = PPO(configs)
    
    # writer = SummaryWriter(log_dir=configs.logdir, flush_secs=180)

    # writer.add_graph(dict(ppo.policy.named_parameters()))
    # writer.close()
    return ppo
