from dataclasses import dataclass
import numpy as np
import numpy.ma as ma
import copy
from params import configs
import sys
import torch
import pandas as pd
import pygame
import random
import plotly.express as px
# 图形工厂
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.colors import qualitative

@dataclass
class EnvState:
    """
        state definition
    """
    fea_j_tensor: torch.Tensor = None  # 表示操作特征向量
    op_mask_tensor: torch.Tensor = None  # 操作掩码
    fea_m_tensor: torch.Tensor = None  # 机器的特征向量
    mch_mask_tensor: torch.Tensor = None  # 机器之间的注意力系数掩码
    dynamic_pair_mask_tensor: torch.Tensor = None  # 动态操作-机器对掩码 掩盖不兼容的操作-机器对
    comp_idx_tensor: torch.Tensor = None  # 
    candidate_tensor: torch.Tensor = None  # 候选操作的索引
    fea_pairs_tensor: torch.Tensor = None  # 操作-机器对的特征向量

    device = torch.device(configs.device)

    def update(self, fea_j, op_mask, fea_m, mch_mask, dynamic_pair_mask,
               comp_idx, candidate, fea_pairs):
        """
            update the state information
        :param fea_j: input operation feature vectors with shape [sz_b, N, 10]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 8]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :return:
        """
        device = self.device
        self.fea_j_tensor = torch.from_numpy(np.copy(fea_j)).float().to(device)
        self.fea_m_tensor = torch.from_numpy(np.copy(fea_m)).float().to(device)
        self.fea_pairs_tensor = torch.from_numpy(np.copy(fea_pairs)).float().to(device)

        self.op_mask_tensor = torch.from_numpy(np.copy(op_mask)).to(device)
        self.candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        self.mch_mask_tensor = torch.from_numpy(np.copy(mch_mask)).float().to(device)
        self.comp_idx_tensor = torch.from_numpy(np.copy(comp_idx)).to(device)
        self.dynamic_pair_mask_tensor = torch.from_numpy(np.copy(dynamic_pair_mask)).to(device)

    def print_shape(self):
        print(self.fea_j_tensor.shape)
        print(self.op_mask_tensor.shape)
        print(self.candidate_tensor.shape)
        print(self.fea_m_tensor.shape)
        print(self.mch_mask_tensor.shape)
        print(self.comp_idx_tensor.shape)
        print(self.dynamic_pair_mask_tensor.shape)
        print(self.fea_pairs_tensor.shape)

    def print_first(self):
        """
        返回对象的字符串表示，用于调试和日志记录。
        """
        return (f"EnvState(\n"
                f"  fea_j_tensor 形状: {self.fea_j_tensor[0]},\n"
                f"  op_mask_tensor 形状: {self.op_mask_tensor[0]},\n"
                f"  candidate_tensor 形状: {self.candidate_tensor[0]},\n"
                f"  fea_m_tensor 形状: {self.fea_m_tensor[0]},\n"
                f"  mch_mask_tensor 形状: {self.mch_mask_tensor[0]},\n"
                f"  comp_idx_tensor 形状: {self.comp_idx_tensor[0]},\n"
                f"  dynamic_pair_mask_tensor 形状: {self.dynamic_pair_mask_tensor[0]},\n"
                f"  fea_pairs_tensor 形状: {self.fea_pairs_tensor[0]}\n"
                f")")


@dataclass
class EnvECState:
    """
        state definition
    """
    fea_j_tensor: torch.Tensor = None  # 表示操作特征向量
    op_mask_tensor: torch.Tensor = None  # 操作掩码
    fea_m_tensor: torch.Tensor = None  # 机器的特征向量
    mch_mask_tensor: torch.Tensor = None  # 机器之间的注意力系数掩码
    dynamic_pair_mask_tensor: torch.Tensor = None  # 动态操作-机器对掩码 掩盖不兼容的操作-机器对
    comp_idx_tensor: torch.Tensor = None  # 
    candidate_tensor: torch.Tensor = None  # 候选操作的索引
    fea_pairs_tensor: torch.Tensor = None  # 操作-机器对的特征向量

    device = torch.device(configs.device)

    def update(self, fea_j, op_mask, fea_m, mch_mask, dynamic_pair_mask,
               comp_idx, candidate, fea_pairs):
        """
            update the state information
        :param fea_j: input operation feature vectors with shape [sz_b, N, 10]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 8]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :return:
        """
        device = self.device
        self.fea_j_tensor = torch.from_numpy(np.copy(fea_j)).float().to(device)
        self.fea_m_tensor = torch.from_numpy(np.copy(fea_m)).float().to(device)
        self.fea_pairs_tensor = torch.from_numpy(np.copy(fea_pairs)).float().to(device)

        self.op_mask_tensor = torch.from_numpy(np.copy(op_mask)).to(device)
        self.candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
        self.mch_mask_tensor = torch.from_numpy(np.copy(mch_mask)).float().to(device)
        self.comp_idx_tensor = torch.from_numpy(np.copy(comp_idx)).to(device)
        self.dynamic_pair_mask_tensor = torch.from_numpy(np.copy(dynamic_pair_mask)).to(device)

    def print_shape(self):
        print(self.fea_j_tensor.shape)
        print(self.op_mask_tensor.shape)
        print(self.candidate_tensor.shape)
        print(self.fea_m_tensor.shape)
        print(self.mch_mask_tensor.shape)
        print(self.comp_idx_tensor.shape)
        print(self.dynamic_pair_mask_tensor.shape)
        print(self.fea_pairs_tensor.shape)

    def print_first(self):
        """
        返回对象的字符串表示，用于调试和日志记录。
        """
        return (f"EnvState(\n"
                f"  fea_j_tensor 形状: {self.fea_j_tensor[0]},\n"
                f"  op_mask_tensor 形状: {self.op_mask_tensor[0]},\n"
                f"  candidate_tensor 形状: {self.candidate_tensor[0]},\n"
                f"  fea_m_tensor 形状: {self.fea_m_tensor[0]},\n"
                f"  mch_mask_tensor 形状: {self.mch_mask_tensor[0]},\n"
                f"  comp_idx_tensor 形状: {self.comp_idx_tensor[0]},\n"
                f"  dynamic_pair_mask_tensor 形状: {self.dynamic_pair_mask_tensor[0]},\n"
                f"  fea_pairs_tensor 形状: {self.fea_pairs_tensor[0]}\n"
                f")")


class FJSPEnvForSameOpNums:
    """
        a batch of fjsp environments that have the same number of operations

        let E/N/J/M denote the number of envs/operations/jobs/machines
        Remark: The index of operations has been rearranged in natural order
        eg. {O_{11}, O_{12}, O_{13}, O_{21}, O_{22}}  <--> {0,1,2,3,4}

        Attributes:

        job_length: the number of operations in each job (shape [J])
        op_pt: the processing time matrix with shape [N, M],
                where op_pt[i,j] is the processing time of the ith operation
                on the jth machine or 0 if $O_i$ can not process on $M_j$

        candidate: the index of candidates  [sz_b, J]
        fea_j: input operation feature vectors with shape [sz_b, N, 8]
        op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        fea_m: input operation feature vectors with shape [sz_b, M, 6]
        mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking incompatible op-mch pairs
        fea_pairs: pair features with shape [sz_b, J, M, 8]
    """

    def __init__(self, n_j, n_m):
        """
        :param n_j: the number of jobs
        :param n_m: the number of machines
        """
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.old_state = EnvState()

        # the dimension of operation raw features
        self.op_fea_dim = 10
        # the dimension of machine raw features
        self.mch_fea_dim = 8
        self.fig = go.Figure()

    def set_static_properties(self):
        """
            define static properties
        """
        self.multi_env_mch_diag = np.tile(np.expand_dims(np.eye(self.number_of_machines, dtype=bool), axis=0),
                                          (self.number_of_envs, 1, 1))
        '''这是一个三维布尔数组，用于表示多个环境中的机器对角线（diagonal）上的元素。它的维度是 (self.number_of_envs,
          self.number_of_machines, self.number_of_machines)，每个环境中的机器对角线上的元素都为 True，而其他位置为 False。
          这可能用于标记每个环境中的机器之间的兼容关系。'''
        self.env_idxs = np.arange(self.number_of_envs)
        # 这是一个包含从0到self.number_of_envs-1的整数的一维数组，表示环境的索引。

        self.env_job_idx = self.env_idxs.repeat(self.number_of_jobs).reshape(self.number_of_envs, self.number_of_jobs)
        '''这是一个二维整数数组，其维度为 (self.number_of_envs, self.number_of_jobs)，用于表示每个环境中每个作业的索引。
        该数组的每一行都是一个环境中的作业索引，行数代表环境的数量。'''

        self.op_idx = np.arange(self.number_of_ops)[np.newaxis, :]
        # 这是一个一维整数数组，包含从0到self.number_of_ops-1的整数，表示操作的索引。

    def set_initial_data(self, job_length_list, op_pt_list, mch_working_power_list=None, mch_idle_power_list=None):
        """
            initialize the data of the instances

        :param job_length_list: the list of 'job_length'
        :param op_pt_list: the list of 'op_pt'
        """

        self.number_of_envs = len(job_length_list)  # 测试用例环境的数量 100
        # print(self.number_of_envs)
        self.job_length = np.array(job_length_list)  # 测试的
        self.op_pt = np.array(op_pt_list)
        self.number_of_ops = self.op_pt.shape[1] # 每个环境中的操作数量
        self.number_of_machines = op_pt_list[0].shape[1] # 每个环境中的机器数量
        self.number_of_jobs = job_length_list[0].shape[0]  # 每个环境中的作业数量

        self.set_static_properties()

        # [E, N, M]
        # 操作加工时间的下限和上限
        self.pt_lower_bound = np.min(self.op_pt)
        self.pt_upper_bound = np.max(self.op_pt)
        self.true_op_pt = np.copy(self.op_pt) # 操作的真实加工时间

        # normalize the processing time
        self.op_pt = (self.op_pt - self.pt_lower_bound) / (self.pt_upper_bound - self.pt_lower_bound + 1e-8)

        # bool 3-d array formulating the compatible relation with shape [E,N,M]
        self.process_relation = (self.op_pt != 0)  # 一个布尔三维数组，表示操作之间的兼容关系，True 表示兼容。
        self.reverse_process_relation = ~self.process_relation  # 

        # number of compatible machines of each operation ([E,N])
        self.compatible_op = np.sum(self.process_relation, 2) # 每个操作可以在多少台机器上执行
        # number of operations that each machine can process ([E,M])
        self.compatible_mch = np.sum(self.process_relation, 1) # 每台机器可以处理多少个操作

        self.unmasked_op_pt = np.copy(self.op_pt) 

        head_op_id = np.zeros((self.number_of_envs, 1))

        # the index of first operation of each job ([E,J]) 每个作业的第一个和最后一个操作的索引
        self.job_first_op_id = np.concatenate([head_op_id, np.cumsum(self.job_length, axis=1)[:, :-1]], axis=1).astype(
            'int')
        # the index of last operation of each job ([E,J])
        self.job_last_op_id = self.job_first_op_id + self.job_length - 1

        self.initial_vars()

        self.init_op_mask()

        self.op_pt = ma.array(self.op_pt, mask=self.reverse_process_relation)

        """
            compute operation raw features
        """
        self.op_mean_pt = np.mean(self.op_pt, axis=2).data
        # 操作的最小和最大加工时间
        self.op_min_pt = np.min(self.op_pt, axis=-1).data  # 每个工序的最小完工时间
        self.op_max_pt = np.max(self.op_pt, axis=-1).data
        self.pt_span = self.op_max_pt - self.op_min_pt  # 操作加工时间范围
        # [E, M]机器的最小和最大加工时间
        self.mch_min_pt = np.min(self.op_pt, axis=1).data
        self.mch_max_pt = np.max(self.op_pt, axis=1)
        # the estimated lower bound of complete time of operations 操作的完工时间的估计下限
        self.op_ct_lb = copy.deepcopy(self.op_min_pt)

        for k in range(self.number_of_envs):
            for i in range(self.number_of_jobs):
                self.op_ct_lb[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1] = np.cumsum(
                    self.op_ct_lb[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1])

        # job remaining number of operations 每个操作剩余的作业数量
        self.op_match_job_left_op_nums = np.array([np.repeat(self.job_length[k],
                                                             repeats=self.job_length[k])
                                                   for k in range(self.number_of_envs)])
        self.job_remain_work = []
        for k in range(self.number_of_envs):
            self.job_remain_work.append(
                [np.sum(self.op_mean_pt[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1])
                 for i in range(self.number_of_jobs)])
        # 每个操作匹配的作业剩余工作量
        self.op_match_job_remain_work = np.array([np.repeat(self.job_remain_work[k], repeats=self.job_length[k])
                                                  for k in range(self.number_of_envs)])

        

        # shape reward
        self.init_quality = np.max(self.op_ct_lb, axis=1)

        self.max_endTime = self.init_quality
        
        
        """
            compute machine raw features
        """
        self.mch_available_op_nums = np.copy(self.compatible_mch)
        self.mch_current_available_op_nums = np.copy(self.compatible_mch)
        # [E, J, M] 候选操作的加工时间
        self.candidate_pt = np.array([self.unmasked_op_pt[k][self.candidate[k]] for k in range(self.number_of_envs)])

        # construct dynamic pair mask : [E, J, M]
        self.dynamic_pair_mask = (self.candidate_pt == 0)  # 动态操作-机器对掩码
        self.candidate_process_relation = np.copy(self.dynamic_pair_mask) # 候选操作和机器之间的处理关系
        self.mch_current_available_jc_nums = np.sum(~self.candidate_process_relation, axis=1) # 机器上当前可用的作业数量
        
        self.op_pt_current = self.unmasked_op_pt.copy()


        self.mch_mean_pt = np.mean(self.op_pt, axis=1).filled(0) # 
        ##生成机器的能耗   
        if mch_working_power_list is not None:
            self.mch_working_power = np.array(mch_working_power_list)
        else: self.mch_working_power = np.random.uniform(0.3, 1, size=self.mch_mean_pt.shape)
        # 随机生成每个环境中每台机器的待机功率（范围在0.1到0.2之间）
        if mch_idle_power_list is not None:
            self.mch_idle_power = np.array(mch_idle_power_list)
        else: self.mch_idle_power = np.random.uniform(0.1, 0.2, size=self.mch_mean_pt.shape)
        self.true_mch_working_power = np.copy(self.mch_working_power)
        self.wkpower_lb = 0.3
        self.wkpower_ub = 2.0
        self.mch_working_power = (self.mch_working_power - self.wkpower_lb) / (self.wkpower_ub - self.wkpower_lb + 1e-8)
        # 方法1：计算平均功耗

        self.mch_mean_pt_current = np.mean(self.op_pt_current, axis=1)
        mch_power_consumption = self.mch_mean_pt_current * self.mch_working_power
        working_energy = np.sum(mch_power_consumption, axis=1, keepdims=True)
        env_idle_time = np.sum(self.mch_idle_power * self.max_endTime[:, np.newaxis], axis=1)

        self.current_EC = working_energy + env_idle_time
        
        # 方法2： 计算最低功耗
        # op_energy = np.where(self.op_pt > 1e10, float(np.inf), self.op_pt * self.mch_working_power[:, np.newaxis, :])
        self.op_energy = self.op_pt * self.mch_working_power[:, np.newaxis, :]
        self.unmasked_op_energy = self.op_energy.data.copy()

        self.candidate_ec = np.array([self.unmasked_op_energy[k][self.candidate[k]] for k in range(self.number_of_envs)])

        min_energy_per_op = np.array(np.min(self.op_energy, axis=-1))
        self.total_min_energy = np.sum(min_energy_per_op, axis=-1)
        # self.current_EC = np.array(min_energy_per_mch)
        # self.current_EC = np.sum(self.current_EC, axis=-1)
        # self.current_EC = np.full(self.number_of_envs, float("-inf")) # 
        self.current_EC = self.total_min_energy
        # construct machine features [E, M, 6]
        self.energy_lb_idx = np.argmin(self.op_energy, axis=-1)
        self.energy_lb_energy = np.min(self.op_energy, axis=-1)

        self.mch_min_ec = np.min(self.op_energy, axis=-2).data
        self.mch_max_ec = np.max(self.op_energy, axis=-2).data
        self.mch_mean_ec = np.mean(self.op_energy, axis=-2).data
        self.op_energy_lb = np.min(self.op_energy, axis=-1).data
        self.op_min_energy = np.min(self.op_energy, axis=-1).data
        self.op_max_energy = np.max(self.op_energy, axis=-1).data
        self.energy_span = self.op_max_energy - self.op_min_energy
        self.op_mean_ec = np.mean(self.op_energy, axis=2).data
        self.construct_op_features()

        # construct 'come_idx' : [E, M, M, J]
        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)
        self.init_mch_mask()
        self.construct_mch_features()

        self.construct_pair_features()

        self.old_state.update(self.fea_j, self.op_mask,
                              self.fea_m, self.mch_mask,
                              self.dynamic_pair_mask, self.comp_idx, self.candidate,
                              self.fea_pairs)

        # old record
        self.old_op_mask = np.copy(self.op_mask)
        self.old_mch_mask = np.copy(self.mch_mask)
        self.old_op_ct_lb = np.copy(self.op_ct_lb)
        self.old_op_energy_lb = np.copy(self.op_energy_lb)
        self.old_op_match_job_left_op_nums = np.copy(self.op_match_job_left_op_nums)
        self.old_op_match_job_remain_work = np.copy(self.op_match_job_remain_work)
        self.old_init_quality = np.copy(self.init_quality)
        self.old_candidate_pt = np.copy(self.candidate_pt)
        self.old_candidate_ec= np.copy(self.candidate_ec)
        self.old_candidate_process_relation = np.copy(self.candidate_process_relation)
        self.old_mch_current_available_op_nums = np.copy(self.mch_current_available_op_nums)
        self.old_mch_current_available_jc_nums = np.copy(self.mch_current_available_jc_nums)
        self.old_current_EC = np.copy(self.current_EC)
        self.old_energy_lb_energy = np.copy(self.energy_lb_energy)
        self.old_energy_lb_idx = np.copy(self.energy_lb_idx)
        # state
        self.state = copy.deepcopy(self.old_state)
        return self.state

    def reset(self):
        """
           reset the environments
        :return: the state
        """
        self.initial_vars()

        # copy the old data
        self.op_mask = np.copy(self.old_op_mask)
        self.mch_mask = np.copy(self.old_mch_mask)
        self.op_ct_lb = np.copy(self.old_op_ct_lb)
        self.op_energy_lb = np.copy(self.old_op_energy_lb)
        self.op_match_job_left_op_nums = np.copy(self.old_op_match_job_left_op_nums)
        self.op_match_job_remain_work = np.copy(self.old_op_match_job_remain_work)
        self.init_quality = np.copy(self.old_init_quality)
        self.max_endTime = self.init_quality
        self.current_EC = np.copy(self.old_current_EC)
        self.candidate_pt = np.copy(self.old_candidate_pt)
        self.candidate_ec = np.copy(self.old_candidate_ec)
        self.candidate_process_relation = np.copy(self.old_candidate_process_relation)
        self.mch_current_available_op_nums = np.copy(self.old_mch_current_available_op_nums)
        self.mch_current_available_jc_nums = np.copy(self.old_mch_current_available_jc_nums)
        self.energy_lb_energy = np.copy(self.old_energy_lb_energy)
        self.energy_lb_idx = np.copy(self.old_energy_lb_idx)
        self.op_energy = self.op_pt * self.mch_working_power[:, np.newaxis, :]
        # copy the old state
        self.state = copy.deepcopy(self.old_state)

        return self.state

    def initial_vars(self):
        """
            initialize variables for further use
        """
        self.tasks_data = []
        self.step_count = 0 # 记录当前环境中经过的步数或时间步
        # the array that records the makespan of all environments
        self.current_makespan = np.full(self.number_of_envs, float("-inf")) # 
        self.current_makespan_normal = np.full(self.number_of_envs, float("-inf"))
        # self.current_EC = np.full(self.number_of_envs, float("-inf")) 
        # the complete time of operations ([E,N])
        self.op_ct = np.zeros((self.number_of_envs, self.number_of_ops))
        self.mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_remain_work = np.zeros((self.number_of_envs, self.number_of_machines))

        self.mch_waiting_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_working_flag = np.zeros((self.number_of_envs, self.number_of_machines))
        self.total_mch_waiting_time = np.zeros((self.number_of_envs, self.number_of_machines))

        self.next_schedule_time = np.zeros(self.number_of_envs)
        self.candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))

        self.true_op_ct = np.zeros((self.number_of_envs, self.number_of_ops))
        self.true_candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))
        self.true_mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))

        self.candidate = np.copy(self.job_first_op_id)

        # mask[i,j] : whether the jth job of ith env is scheduled (have no unscheduled operations)
        self.mask = np.full(shape=(self.number_of_envs, self.number_of_jobs), fill_value=0, dtype=bool)

        self.op_scheduled_flag = np.zeros((self.number_of_envs, self.number_of_ops))
        self.op_waiting_time = np.zeros((self.number_of_envs, self.number_of_ops))
        self.op_remain_work = np.zeros((self.number_of_envs, self.number_of_ops))

        self.op_available_mch_nums = np.copy(self.compatible_op) / self.number_of_machines
        self.pair_free_time = np.zeros((self.number_of_envs, self.number_of_jobs,
                                        self.number_of_machines))
        self.remain_process_relation = np.copy(self.process_relation)

        self.delete_mask_fea_j = np.full(shape=(self.number_of_envs, self.number_of_ops, self.op_fea_dim),
                                         fill_value=0, dtype=bool)
        # mask[i,j] : whether the jth op of ith env is deleted (from the set $O_u$)
        self.deleted_op_nodes = np.full(shape=(self.number_of_envs, self.number_of_ops),
                                        fill_value=0, dtype=bool)

        self.schedule_mch_working_energy = np.full(shape=(self.number_of_envs, ), fill_value=0, dtype=np.float64)
        

    def step(self, actions):
        """
            perform the state transition & return the next state and reward
        :param actions: the action list with shape [E]
        :return: the next state, reward and the done flag
        """
        chosen_job = actions // self.number_of_machines
        chosen_mch = actions % self.number_of_machines
        chosen_op = self.candidate[self.env_idxs, chosen_job]
        
        if (self.reverse_process_relation[self.env_idxs, chosen_op, chosen_mch]).any():
            print(
                f'FJSP_Env.py Error from choosing action: Op {chosen_op} can\'t be processed by Mch {chosen_mch}')
            sys.exit()

        self.step_count += 1

        # update candidate
        candidate_add_flag = (chosen_op != self.job_last_op_id[self.env_idxs, chosen_job])
        self.candidate[self.env_idxs, chosen_job] += candidate_add_flag
        self.mask[self.env_idxs, chosen_job] = (1 - candidate_add_flag)

        # the start processing time of chosen operations
        chosen_op_st = np.maximum(self.candidate_free_time[self.env_idxs, chosen_job],
                                  self.mch_free_time[self.env_idxs, chosen_mch])

        self.op_ct[self.env_idxs, chosen_op] = chosen_op_st + self.op_pt[
            self.env_idxs, chosen_op, chosen_mch]
        self.candidate_free_time[self.env_idxs, chosen_job] = self.op_ct[self.env_idxs, chosen_op]
        self.mch_free_time[self.env_idxs, chosen_mch] = self.op_ct[self.env_idxs, chosen_op]

        true_chosen_op_st = np.maximum(self.true_candidate_free_time[self.env_idxs, chosen_job],
                                       self.true_mch_free_time[self.env_idxs, chosen_mch])
        self.true_op_ct[self.env_idxs, chosen_op] = true_chosen_op_st + self.true_op_pt[
            self.env_idxs, chosen_op, chosen_mch]
        
        self.tasks_data.append({"Task": f"Job{chosen_job}", "Station": f"Machine{chosen_mch}", "Start": true_chosen_op_st[0], "Duration": self.true_op_pt[
            0, chosen_op, chosen_mch], "Width": 0.4})
        
        self.true_candidate_free_time[self.env_idxs, chosen_job] = self.true_op_ct[
            self.env_idxs, chosen_op]
        self.true_mch_free_time[self.env_idxs, chosen_mch] = self.true_op_ct[
            self.env_idxs, chosen_op]

        self.current_makespan = np.maximum(self.current_makespan, self.true_op_ct[
            self.env_idxs, chosen_op])
        # self.current_EC = np.maximum(self.current_EC, self.op_energy[self.env_idxs, chosen_op])
        # update the candidate message
        mask_temp = candidate_add_flag
        self.candidate_pt[mask_temp, chosen_job[mask_temp]] = self.unmasked_op_pt[mask_temp, chosen_op[mask_temp] + 1]
        self.candidate_ec[mask_temp, chosen_job[mask_temp]] = self.unmasked_op_energy[mask_temp, chosen_op[mask_temp] + 1]

        self.candidate_process_relation[mask_temp, chosen_job[mask_temp]] = \
            self.reverse_process_relation[mask_temp, chosen_op[mask_temp] + 1]
        self.candidate_process_relation[~mask_temp, chosen_job[~mask_temp]] = 1

        # compute the next schedule time

        # [E, J, M]
        candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)
        mchFT_for_compare = np.expand_dims(self.mch_free_time, axis=1)
        self.pair_free_time = np.maximum(candidateFT_for_compare, mchFT_for_compare)

        schedule_matrix = ma.array(self.pair_free_time, mask=self.candidate_process_relation)

        self.next_schedule_time = np.min(
            schedule_matrix.reshape(self.number_of_envs, -1), axis=1).data

        self.remain_process_relation[self.env_idxs, chosen_op] = 0
        self.op_scheduled_flag[self.env_idxs, chosen_op] = 1

        """
            update the mask for deleting nodes
        """
        self.deleted_op_nodes = \
            np.logical_and((self.op_ct <= self.next_schedule_time[:, np.newaxis]),
                           self.op_scheduled_flag)
        self.delete_mask_fea_j = np.tile(self.deleted_op_nodes[:, :, np.newaxis],
                                         (1, 1, self.op_fea_dim))

        """
            update the state
        """
        self.update_op_mask()

        # update operation raw features
        diff = self.op_ct[self.env_idxs, chosen_op] - self.op_ct_lb[self.env_idxs, chosen_op]

        mask1 = (self.op_idx >= chosen_op[:, np.newaxis]) & \
                (self.op_idx < (self.job_last_op_id[self.env_idxs, chosen_job] + 1)[:,
                               np.newaxis])
        self.op_ct_lb[mask1] += np.tile(diff[:, np.newaxis], (1, self.number_of_ops))[mask1]

        mask2 = (self.op_idx >= (self.job_first_op_id[self.env_idxs, chosen_job])[:,
                                np.newaxis]) & \
                (self.op_idx < (self.job_last_op_id[self.env_idxs, chosen_job] + 1)[:,
                               np.newaxis])
        self.op_match_job_left_op_nums[mask2] -= 1
        self.op_match_job_remain_work[mask2] -= \
            np.tile(self.op_mean_pt[self.env_idxs, chosen_op][:, np.newaxis], (1, self.number_of_ops))[mask2]

        self.op_waiting_time = np.zeros((self.number_of_envs, self.number_of_ops))
        self.op_waiting_time[self.env_job_idx, self.candidate] = \
            (1 - self.mask) * np.maximum(np.expand_dims(self.next_schedule_time, axis=1)
                                         - self.candidate_free_time, 0) + self.mask * self.op_waiting_time[
                self.env_job_idx, self.candidate]

        self.op_remain_work = np.maximum(self.op_ct -
                                         np.expand_dims(self.next_schedule_time, axis=1), 0)

        self.construct_op_features()

        # update dynamic pair mask
        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)

        self.unavailable_pairs = self.pair_free_time > self.next_schedule_time[:, np.newaxis, np.newaxis]

        self.dynamic_pair_mask = np.logical_or(self.dynamic_pair_mask, self.unavailable_pairs)

        # update comp_idx
        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)

        self.update_mch_mask()

        # update machine raw features
        self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
        self.mch_current_available_op_nums -= self.process_relation[
            self.env_idxs, chosen_op]

        mch_free_duration = np.expand_dims(self.next_schedule_time, axis=1) - self.mch_free_time
        mch_free_flag = mch_free_duration < 0
        self.mch_working_flag = mch_free_flag + 0
        self.mch_waiting_time = (1 - mch_free_flag) * mch_free_duration

        self.mch_remain_work = np.maximum(-mch_free_duration, 0)

        self.construct_mch_features()

        self.construct_pair_features()


        # compute the reward : R_t = C_{LB}(s_{t}) - C_{LB}(s_{t+1})
        reward = self.max_endTime - np.max(self.op_ct_lb, axis=1)
        # reward = self.max_endTime - np.max(self.op_ct_lb, axis=1)
        self.max_endTime = np.max(self.op_ct_lb, axis=1)

        # update the state
        self.state.update(self.fea_j, self.op_mask, self.fea_m, self.mch_mask,
                          self.dynamic_pair_mask, self.comp_idx, self.candidate,
                          self.fea_pairs)

        return self.state, np.array(reward), self.done()



    def done(self):
        """
            compute the done flag
        """
        return np.ones(self.number_of_envs) * (self.step_count >= self.number_of_ops)

    def construct_op_features(self):
        """
            construct operation raw features
        """
        self.fea_j = np.stack((self.op_scheduled_flag, # 操作是否已被调度
                               self.op_ct_lb,##操作的完工时间的估计下限
                               self.op_min_pt, ###操作的最小处理时间，最短的
                               self.pt_span, ##操作的处理时间变化范围。
                               self.op_mean_pt,#操作的平均处理时间
                               self.op_waiting_time,##操作的等待时间
                               self.op_remain_work,###操作剩余的工作量
                               self.op_match_job_left_op_nums,##与操作匹配的作业中剩余的操作数量
                               self.op_match_job_remain_work,#与操作匹配的作业中剩余的工作量
                               self.op_available_mch_nums, #可用于该操作的机器数量
                               ), axis=2)

        if self.step_count != self.number_of_ops:
            self.norm_op_features()

    def norm_op_features(self):
        """
            normalize operation raw features (across the second dimension)
        """
        self.fea_j[self.delete_mask_fea_j] = 0
        num_delete_nodes = np.count_nonzero(self.deleted_op_nodes, axis=1)
        num_delete_nodes = num_delete_nodes[:, np.newaxis]
        num_left_nodes = self.number_of_ops - num_delete_nodes
        mean_fea_j = np.sum(self.fea_j, axis=1) / num_left_nodes
        temp = np.where(self.delete_mask_fea_j, mean_fea_j[:, np.newaxis, :], self.fea_j)
        var_fea_j = np.var(temp, axis=1)

        std_fea_j = np.sqrt(var_fea_j * self.number_of_ops / num_left_nodes)

        self.fea_j = (temp - mean_fea_j[:, np.newaxis, :]) / \
                     (std_fea_j[:, np.newaxis, :] + 1e-8)

    def construct_mch_features(self):
        """
            construct machine raw features
        """
        self.fea_m = np.stack((self.mch_current_available_jc_nums, #当前可用于机器的作业计数（job count）数量
                               self.mch_current_available_op_nums, #当前机器可用的操作（operation）数量
                               self.mch_min_pt, ##机器上操作的最小处理时间
                               self.mch_mean_pt, # 机器上操作的平均处理时间
                               self.mch_waiting_time, # 机器的等待时间
                               self.mch_remain_work, # 机器上剩余的工作量
                               self.mch_free_time, ### 机器的空闲时间，可能与等待时间有所不同，更多地关注于机器的可用性。
                               self.mch_working_flag, # 标志机器是否正在工作
                               ), axis=2)

        if self.step_count != self.number_of_ops:
            self.norm_machine_features()

    def norm_machine_features(self):
        """
            normalize machine raw features (across the second dimension)
        """
        self.fea_m[self.delete_mask_fea_m] = 0
        num_delete_mchs = np.count_nonzero(self.delete_mask_fea_m[:, :, 0], axis=1)
        num_delete_mchs = num_delete_mchs[:, np.newaxis]
        num_left_mchs = self.number_of_machines - num_delete_mchs
        mean_fea_m = np.sum(self.fea_m, axis=1) / num_left_mchs
        temp = np.where(self.delete_mask_fea_m,
                        mean_fea_m[:, np.newaxis, :], self.fea_m)
        var_fea_m = np.var(temp, axis=1)
        std_fea_m = np.sqrt(var_fea_m * self.number_of_machines / num_left_mchs)

        self.fea_m = (temp - mean_fea_m[:, np.newaxis, :]) / \
                     (std_fea_m[:, np.newaxis, :] + 1e-8)
        
    def construct_pair_features(self):
        """
            构建成对特征
        """
        # 对于每个作业中的每个操作，计算其剩余加工时间。如果操作已完成，则掩盖其加工时间。
        remain_op_pt = ma.array(self.op_pt, mask=~self.remain_process_relation)

        # 选定作业中的最大加工时间
        chosen_op_max_pt = np.expand_dims(self.op_max_pt[self.env_job_idx, self.candidate], axis=-1)

        # 计算所有剩余操作中的最大加工时间
        max_remain_op_pt = np.max(np.max(remain_op_pt, axis=1, keepdims=True), axis=2, keepdims=True).filled(0 + 1e-8)

        # 计算每台机器上剩余操作的最大加工时间
        mch_max_remain_op_pt = np.max(remain_op_pt, axis=1, keepdims=True).filled(0 + 1e-8)

        # 计算候选操作中的最大加工时间
        pair_max_pt = np.max(np.max(self.candidate_pt, axis=1, keepdims=True), axis=2, keepdims=True) + 1e-8

        # 计算每台机器上候选操作的最大加工时间
        mch_max_candidate_pt = np.max(self.candidate_pt, axis=1, keepdims=True) + 1e-8

        # 计算候选操作的等待时间，考虑作业和机器的等待时间
        pair_wait_time = self.op_waiting_time[self.env_job_idx, self.candidate][:, :, np.newaxis] + self.mch_waiting_time[:, np.newaxis, :]

        # 计算选定作业的剩余工作量
        chosen_job_remain_work = np.expand_dims(self.op_match_job_remain_work[self.env_job_idx, self.candidate], axis=-1) + 1e-8

        # 将所有计算出的特征堆叠成一个特征数组，以便后续处理
        self.fea_pairs = np.stack((self.candidate_pt,
                                self.candidate_pt / chosen_op_max_pt,
                                self.candidate_pt / mch_max_candidate_pt,
                                self.candidate_pt / max_remain_op_pt,
                                self.candidate_pt / mch_max_remain_op_pt,
                                self.candidate_pt / pair_max_pt,
                                self.candidate_pt / chosen_job_remain_work,
                                pair_wait_time), axis=-1)

    def update_mch_mask(self):
        """
            update 'mch_mask'
        """
        self.mch_mask = self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        self.delete_mask_fea_m = np.tile(~(np.sum(self.mch_mask, keepdims=True, axis=-1).astype(bool)),
                                         (1, 1, self.mch_fea_dim))
        self.mch_mask[self.multi_env_mch_diag] = 1

    def init_mch_mask(self):
        """
            initialize 'mch_mask'
        """
        self.mch_mask = self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        self.delete_mask_fea_m = np.tile(~(np.sum(self.mch_mask, keepdims=True, axis=-1).astype(bool)),
                                         (1, 1, self.mch_fea_dim))
        self.mch_mask[self.multi_env_mch_diag] = 1

    def init_op_mask(self):
        """
            initialize 'op_mask'
        """
        self.op_mask = np.full(shape=(self.number_of_envs, self.number_of_ops, 3),
                               fill_value=0, dtype=np.float32)
        self.op_mask[self.env_job_idx, self.job_first_op_id, 0] = 1
        self.op_mask[self.env_job_idx, self.job_last_op_id, 2] = 1

    def update_op_mask(self):
        """
            update 'op_mask'
        """
        object_mask = np.zeros_like(self.op_mask)
        object_mask[:, :, 2] = self.deleted_op_nodes
        object_mask[:, 1:, 0] = self.deleted_op_nodes[:, :-1]
        self.op_mask = np.logical_or(object_mask, self.op_mask).astype(np.float32)

    def logic_operator(self, x, flagT=True):
        """
            a customized operator for computing some masks
        :param x: a 3-d array with shape [s,a,b]
        :param flagT: whether transpose x in the last two dimensions
        :return:  a 4-d array c, where c[i,j,k,l] = x[i,j,l] & x[i,k,l] for each i,j,k,l
        """
        if flagT:
            x = x.transpose(0, 2, 1)
        d1 = np.expand_dims(x, 2)
        d2 = np.expand_dims(x, 1)

        return np.logical_and(d1, d2).astype(np.float32)
    
    def render(self):
        '''tasks_data = [
        {"Task": "Job1-Task1", "Station": "Machine1", "Start": 0, "Duration": 4, "Width": 0.4},
        {"Task": "Job2-Task1", "Station": "Machine2", "Start": 5, "Duration": 3, "Width": 0.4},
        {"Task": "Job3-Task1", "Station": "Machine3", "Start": 9, "Duration": 2, "Width": 0.4},
        ]   '''

        
        # 获取唯一的Job名称列表
        unique_jobs = list(set(task['Task'] for task in self.tasks_data))

        # 使用Plotly的定性颜色循环
        color_sequence = qualitative.Plotly

        # 如果Job数量超过内置颜色，可以生成新颜色
        if len(unique_jobs) > len(color_sequence):
            # 可以添加一个生成颜色代码的方法
            # 这里仅为了演示，我们重复使用内置颜色序列
            extra_colors_needed = len(unique_jobs) - len(color_sequence)
            color_sequence.extend(color_sequence[:extra_colors_needed])

        # 创建颜色映射
        color_map = {job: color for job, color in zip(unique_jobs, color_sequence)}
        self.fig.data = []
        for task in self.tasks_data:
            self.fig.add_trace(go.Bar(
                x=[task["Duration"]],
                y=[task["Station"]],
                base=[task["Start"]],
                width=[task["Width"]],
                orientation='h',
                name=task["Task"],
                marker_color=color_map[task["Task"]],
            ))

        # 更新图表布局
        self.fig.update_layout(
            title="按Job上色的FJSP调度示意图 - 多种颜色",
            xaxis_title="时间",
            yaxis=dict(
                type='category',
                categoryorder='array',
                categoryarray=[task['Station'] for task in self.tasks_data]
            ),
            barmode='stack',
            legend_title="Job",
        )

        self.fig.show()
    
    def initialize_pygame(self):
        pygame.init()
        self.screen_width = self.number_of_jobs * np.mean(self.op_mean_pt) * 120 + 100
        self.screen_height = self.number_of_machines * 100 + 100
        # print((self.screen_width, self.screen_height))
        screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        return screen
    
    def get_color_for_task(self, task_name,):
        if task_name not in self.task_colors:
            # 随机生成颜色，避免颜色过暗
            random_color = (random.randint(64, 255), random.randint(64, 255), random.randint(64, 255))
            self.task_colors[task_name] = random_color
        return self.task_colors[task_name]


    def render_gantt_chart(self, tasks_data, screen):
        WHITE = (255, 255, 255)
        
        
        screen.fill(WHITE)
        font = pygame.font.Font(None, 20)
        # 画出每个任务的颜色图例
        task_set = set()
        for task in self.tasks_data:
            task_set.add(task['Task'])
        tasks_list = sorted(list(task_set))
        
        for i, task in enumerate(tasks_list):
            color = self.get_color_for_task(task)
            pygame.draw.rect(screen, color, (self.screen_width - 100, 30 + i * 30, 20, 20))
            text = font.render(task, True, (0, 0, 0))
            screen.blit(text, (self.screen_width - 70, 30 + i * 30))

        # 绘制机器标签和任务
        machine_set = set()
        for task in tasks_data:
            machine_set.add(task['Station'])
        machine_list = sorted(list(machine_set), key=lambda x: int(x.replace('Machine', '')))

        for task in tasks_data:
            color = self.get_color_for_task(task['Task'])
            x = 100 + task['Start']  # 缩放因子调整
            y = 100 + int(task['Station'].replace('Machine', '')) * 100  # 间隔调整
            width = task['Duration'] - 1   # 缩放因子调整
            height = int(60 * task['Width'])  # 高度调整
            pygame.draw.rect(screen, color, (x, y, width, height))
        
        # 绘制机器标签
        for i, machine in enumerate(machine_list):
            text = font.render(machine, True, (0, 0, 0))
            screen.blit(text, (25, 106 + i * 100))

        pygame.display.flip()

    def render_pygame(self):
        if self.step_count == 0: return
        if self.step_count == 1:
            self.task_colors = {}
            self.screen = self.initialize_pygame()
        self.render_gantt_chart(self.tasks_data, self.screen)

class FJSPEnvForSameOpNumsEnergy(FJSPEnvForSameOpNums):
    # def calculate_working_energy(self, operation_time, machine_id):
    #     # 基于操作时间计算机器工作时的能耗
    #     energy_rate_working = 1.0  # 假设工作能耗率为1.0
    #     energy_consumed = operation_time * energy_rate_working
    #     return energy_consumed

    # def calculate_idle_energy(self, idle_time, machine_id):
    #     # 基于空闲时间计算机器空闲时的能耗
    #     energy_rate_idle = 0.1  # 假设空闲能耗率为0.1
    #     energy_consumed = idle_time * energy_rate_idle
    #     return energy_consumed
    def __init__(self, n_j, n_m, factor_Mk=0.0, factor_Ec=1):
        """
        :param n_j: the number of jobs
        :param n_m: the number of machines
        """
        self.number_of_jobs = n_j
        self.number_of_machines = n_m
        self.old_state = EnvState()
        self.factor_Mk = factor_Mk
        self.factor_Ec = factor_Ec
        # the dimension of operation raw features
        self.op_fea_dim = 12
        # the dimension of machine raw features
        self.mch_fea_dim = 9
        self.fig = go.Figure()

    def step(self, actions):
        """
            perform the state transition & return the next state and reward
        :param actions: the action list with shape [E]
        :return: the next state, reward and the done flag
        """
        chosen_job = actions // self.number_of_machines
        chosen_mch = actions % self.number_of_machines
        chosen_op = self.candidate[self.env_idxs, chosen_job]



        if (self.reverse_process_relation[self.env_idxs, chosen_op, chosen_mch]).any():
            print(
                f'FJSP_Env.py Error from choosing action: Op {chosen_op} can\'t be processed by Mch {chosen_mch}')
            sys.exit()

        self.step_count += 1

        # update candidate
        candidate_add_flag = (chosen_op != self.job_last_op_id[self.env_idxs, chosen_job])
        self.candidate[self.env_idxs, chosen_job] += candidate_add_flag
        self.mask[self.env_idxs, chosen_job] = (1 - candidate_add_flag)

        # the start processing time of chosen operations
        chosen_op_st = np.maximum(self.candidate_free_time[self.env_idxs, chosen_job],
                                  self.mch_free_time[self.env_idxs, chosen_mch])

        self.op_ct[self.env_idxs, chosen_op] = chosen_op_st + self.op_pt[
            self.env_idxs, chosen_op, chosen_mch]
        # 修改工件和机器的空闲时间
        self.candidate_free_time[self.env_idxs, chosen_job] = self.op_ct[self.env_idxs, chosen_op]
        self.mch_free_time[self.env_idxs, chosen_mch] = self.op_ct[self.env_idxs, chosen_op]
        self.current_makespan_normal = np.maximum(self.current_makespan_normal, self.op_ct[
            self.env_idxs, chosen_op])
        
        #修改相关真实值
        true_chosen_op_st = np.maximum(self.true_candidate_free_time[self.env_idxs, chosen_job],
                                       self.true_mch_free_time[self.env_idxs, chosen_mch])
        self.true_op_ct[self.env_idxs, chosen_op] = true_chosen_op_st + self.true_op_pt[
            self.env_idxs, chosen_op, chosen_mch]
        
        self.tasks_data.append({"Task": f"Job{chosen_job[0]}", "Station": f"Machine{chosen_mch[0]}", "Start": true_chosen_op_st[0], 
                                "Duration": self.true_op_pt[self.env_idxs, chosen_op, chosen_mch][0], "Width": 0.4})
        
        self.true_candidate_free_time[self.env_idxs, chosen_job] = self.true_op_ct[
            self.env_idxs, chosen_op]
        self.true_mch_free_time[self.env_idxs, chosen_mch] = self.true_op_ct[
            self.env_idxs, chosen_op]

        self.current_makespan = np.maximum(self.current_makespan, self.true_op_ct[
            self.env_idxs, chosen_op])
        # self.current_EC = np.maximum(self.current_EC, self.total_min_energy)
        # update the candidate message
        mask_temp = candidate_add_flag
        self.candidate_pt[mask_temp, chosen_job[mask_temp]] = self.unmasked_op_pt[mask_temp, chosen_op[mask_temp] + 1]

        self.candidate_process_relation[mask_temp, chosen_job[mask_temp]] = \
            self.reverse_process_relation[mask_temp, chosen_op[mask_temp] + 1]
        self.candidate_process_relation[~mask_temp, chosen_job[~mask_temp]] = 1

        # compute the next schedule time

        # [E, J, M]
        candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)
        mchFT_for_compare = np.expand_dims(self.mch_free_time, axis=1)
        self.pair_free_time = np.maximum(candidateFT_for_compare, mchFT_for_compare)

        schedule_matrix = ma.array(self.pair_free_time, mask=self.candidate_process_relation)

        self.next_schedule_time = np.min(
            schedule_matrix.reshape(self.number_of_envs, -1), axis=1).data

        self.remain_process_relation[self.env_idxs, chosen_op] = 0
        self.op_scheduled_flag[self.env_idxs, chosen_op] = 1

        """
            update the mask for deleting nodes
        """
        self.deleted_op_nodes = \
            np.logical_and((self.op_ct <= self.next_schedule_time[:, np.newaxis]),
                           self.op_scheduled_flag)
        self.delete_mask_fea_j = np.tile(self.deleted_op_nodes[:, :, np.newaxis],
                                         (1, 1, self.op_fea_dim))

        """
            update the state
        """
        self.update_op_mask()

        # update operation raw features
        diff = self.op_ct[self.env_idxs, chosen_op] - self.op_ct_lb[self.env_idxs, chosen_op]

        mask1 = (self.op_idx >= chosen_op[:, np.newaxis]) & \
                (self.op_idx < (self.job_last_op_id[self.env_idxs, chosen_job] + 1)[:,
                               np.newaxis])
        self.op_ct_lb[mask1] += np.tile(diff[:, np.newaxis], (1, self.number_of_ops))[mask1]

        mask2 = (self.op_idx >= (self.job_first_op_id[self.env_idxs, chosen_job])[:,
                                np.newaxis]) & \
                (self.op_idx < (self.job_last_op_id[self.env_idxs, chosen_job] + 1)[:,
                               np.newaxis])
        self.op_match_job_left_op_nums[mask2] -= 1
        self.op_match_job_remain_work[mask2] -= \
            np.tile(self.op_mean_pt[self.env_idxs, chosen_op][:, np.newaxis], (1, self.number_of_ops))[mask2]

        self.op_waiting_time = np.zeros((self.number_of_envs, self.number_of_ops))
        self.op_waiting_time[self.env_job_idx, self.candidate] = \
            (1 - self.mask) * np.maximum(np.expand_dims(self.next_schedule_time, axis=1)
                                         - self.candidate_free_time, 0) + self.mask * self.op_waiting_time[
                self.env_job_idx, self.candidate]

        self.op_remain_work = np.maximum(self.op_ct -
                                         np.expand_dims(self.next_schedule_time, axis=1), 0)

        

        # update dynamic pair mask
        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)

        self.unavailable_pairs = self.pair_free_time > self.next_schedule_time[:, np.newaxis, np.newaxis]

        self.dynamic_pair_mask = np.logical_or(self.dynamic_pair_mask, self.unavailable_pairs)

        # update comp_idx
        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)

        self.update_mch_mask()

        # update machine raw features
        self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
        self.mch_current_available_op_nums -= self.process_relation[
            self.env_idxs, chosen_op]

        mch_free_duration = np.expand_dims(self.next_schedule_time, axis=1) - self.mch_free_time
        mch_free_flag = mch_free_duration < 0
        self.mch_working_flag = mch_free_flag + 0
        self.mch_waiting_time = (1 - mch_free_flag) * mch_free_duration 

        self.mch_remain_work = np.maximum(-mch_free_duration, 0)

        # 计算机器空闲时的能耗
        self.total_mch_waiting_time = self.total_mch_waiting_time + self.mch_waiting_time if (self.mch_waiting_time < 1.e10).any() else 0.01
        idle_energy_consumption = self.total_mch_waiting_time * self.mch_working_power
        idle_energy = np.sum(idle_energy_consumption, axis = -1)

        # 计算机器工作时的能耗
        
        self.schedule_mch_working_energy += self.op_pt[self.env_idxs, chosen_op, chosen_mch] * self.mch_working_power[self.env_idxs, chosen_mch]
        
        self.op_energy[self.env_idxs, chosen_op, :] = 0
        unscheduled_energy_lb_idx = np.argmin(self.op_energy, axis=-1)  ###每个op最小的能量消耗的索引
        unscheduled_energy_lb_energy = np.min(self.op_energy, axis=-1)
        self.op_energy_lb = unscheduled_energy_lb_energy.copy()
        min_energy_per_mch = np.zeros(shape=(self.number_of_envs, self.number_of_machines))
        # 遍历并将能量最小值加到对应的机器上
        for i in range(unscheduled_energy_lb_idx.shape[0]):  # 遍历每行
            for j in range(unscheduled_energy_lb_idx.shape[1]):  # 遍历每列
                idx = unscheduled_energy_lb_idx[i, j]  # 获取当前能量值对应的机器索引
                energy = unscheduled_energy_lb_energy[i, j]  # 获取当前的能量值
                min_energy_per_mch[i, idx] += energy  # 将能量值加到对应的机器上
        self.energy_lb_idx = unscheduled_energy_lb_idx
        self.energy_lb_energy = unscheduled_energy_lb_energy
        working_energy = self.schedule_mch_working_energy + np.sum(min_energy_per_mch, axis=-1)
        
        self.construct_op_features()
        self.construct_mch_features()

        self.construct_pair_features()

        # mch_working_time = self.current_makespan_normal[:, np.newaxis] - self.total_mch_waiting_time

        # mch_power_consumption = mch_working_time * self.mch_working_power
        # working_energy = np.sum(mch_power_consumption, axis = -1)
        # 总能耗为工作能耗加上空闲能耗
        total_energy_consumed = working_energy + idle_energy
        
        # self.current_EC = np.maximum(self.current_EC, total_energy_consumed)
        # compute the reward : R_t = C_{LB}(s_{t}) - C_{LB}(s_{t+1})
        rewardEC = self.current_EC - total_energy_consumed
        reward_mk = self.max_endTime - np.max(self.op_ct_lb, axis=1)
        reward = self.factor_Mk * reward_mk + self.factor_Ec * rewardEC

        # reward = self.max_endTime - np.max(self.op_ct_lb, axis=1)
        self.max_endTime = np.max(self.op_ct_lb, axis=1)
        self.current_EC = total_energy_consumed

        # update the state
        self.state.update(self.fea_j, self.op_mask, self.fea_m, self.mch_mask,
                          self.dynamic_pair_mask, self.comp_idx, self.candidate,
                          self.fea_pairs)

        return self.state, np.array(reward), self.done()


    def construct_mch_features(self):
        """
            construct machine raw features
        """
        self.fea_m = np.stack((self.mch_current_available_jc_nums, #当前可用于机器的作业计数（job count）数量
                               self.mch_current_available_op_nums, #当前机器可用的操作（operation）数量
                               self.mch_min_ec, ##机器上操作的最小处理时间
                               self.mch_mean_ec, # 机器上操作的平均处理时间
                               self.mch_waiting_time, # 机器的等待时间
                               self.mch_remain_work, # 机器上剩余的工作量
                               self.mch_free_time, ### 机器的空闲时间，可能与等待时间有所不同，更多地关注于机器的可用性。
                               self.mch_working_flag, # 标志机器是否正在工作
                               self.mch_working_power
                               ), axis=2)

    def construct_op_features(self):
        """
            construct operation raw features
        """
        self.fea_j = np.stack((self.op_scheduled_flag, # 操作是否已被调度
                               self.op_energy_lb,##操作的完工时间的估计下限
                               self.op_min_energy, ###操作的最小处理时间，最短的
                               self.energy_span, ##操作的处理时间变化范围。
                               self.op_mean_ec,#操作的平均处理时间
                               self.op_waiting_time,##操作的等待时间
                               self.op_remain_work,###操作剩余的工作量
                               self.op_match_job_left_op_nums,##与操作匹配的作业中剩余的操作数量
                               self.op_match_job_remain_work,#与操作匹配的作业中剩余的工作量
                               self.op_available_mch_nums, #可用于该操作的机器数量
                                self.energy_lb_idx,
                                self.energy_lb_energy, 
                               ), axis=2)

        if self.step_count != self.number_of_ops:
            self.norm_op_features()

    def construct_pair_features_(self):
        """
            构建成对特征
        """
        # 对于每个作业中的每个操作，计算其剩余加工时间。如果操作已完成，则掩盖其加工时间。
        remain_op_energy = ma.array(self.op_energy, mask=~self.remain_process_relation)
        remain_op_pt = ma.array(self.op_pt, mask=~self.remain_process_relation)

        # 选定作业中的最大加工时间
        chosen_op_max_energy = np.expand_dims(self.op_max_energy[self.env_job_idx, self.candidate], axis=-1)
        chosen_op_max_pt = np.expand_dims(self.op_max_pt[self.env_job_idx, self.candidate], axis=-1)

        # 计算所有剩余操作中的最大加工时间
        max_remain_op_energy = np.max(np.max(remain_op_energy, axis=1, keepdims=True), axis=2, keepdims=True).filled(0 + 1e-8)
        max_remain_op_pt = np.max(np.max(remain_op_pt, axis=1, keepdims=True), axis=2, keepdims=True).filled(0 + 1e-8)

        # 计算每台机器上剩余操作的最大加工时间
        mch_max_remain_op_energy = np.max(remain_op_energy, axis=1, keepdims=True).filled(0 + 1e-8)
        mch_max_remain_op_pt = np.max(remain_op_pt, axis=1, keepdims=True).filled(0 + 1e-8)

        # 计算候选操作中的最大加工时间
        pair_max_ec = np.max(np.max(self.candidate_ec, axis=1, keepdims=True), axis=2, keepdims=True) + 1e-8
        pair_max_pt = np.max(np.max(self.candidate_pt, axis=1, keepdims=True), axis=2, keepdims=True) + 1e-8

        # 计算每台机器上候选操作的最大加工时间
        mch_max_candidate_ec = np.max(self.candidate_ec, axis=1, keepdims=True) + 1e-8
        mch_max_candidate_pt = np.max(self.candidate_pt, axis=1, keepdims=True) + 1e-8

        # 计算候选操作的等待时间，考虑作业和机器的等待时间
        pair_wait_time = self.op_waiting_time[self.env_job_idx, self.candidate][:, :, np.newaxis] + self.mch_waiting_time[:, np.newaxis, :]

        # 计算选定作业的剩余工作量
        chosen_job_remain_work = np.expand_dims(self.op_match_job_remain_work[self.env_job_idx, self.candidate], axis=-1) + 1e-8

        # 将所有计算出的特征堆叠成一个特征数组，以便后续处理
        self.fea_pairs = np.stack((self.candidate_ec,
                                self.candidate_ec / chosen_op_max_pt,
                                self.candidate_ec / mch_max_candidate_ec,
                                self.candidate_ec / max_remain_op_energy,
                                self.candidate_ec / mch_max_remain_op_energy,
                                self.candidate_ec / pair_max_ec,
                                self.candidate_ec / chosen_job_remain_work,
                                pair_wait_time), axis=-1)
