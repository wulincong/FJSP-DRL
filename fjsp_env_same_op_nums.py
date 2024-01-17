from dataclasses import dataclass
import numpy as np
import numpy.ma as ma
import copy
from params import configs
import sys
import torch


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

    def set_initial_data(self, job_length_list, op_pt_list):
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
        self.op_min_pt = np.min(self.op_pt, axis=-1).data
        self.op_max_pt = np.max(self.op_pt, axis=-1).data
        self.pt_span = self.op_max_pt - self.op_min_pt  # 操作加工时间范围
        # [E, M]机器的最小和最大加工时间
        self.mch_min_pt = np.max(self.op_pt, axis=1).data
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

        self.construct_op_features()

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

        self.mch_mean_pt = np.mean(self.op_pt, axis=1).filled(0) # 
        # construct machine features [E, M, 6]

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
        self.old_op_match_job_left_op_nums = np.copy(self.op_match_job_left_op_nums)
        self.old_op_match_job_remain_work = np.copy(self.op_match_job_remain_work)
        self.old_init_quality = np.copy(self.init_quality)
        self.old_candidate_pt = np.copy(self.candidate_pt)
        self.old_candidate_process_relation = np.copy(self.candidate_process_relation)
        self.old_mch_current_available_op_nums = np.copy(self.mch_current_available_op_nums)
        self.old_mch_current_available_jc_nums = np.copy(self.mch_current_available_jc_nums)
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
        self.op_match_job_left_op_nums = np.copy(self.old_op_match_job_left_op_nums)
        self.op_match_job_remain_work = np.copy(self.old_op_match_job_remain_work)
        self.init_quality = np.copy(self.old_init_quality)
        self.max_endTime = self.init_quality
        self.candidate_pt = np.copy(self.old_candidate_pt)
        self.candidate_process_relation = np.copy(self.old_candidate_process_relation)
        self.mch_current_available_op_nums = np.copy(self.old_mch_current_available_op_nums)
        self.mch_current_available_jc_nums = np.copy(self.old_mch_current_available_jc_nums)
        # copy the old state
        self.state = copy.deepcopy(self.old_state)
        return self.state

    def initial_vars(self):
        """
            initialize variables for further use
        """
        self.step_count = 0 # 记录当前环境中经过的步数或时间步
        # the array that records the makespan of all environments
        self.current_makespan = np.full(self.number_of_envs, float("-inf")) # 
        # the complete time of operations ([E,N])
        self.op_ct = np.zeros((self.number_of_envs, self.number_of_ops))
        self.mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_remain_work = np.zeros((self.number_of_envs, self.number_of_machines))

        self.mch_waiting_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_working_flag = np.zeros((self.number_of_envs, self.number_of_machines))

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
        self.true_candidate_free_time[self.env_idxs, chosen_job] = self.true_op_ct[
            self.env_idxs, chosen_op]
        self.true_mch_free_time[self.env_idxs, chosen_mch] = self.true_op_ct[
            self.env_idxs, chosen_op]

        self.current_makespan = np.maximum(self.current_makespan, self.true_op_ct[
            self.env_idxs, chosen_op])

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
                               self.op_available_mch_nums#可用于该操作的机器数量
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
                               self.mch_working_flag # 标志机器是否正在工作
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
