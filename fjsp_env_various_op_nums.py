import numpy as np
import numpy.ma as ma
import copy
import sys
from fjsp_env_same_op_nums import EnvState
from plotly.colors import qualitative
import plotly.graph_objects as go
import colorsys
import random


class FJSPEnvForVariousOpNums:
    """
        a batch of fjsp environments that have various number of operations
        Attributes:

    """

    def __init__(self, n_j, n_m):
        self.number_of_jobs = n_j  # 设置作业数量
        self.number_of_machines = n_m  # 设置机器数量
        self.old_state = EnvState() 

        self.op_fea_dim = 10  # 操作特征维度
        self.mch_fea_dim = 8  # 机器特征维度
        self.fig = go.Figure()

    def set_static_properties(self):
        """
            define static properties
        """
        self.multi_env_mch_diag = np.tile(np.expand_dims(np.eye(self.number_of_machines, dtype=bool), axis=0),
                                          (self.number_of_envs, 1, 1))

        self.env_idxs = np.arange(self.number_of_envs)
        self.env_job_idx = self.env_idxs.repeat(self.number_of_jobs).reshape(self.number_of_envs, self.number_of_jobs)

        # [E, N]
        self.mask_dummy_node = np.full(shape=[self.number_of_envs, self.max_number_of_ops],
                                       fill_value=False, dtype=bool)

        cols = np.arange(self.max_number_of_ops)
        self.mask_dummy_node[cols >= self.env_number_of_ops[:, None]] = True

        a = self.mask_dummy_node[:, :, np.newaxis]
        self.dummy_mask_fea_j = np.tile(a, (1, 1, self.op_fea_dim))

        self.flag_exist_dummy_node = ~(self.env_number_of_ops == self.max_number_of_ops).all()

    def set_initial_data(self, job_length_list, op_pt_list):
        self.number_of_envs = len(job_length_list)
        self.job_length = np.array(job_length_list)
        self.number_of_machines = op_pt_list[0].shape[1]
        self.number_of_jobs = job_length_list[0].shape[0]

        # 异工序数环境并行化
        self.env_number_of_ops = np.array([op_pt_list[k].shape[0] for k in range(self.number_of_envs)])
        self.max_number_of_ops = np.max(self.env_number_of_ops)

        self.set_static_properties()

        self.virtual_job_length = np.copy(self.job_length)
        self.virtual_job_length[:, -1] += self.max_number_of_ops - self.env_number_of_ops

        # [E, N, M]
        self.op_pt = np.array([np.pad(op_pt_list[k],
                                      ((0, self.max_number_of_ops - self.env_number_of_ops[k]),
                                       (0, 0)),
                                      'constant', constant_values=0)
                               for k in range(self.number_of_envs)]).astype(np.float64)

        self.pt_lower_bound = np.min(self.op_pt)
        self.pt_upper_bound = np.max(self.op_pt)
        self.true_op_pt = np.copy(self.op_pt)
        epsilon = 1e-6  # 小正数，用于归一化的最小值
        non_zero_mask = (self.op_pt > 0)  # 标记加工时间大于 0 的位置

        # 仅对非零位置进行归一化，零值位置保持为 0
        normalized_op_pt = epsilon + (1 - epsilon) * (self.op_pt - self.pt_lower_bound) / (self.pt_upper_bound - self.pt_lower_bound + 1e-8)
        self.op_pt = np.where(non_zero_mask, normalized_op_pt, 0)

        # self.op_pt = (self.op_pt - self.pt_lower_bound) / (self.pt_upper_bound - self.pt_lower_bound + 1e-8)

        self.process_relation = (self.op_pt != 0)
        self.reverse_process_relation = ~self.process_relation

        self.compatible_op = np.sum(self.process_relation, 2)
        self.compatible_mch = np.sum(self.process_relation, 1)

        self.unmasked_op_pt = np.copy(self.op_pt)

        head_op_id = np.zeros((self.number_of_envs, 1))

        self.job_first_op_id = np.concatenate([head_op_id, np.cumsum(self.job_length, axis=1)[:, :-1]], axis=1).astype(
            'int')
        self.job_last_op_id = self.job_first_op_id + self.job_length - 1
        self.job_last_op_id[:, -1] = self.env_number_of_ops - 1

        self.initial_vars()
        self.init_op_mask()

        self.op_pt = ma.array(self.op_pt, mask=self.reverse_process_relation)
        self.op_mean_pt = np.mean(self.op_pt, axis=2).data

        self.op_min_pt = np.min(self.op_pt, axis=-1).data
        self.op_max_pt = np.max(self.op_pt, axis=-1).data
        self.pt_span = self.op_max_pt - self.op_min_pt

        self.mch_min_pt = np.max(self.op_pt, axis=1).data
        self.mch_max_pt = np.max(self.op_pt, axis=1)

        self.op_ct_lb = copy.deepcopy(self.op_min_pt)

        for k in range(self.number_of_envs):
            for i in range(self.number_of_jobs):
                self.op_ct_lb[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1] = np.cumsum(
                    self.op_ct_lb[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1])

        self.op_match_job_left_op_nums = np.array([np.repeat(self.job_length[k],
                                                             repeats=self.virtual_job_length[k])
                                                   for k in range(self.number_of_envs)])
        self.job_remain_work = []
        for k in range(self.number_of_envs):
            self.job_remain_work.append(
                [np.sum(self.op_mean_pt[k][self.job_first_op_id[k][i]:self.job_last_op_id[k][i] + 1])
                 for i in range(self.number_of_jobs)])

        self.op_match_job_remain_work = np.array([np.repeat(self.job_remain_work[k], repeats=self.virtual_job_length[k])
                                                  for k in range(self.number_of_envs)])

        self.construct_op_features()

        # shape reward
        self.init_quality = np.max(self.op_ct_lb, axis=1)

        self.max_endTime = self.init_quality
        # old
        self.mch_available_op_nums = np.copy(self.compatible_mch)
        self.mch_current_available_op_nums = np.copy(self.compatible_mch)
        self.candidate_pt = np.array([self.unmasked_op_pt[k][self.candidate[k]] for k in range(self.number_of_envs)])

        self.dynamic_pair_mask = (self.candidate_pt == 0)
        self.candidate_process_relation = np.copy(self.dynamic_pair_mask)
        self.mch_current_available_jc_nums = np.sum(~self.candidate_process_relation, axis=1)

        self.mch_mean_pt = np.mean(self.op_pt, axis=1).filled(0)
        # construct machine features [E, M, 6]

        # construct Compete Tensor : [E, M, M, J]
        self.comp_idx = self.logic_operator(~self.dynamic_pair_mask)
        # construct mch graph adjacency matrix : [E, M, M]
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
        # self.old_pairMessage = np.copy(self.pairMessage)
        self.old_candidate_process_relation = np.copy(self.candidate_process_relation)
        self.old_mch_current_available_op_nums = np.copy(self.mch_current_available_op_nums)
        self.old_mch_current_available_jc_nums = np.copy(self.mch_current_available_jc_nums)
        # state
        self.state = copy.deepcopy(self.old_state)
        
        # for disturbance events
        self.tasks_data = [{"Task": "Job-1", "Station": f"M{i}", "Start": 0, "Duration": 0, "Width": 0.4} for i in range(self.number_of_machines-1, -1, -1) ]
        self.maintenance_schedule = {}
        return self.state

    def reset(self):
        self.initial_vars()
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
        # state
        self.state = copy.deepcopy(self.old_state)
        self.tasks_data = [{"Task": "Job-1", "Station": f"M{i}", "Start": 0, "Duration": 0, "Width": 0.4} for i in range(self.number_of_machines-1, -1, -1) ]
        return self.state

    def initial_vars(self):
        self.step_count = 0
        self.done_flag = np.full(shape=(self.number_of_envs,), fill_value=0, dtype=bool)
        self.current_makespan = np.full(self.number_of_envs, float("-inf"))
        self.mch_queue = np.full(shape=[self.number_of_envs, self.number_of_machines,
                                        self.max_number_of_ops + 1], fill_value=-99, dtype=int)
        self.mch_queue_len = np.zeros((self.number_of_envs, self.number_of_machines), dtype=int)
        self.mch_queue_last_op_id = np.zeros((self.number_of_envs, self.number_of_machines), dtype=int)
        self.op_ct = np.zeros((self.number_of_envs, self.max_number_of_ops))

        self.mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_remain_work = np.zeros((self.number_of_envs, self.number_of_machines))

        self.mch_waiting_time = np.zeros((self.number_of_envs, self.number_of_machines))
        self.mch_working_flag = np.zeros((self.number_of_envs, self.number_of_machines))

        self.next_schedule_time = np.zeros(self.number_of_envs)
        self.candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))

        self.true_op_ct = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.true_candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))
        self.true_mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))

        self.candidate = np.copy(self.job_first_op_id)

        self.unscheduled_op_nums = np.copy(self.env_number_of_ops)
        self.mask = np.full(shape=(self.number_of_envs, self.number_of_jobs), fill_value=0, dtype=bool)

        self.op_scheduled_flag = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.op_waiting_time = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.op_remain_work = np.zeros((self.number_of_envs, self.max_number_of_ops))

        self.op_available_mch_nums = np.copy(self.compatible_op) / self.number_of_machines
        self.pair_free_time = np.zeros((self.number_of_envs, self.number_of_jobs,
                                        self.number_of_machines))
        self.remain_process_relation = np.copy(self.process_relation)

        self.delete_mask_fea_j = np.full(shape=(self.number_of_envs, self.max_number_of_ops, self.op_fea_dim),
                                         fill_value=0, dtype=bool)

    def step(self, actions):
        self.incomplete_env_idx = np.where(self.done_flag == 0)[0]
        self.number_of_incomplete_envs = int(self.number_of_envs - np.sum(self.done_flag))
        chosen_job = actions // self.number_of_machines
        chosen_mch = actions % self.number_of_machines
        chosen_op = self.candidate[self.incomplete_env_idx, chosen_job]

        if (self.reverse_process_relation[self.incomplete_env_idx, chosen_op, chosen_mch]).any():
            print(
                f'FJSP_Env.py Error from choosing action: Op {chosen_op} can\'t be processed by Mch {chosen_mch}')
            sys.exit()

        self.step_count += 1

        # update candidate
        candidate_add_flag = (chosen_op != self.job_last_op_id[self.incomplete_env_idx, chosen_job])
        self.candidate[self.incomplete_env_idx, chosen_job] += candidate_add_flag
        self.mask[self.incomplete_env_idx, chosen_job] = (1 - candidate_add_flag)

        self.mch_queue[
            self.incomplete_env_idx, chosen_mch, self.mch_queue_len[self.incomplete_env_idx, chosen_mch]] = chosen_op

        self.mch_queue_len[self.incomplete_env_idx, chosen_mch] += 1

        # [E]
        chosen_op_st = np.maximum(self.candidate_free_time[self.incomplete_env_idx, chosen_job],
                                  self.mch_free_time[self.incomplete_env_idx, chosen_mch])

        self.op_ct[self.incomplete_env_idx, chosen_op] = chosen_op_st + self.op_pt[
            self.incomplete_env_idx, chosen_op, chosen_mch]
        self.candidate_free_time[self.incomplete_env_idx, chosen_job] = self.op_ct[self.incomplete_env_idx, chosen_op]
        self.mch_free_time[self.incomplete_env_idx, chosen_mch] = self.op_ct[self.incomplete_env_idx, chosen_op]

        true_chosen_op_st = np.maximum(self.true_candidate_free_time[self.incomplete_env_idx, chosen_job],
                                       self.true_mch_free_time[self.incomplete_env_idx, chosen_mch])
        self.true_op_ct[self.incomplete_env_idx, chosen_op] = true_chosen_op_st + self.true_op_pt[
            self.incomplete_env_idx, chosen_op, chosen_mch]
        self.true_candidate_free_time[self.incomplete_env_idx, chosen_job] = self.true_op_ct[
            self.incomplete_env_idx, chosen_op]
        self.true_mch_free_time[self.incomplete_env_idx, chosen_mch] = self.true_op_ct[
            self.incomplete_env_idx, chosen_op]

        self.current_makespan[self.incomplete_env_idx] = np.maximum(self.current_makespan[self.incomplete_env_idx],
                                                                    self.true_op_ct[
                                                                        self.incomplete_env_idx, chosen_op])
        
        self.tasks_data.append({"Task": f"Job-{chosen_job[0]}", "Station": f"M{chosen_mch[0]}", "Start": true_chosen_op_st[0], "Duration": self.true_op_pt[
            0, chosen_op, chosen_mch][0], "Width": 0.4})

        for k, j in enumerate(self.incomplete_env_idx):
            if candidate_add_flag[k]:
                self.candidate_pt[j, chosen_job[k]] = self.unmasked_op_pt[j, chosen_op[k] + 1]
                self.candidate_process_relation[j, chosen_job[k]] = self.reverse_process_relation[j, chosen_op[k] + 1]
            else:
                self.candidate_process_relation[j, chosen_job[k]] = 1

        candidateFT_for_compare = np.expand_dims(self.candidate_free_time, axis=2)
        mchFT_for_compare = np.expand_dims(self.mch_free_time, axis=1)
        self.pair_free_time = np.maximum(candidateFT_for_compare, mchFT_for_compare)

        pair_free_time = self.pair_free_time[self.incomplete_env_idx]
        schedule_matrix = ma.array(pair_free_time, mask=self.candidate_process_relation[self.incomplete_env_idx])

        self.next_schedule_time[self.incomplete_env_idx] = np.min(
            schedule_matrix.reshape(self.number_of_incomplete_envs, -1), axis=1).data

        self.remain_process_relation[self.incomplete_env_idx, chosen_op] = 0
        self.op_scheduled_flag[self.incomplete_env_idx, chosen_op] = 1

        self.deleted_op_nodes = \
            np.logical_and((self.op_ct <= self.next_schedule_time[:, np.newaxis]),
                           self.op_scheduled_flag)

        self.delete_mask_fea_j = np.tile(self.deleted_op_nodes[:, :, np.newaxis],
                                         (1, 1, self.op_fea_dim))

        self.update_op_mask()

        self.mch_queue_last_op_id[self.incomplete_env_idx, chosen_mch] = chosen_op

        self.unscheduled_op_nums[self.incomplete_env_idx] -= 1

        diff = self.op_ct[self.incomplete_env_idx, chosen_op] - self.op_ct_lb[self.incomplete_env_idx, chosen_op]
        for k, j in enumerate(self.incomplete_env_idx):
            self.op_ct_lb[j][chosen_op[k]:self.job_last_op_id[j, chosen_job[k]] + 1] += diff[k]
            self.op_match_job_left_op_nums[j][
                self.job_first_op_id[j, chosen_job[k]]:self.job_last_op_id[j, chosen_job[k]] + 1] -= 1
            self.op_match_job_remain_work[j][
                self.job_first_op_id[j, chosen_job[k]]:self.job_last_op_id[j, chosen_job[k]] + 1] -= \
                self.op_mean_pt[j, chosen_op[k]]

        self.op_waiting_time = np.zeros((self.number_of_envs, self.max_number_of_ops))
        self.op_waiting_time[self.env_job_idx, self.candidate] = \
            (1 - self.mask) * np.maximum(np.expand_dims(self.next_schedule_time, axis=1)
                                         - self.candidate_free_time, 0) + self.mask * self.op_waiting_time[
                self.env_job_idx, self.candidate]

        self.op_remain_work = np.maximum(self.op_ct -
                                         np.expand_dims(self.next_schedule_time, axis=1), 0)

        self.construct_op_features()

        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)

        self.unavailable_pairs = np.array([pair_free_time[k] > self.next_schedule_time[j]
                                           for k, j in enumerate(self.incomplete_env_idx)])
        self.dynamic_pair_mask[self.incomplete_env_idx] = np.logical_or(self.dynamic_pair_mask[self.incomplete_env_idx],
                                                                        self.unavailable_pairs)

        self.comp_idx = self.logic_operator(~self.dynamic_pair_mask)

        self.update_mch_mask()

        # mch features update
        self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=1)
        self.mch_current_available_op_nums[self.incomplete_env_idx] -= self.process_relation[
            self.incomplete_env_idx, chosen_op]

        mch_free_duration = np.expand_dims(self.next_schedule_time[self.
                                           incomplete_env_idx], axis=1) - self.mch_free_time[self.incomplete_env_idx]
        mch_free_flag = mch_free_duration < 0
        self.mch_working_flag[self.incomplete_env_idx] = mch_free_flag + 0
        self.mch_waiting_time[self.incomplete_env_idx] = (1 - mch_free_flag) * mch_free_duration

        self.mch_remain_work[self.incomplete_env_idx] = np.maximum(-mch_free_duration, 0)

        self.construct_mch_features()

        self.construct_pair_features()

        reward = self.max_endTime - np.max(self.op_ct_lb, axis=1)
        self.max_endTime = np.max(self.op_ct_lb, axis=1)

        self.state.update(self.fea_j, self.op_mask, self.fea_m, self.mch_mask,
                          self.dynamic_pair_mask, self.comp_idx, self.candidate,
                          self.fea_pairs)
        self.done_flag = self.done()

        return self.state, np.array(reward), self.done_flag

    def done(self):
        return np.all(self.dynamic_pair_mask, axis=(1,2)) or \
        self.step_count >= self.env_number_of_ops

    def construct_op_features(self):

        self.fea_j = np.stack((self.op_scheduled_flag,
                               self.op_ct_lb,
                               self.op_min_pt,
                               self.pt_span,
                               self.op_mean_pt,
                               self.op_waiting_time,
                               self.op_remain_work,
                               self.op_match_job_left_op_nums,
                               self.op_match_job_remain_work,
                               self.op_available_mch_nums), axis=2)

        if self.flag_exist_dummy_node:
            mask_all = np.logical_or(self.dummy_mask_fea_j, self.delete_mask_fea_j)
        else:
            mask_all = self.delete_mask_fea_j

        self.norm_operation_features(mask=mask_all)

    def norm_operation_features(self, mask):
        self.fea_j[mask] = 0
        num_delete_nodes = np.count_nonzero(mask[:, :, 0], axis=1)

        num_delete_nodes = num_delete_nodes[:, np.newaxis]
        num_left_nodes = self.max_number_of_ops - num_delete_nodes

        num_left_nodes = np.maximum(num_left_nodes, 1e-8)

        mean_fea_j = np.sum(self.fea_j, axis=1) / num_left_nodes

        temp = np.where(self.delete_mask_fea_j,
                        mean_fea_j[:, np.newaxis, :], self.fea_j)
        var_fea_j = np.var(temp, axis=1)

        std_fea_j = np.sqrt(var_fea_j * self.max_number_of_ops / num_left_nodes)

        self.fea_j = ((temp - mean_fea_j[:, np.newaxis, :]) / \
                      (std_fea_j[:, np.newaxis, :] + 1e-8))

    def construct_mch_features(self):

        self.fea_m = np.stack((self.mch_current_available_jc_nums,
                               self.mch_current_available_op_nums,
                               self.mch_min_pt,
                               self.mch_mean_pt,
                               self.mch_waiting_time,
                               self.mch_remain_work,
                               self.mch_free_time,
                               self.mch_working_flag), axis=2)

        self.norm_machine_features()

    def norm_machine_features(self):
        self.fea_m[self.delete_mask_fea_m] = 0
        num_delete_mchs = np.count_nonzero(self.delete_mask_fea_m[:, :, 0], axis=1)
        num_delete_mchs = num_delete_mchs[:, np.newaxis]
        num_left_mchs = self.number_of_machines - num_delete_mchs

        num_left_mchs = np.maximum(num_left_mchs, 1e-8)

        mean_fea_m = np.sum(self.fea_m, axis=1) / num_left_mchs

        temp = np.where(self.delete_mask_fea_m,
                        mean_fea_m[:, np.newaxis, :], self.fea_m)
        var_fea_m = np.var(temp, axis=1)

        std_fea_m = np.sqrt(var_fea_m * self.number_of_machines / num_left_mchs)

        self.fea_m = ((temp - mean_fea_m[:, np.newaxis, :]) / \
                      (std_fea_m[:, np.newaxis, :] + 1e-8))

    def construct_pair_features(self):

        remain_op_pt = ma.array(self.op_pt, mask=~self.remain_process_relation)

        chosen_op_max_pt = np.expand_dims(self.op_max_pt[self.env_job_idx, self.candidate], axis=-1)

        max_remain_op_pt = np.max(np.max(remain_op_pt, axis=1, keepdims=True), axis=2, keepdims=True) \
            .filled(0 + 1e-8)

        mch_max_remain_op_pt = np.max(remain_op_pt, axis=1, keepdims=True). \
            filled(0 + 1e-8)

        pair_max_pt = np.max(np.max(self.candidate_pt, axis=1, keepdims=True),
                             axis=2, keepdims=True) + 1e-8

        mch_max_candidate_pt = np.max(self.candidate_pt, axis=1, keepdims=True) + 1e-8

        pair_wait_time = self.op_waiting_time[self.env_job_idx, self.candidate][:, :,
                         np.newaxis] + self.mch_waiting_time[:, np.newaxis, :]

        chosen_job_remain_work = np.expand_dims(self.op_match_job_remain_work
                                                [self.env_job_idx, self.candidate],
                                                axis=-1) + 1e-8

        self.fea_pairs = np.stack((self.candidate_pt,
                                   self.candidate_pt / chosen_op_max_pt,
                                   self.candidate_pt / mch_max_candidate_pt,
                                   self.candidate_pt / max_remain_op_pt,
                                   self.candidate_pt / mch_max_remain_op_pt,
                                   self.candidate_pt / pair_max_pt,
                                   self.candidate_pt / chosen_job_remain_work,
                                   pair_wait_time), axis=-1)

    def update_mch_mask(self):

        self.mch_mask = self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        self.delete_mask_fea_m = np.tile(~(np.sum(self.mch_mask, keepdims=True, axis=-1).astype(bool)),
                                         (1, 1, self.mch_fea_dim))
        self.mch_mask[self.multi_env_mch_diag] = 1

    def init_mch_mask(self):

        self.mch_mask = self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        self.delete_mask_fea_m = np.tile(~(np.sum(self.mch_mask, keepdims=True, axis=-1).astype(bool)),
                                         (1, 1, self.mch_fea_dim))
        self.mch_mask[self.multi_env_mch_diag] = 1

    def init_op_mask(self):
        self.op_mask = np.full(shape=(self.number_of_envs, self.max_number_of_ops, 3),
                               fill_value=0, dtype=np.float32)
        self.op_mask[self.env_job_idx, self.job_first_op_id, 0] = 1
        self.op_mask[self.env_job_idx, self.job_last_op_id, 2] = 1

    def update_op_mask(self):
        object_mask = np.zeros_like(self.op_mask)
        object_mask[:, :, 2] = self.deleted_op_nodes
        object_mask[:, 1:, 0] = self.deleted_op_nodes[:, :-1]
        self.op_mask = np.logical_or(object_mask, self.op_mask).astype(np.float32)

    def logic_operator(self, x, flagT=True):
        if flagT:
            x = x.transpose(0, 2, 1)
        d1 = np.expand_dims(x, 2)
        d2 = np.expand_dims(x, 1)

        return np.logical_and(d1, d2).astype(np.float32)
    
    def render(self):
        '''tasks_data = [
        {"Task": "Job1", "Station": "M1", "Start": 0, "Duration": 4, "Width": 0.4},
        {"Task": "Job2", "Station": "M2", "Start": 5, "Duration": 3, "Width": 0.4},
        {"Task": "Job3", "Station": "M3", "Start": 9, "Duration": 2, "Width": 0.4},
        ...
        ]   '''
        # 获取唯一的Job名称列表
        unique_jobs = sorted(list(set(task['Task'] for task in self.tasks_data)), key=lambda x: int(x[3:]))

        # 使用Plotly的定性颜色循环作为基础
        color_sequence = qualitative.Plotly

        # 如果Job数量超过内置颜色，动态生成更多颜色
        if len(unique_jobs) > len(color_sequence):
            extra_colors_needed = len(unique_jobs) - len(color_sequence)
            # 动态生成新颜色
            random.seed(42)  # 固定种子以保证结果可复现
            extra_colors = []
            for _ in range(extra_colors_needed):
                h, s, l = random.random(), 0.5 + random.random() / 2.0, 0.4 + random.random() / 5.0
                r, g, b = colorsys.hls_to_rgb(h, l, s)
                extra_colors.append(f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})")
            color_sequence = color_sequence + extra_colors

        # 创建颜色映射
        color_map = {job: color for job, color in zip(unique_jobs, color_sequence)}

        # 清空现有的绘图数据
        self.fig.data = []

        # 绘制条形图
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
            # title="Schedule result",
            xaxis_title="Time",
            yaxis=dict(
                type='category',
                categoryorder='array',
                categoryarray=[task['Station'] for task in self.tasks_data]
            ),
            barmode='stack',
            legend_title="Job",
        )

        # 显示图表
        self.fig.show()

    def remain_work_status(self):
        """
        获取当前时刻的环境状态，返回剩余工件操作数和 op_pt_dict。
        """
        # 计算每个工件剩余的待调度工序数
        remaining_ops_list = []
        op_pt_list = []
        for env_idx in range(self.number_of_envs):

            remaining_ops = self.job_last_op_id[env_idx] - self.candidate[env_idx]

            op_pt = []

            for job_id in range(len(self.job_length[env_idx])):
                cnt = 0
                for op_id in range(self.candidate[env_idx][job_id], self.job_last_op_id[env_idx][job_id] + 1):
                    if self.op_scheduled_flag[env_idx][op_id] == 0:
                        op_pt.append(list(self.true_op_pt[env_idx][op_id].data))  # 将时间加入字典
                        cnt += 1
                remaining_ops[job_id] = cnt
            remaining_ops_list.append(remaining_ops)
            op_pt_list.append(op_pt)
        return np.array(remaining_ops_list), np.array(op_pt, dtype=np.float32)

    def mask_machine(self, chose_env_idx,  mch_id):
        op_pt_list = self.op_pt[chose_env_idx].data
        self.old_op_pt_list = op_pt_list.copy()
        op_pt_list[:, mch_id] = 0
        zero_row_indices = np.where(np.all(op_pt_list == 0, axis=1))[0]
        # Determine which job each index belongs to
        #检查op_pt_list是否有全0的行，如果有，把这一行的id计算出来
        task_ids = []
        for idx in zero_row_indices:
            task_id = np.where((self.job_first_op_id[chose_env_idx] <= idx) & (self.job_last_op_id[chose_env_idx] >= idx))[0]
            self.candidate_process_relation[chose_env_idx, task_id, :] = 1
            self.process_relation[chose_env_idx, task_id, :] = 0
            self.remain_process_relation[chose_env_idx, task_id, :] = 0
            task_ids.append(task_id[0] if len(task_id) > 0 else None)
        self.old_process_relation = self.process_relation.copy()
        self.process_relation[chose_env_idx, :, mch_id] = 0
        self.reverse_process_relation = ~self.process_relation
        # 把self.op_pt的chose_env_idx改为op_pt_list，并且重新mask
        self.op_pt.data[chose_env_idx] = op_pt_list  # 替换数据
        self.op_pt.mask[chose_env_idx] = self.reverse_process_relation[chose_env_idx]  # 更新掩码
        self.remain_process_relation[chose_env_idx, :, mch_id] = 0
        self.candidate_process_relation[chose_env_idx, :, mch_id] = 1
        # self.candidate_free_time
        # self.mch_free_time[chose_env_idx, mch_id] = 1000

        return {"unavaliable job id":task_ids}
    
    def unmask_machine(self):
        op_pt_list = self.old_op_pt_list
        self.process_relation = self.old_process_relation
        self.reverse_process_relation = ~self.process_relation
        # 把self.op_pt的chose_env_idx改为op_pt_list，并且重新mask
        self.op_pt.data[0] = op_pt_list  # 替换数据
        self.op_pt.mask[0] = self.reverse_process_relation[0]  # 更新掩码
        # self.remain_process_relation[0, :, mch_id] = 0
        # self.candidate_process_relation[0, :, mch_id] = 1
        # self.candidate_free_time

    def insert_new_jobs_gpt(self, new_job_length_list, new_op_pt_list):
        """
        插入新工件/作业：
        Args:
            new_job_length_list: 新插入的工件各自操作数量列表，形如 [array([op_num_of_jobX, ...]), ...]
                                如果只插 1 个环境，可以是 [array([...])]
            new_op_pt_list: 新插入工件的加工时间矩阵列表（未归一化），形如：
                            [
                            array([[op1_m1, op1_m2,...],   # 第一道工序对每台机器的加工时间
                                    [op2_m1, op2_m2,...],
                                    ...]),
                            ...
                            ]
                            同样，如果只插 1 个环境，可对应一个 array([...]) 即可。
        """
        # 0. 假设 new_job_length_list 和 new_op_pt_list 与当前已有环境数 self.number_of_envs 一一对应
        #    即 new_job_length_list[k] 和 new_op_pt_list[k] 对应同一个环境
        #    如果只有单环境，也可以自行简化此部分逻辑

        # ---------------------------
        # 1. 记录当前调度进度和基础信息
        # ---------------------------
        old_number_of_jobs = self.number_of_jobs
        old_env_number_of_ops = self.env_number_of_ops.copy()        # [E]
        old_max_number_of_ops = self.max_number_of_ops
        old_job_length = self.job_length.copy()                      # [E, N]
        old_virtual_job_length = self.virtual_job_length.copy()      # [E, N]

        # 记录目前的 candidate 等关键动态属性，以便后续扩展 shape
        old_candidate = self.candidate.copy()                        # [E, old_number_of_jobs]
        old_candidate_pt = self.candidate_pt.copy()                  # [E, old_number_of_jobs]
        old_candidate_process_relation = self.candidate_process_relation.copy()  # [E, old_number_of_jobs]
        
        # 记录部分需要展开的动态属性(比如这些会涉及到维度 [E, max_number_of_ops], [E, N], [E, N, M]等)
        old_op_ct = self.op_ct.copy()           # [E, old_max_number_of_ops]
        old_true_op_ct = self.true_op_ct.copy()
        old_op_scheduled_flag = self.op_scheduled_flag.copy()
        old_op_mask = self.op_mask.copy()       # [E, old_max_number_of_ops]
        old_deleted_op_nodes = self.deleted_op_nodes.copy()  # [E, old_max_number_of_ops]

        # 机器相关的一些属性
        old_mch_queue = self.mch_queue.copy()                   # [E, M, old_max_number_of_ops+1]
        old_mch_queue_len = self.mch_queue_len.copy()           # [E, M]
        old_mch_queue_last_op_id = self.mch_queue_last_op_id.copy()
        old_mch_free_time = self.mch_free_time.copy()           # [E, M]
        old_true_mch_free_time = self.true_mch_free_time.copy()
        old_mch_remain_work = self.mch_remain_work.copy()
        old_mch_mask = self.mch_mask.copy()                     # [E, M, ...] 具体形状见初始化

        # 其他动态属性
        old_unscheduled_op_nums = self.unscheduled_op_nums.copy()
        old_remain_process_relation = self.remain_process_relation.copy()  # [E, max_number_of_ops, M] (或 [E, max_number_of_ops]?)
        old_op_waiting_time = self.op_waiting_time.copy()
        old_op_remain_work = self.op_remain_work.copy()
        old_mask = self.mask.copy()                  # [E, old_number_of_jobs]
        old_candidate_free_time = self.candidate_free_time.copy()    # [E, old_number_of_jobs]
        old_true_candidate_free_time = self.true_candidate_free_time.copy()
        
        # ---------------------------
        # 2. 解析并更新新插入作业的操作数量等
        # ---------------------------
        # 2.1 将新工件的数量合并到原工件数上
        #     这里假设所有环境一次性同时插入 new_jobs_count 个新工件
        #     也可对每个 environment 分别统计
        for env_idx in range(self.number_of_envs):
            # 新增的工件数
            new_jobs_count = len(new_job_length_list[env_idx])
            # 原工件数 + 新工件数
            total_jobs = old_number_of_jobs + new_jobs_count

            # 拼接 job_length
            # 比如原 job_length[env_idx] shape = (old_number_of_jobs,)
            # 新增部分 shape = (new_jobs_count,)
            self.job_length[env_idx] = np.concatenate(
                [self.job_length[env_idx], new_job_length_list[env_idx]]
            )
            self.virtual_job_length[env_idx] = np.concatenate(
                [self.virtual_job_length[env_idx], new_job_length_list[env_idx]]
            )

            # 重新计算 env_number_of_ops
            # 原本 env_number_of_ops[env_idx] = sum of old jobs
            # 新的 = old_env_number_of_ops[env_idx] + sum_of_new
            self.env_number_of_ops[env_idx] += np.sum(new_job_length_list[env_idx])

        self.number_of_jobs = old_number_of_jobs + len(new_job_length_list[0])  # 假设所有 env 新增作业数相同
        self.max_number_of_ops = np.max(self.env_number_of_ops)

        # ---------------------------
        # 3. 扩展各类关键数组(与操作数相关)
        # ---------------------------
        # 3.1 扩展 self.op_pt, self.true_op_pt
        #     需要把新工件的加工时间拼接到 op_pt 最后一部分
        #     因为 op_pt shape = [E, max_number_of_ops, M]
        old_op_pt = self.op_pt.copy()         # [E, old_max_number_of_ops, M]
        old_true_op_pt = self.true_op_pt.copy()

        # 先构造新的 self.op_pt 和 self.true_op_pt，shape = [E, new_max_number_of_ops, M]
        new_op_pt = np.zeros((self.number_of_envs, self.max_number_of_ops, self.number_of_machines))
        new_true_op_pt = np.zeros((self.number_of_envs, self.max_number_of_ops, self.number_of_machines))

        # 填充之前已有的部分
        new_op_pt[:, :old_max_number_of_ops, :] = old_op_pt
        new_true_op_pt[:, :old_max_number_of_ops, :] = old_true_op_pt

        # 3.2 把新插入工件的加工时间放到相应的后面区域
        #     需要根据每个 environment 新增操作的数量和顺序依次填充
        for env_idx in range(self.number_of_envs):
            # 计算原先末尾操作ID = old_env_number_of_ops[env_idx]
            original_ops_end = old_env_number_of_ops[env_idx]
            # 新增工件的总操作数
            new_ops_count = np.sum(new_job_length_list[env_idx])

            # 直接把 new_op_pt_list[env_idx] 贴到 new_true_op_pt 对应区域
            # new_op_pt_list[env_idx] shape = [ sum_of_new_ops, M ]
            # sum_of_new_ops = new_ops_count
            new_true_op_pt[env_idx, original_ops_end:original_ops_end + new_ops_count, :] = new_op_pt_list[env_idx]

            # 对于归一化部分，如果不想影响旧工序，可只对新增工序做相对归一化
            # 这里演示“简单复用之前的 min/max”对新工序处理，或你也可自行设计
            pt_lower_bound = self.pt_lower_bound
            pt_upper_bound = self.pt_upper_bound
            epsilon = 1e-6
            # 仅对大于0的加工时间做归一化
            mask_non_zero = new_true_op_pt[env_idx, original_ops_end:original_ops_end + new_ops_count, :] > 0
            block_data = new_true_op_pt[env_idx, original_ops_end:original_ops_end + new_ops_count, :]
            norm_block = epsilon + (1 - epsilon) * (block_data - pt_lower_bound) / (pt_upper_bound - pt_lower_bound + 1e-8)
            # 原地更新
            block_data[mask_non_zero] = norm_block[mask_non_zero]

        self.op_pt = new_op_pt
        self.true_op_pt = new_true_op_pt

        # ---------------------------
        # 4. 扩展操作相关的调度动态属性
        #    如 op_ct, op_scheduled_flag, remain_process_relation, ...
        # ---------------------------
        # 4.1 op_ct, true_op_ct
        new_op_ct = np.zeros((self.number_of_envs, self.max_number_of_ops))
        new_true_op_ct = np.zeros((self.number_of_envs, self.max_number_of_ops))
        # 保留旧数据
        new_op_ct[:, :old_max_number_of_ops] = old_op_ct
        new_true_op_ct[:, :old_max_number_of_ops] = old_true_op_ct
        self.op_ct = new_op_ct
        self.true_op_ct = new_true_op_ct

        # op_scheduled_flag
        new_op_scheduled_flag = np.zeros((self.number_of_envs, self.max_number_of_ops))
        new_op_scheduled_flag[:, :old_max_number_of_ops] = old_op_scheduled_flag
        self.op_scheduled_flag = new_op_scheduled_flag

        # 删除节点等标记
        new_deleted_op_nodes = np.zeros((self.number_of_envs, self.max_number_of_ops), dtype=bool)
        new_deleted_op_nodes[:, :old_max_number_of_ops] = old_deleted_op_nodes
        self.deleted_op_nodes = new_deleted_op_nodes

        # remain_process_relation 需要扩展到 shape = [E, max_number_of_ops, M]
        new_remain_process_relation = np.zeros((self.number_of_envs, self.max_number_of_ops, self.number_of_machines),
                                            dtype=bool)
        new_remain_process_relation[:, :old_max_number_of_ops, :] = old_remain_process_relation
        # 新增工序默认还没被调度，所以 remain_process_relation 需要根据 new_true_op_pt 是否 0 来判断
        for env_idx in range(self.number_of_envs):
            # 重新用 (self.true_op_pt != 0) 更新 remain_process_relation
            new_remain_process_relation[env_idx] = (self.true_op_pt[env_idx] != 0)
        self.remain_process_relation = new_remain_process_relation

        # op_mask
        new_op_mask = np.zeros((self.number_of_envs, self.max_number_of_ops), dtype=bool)
        new_op_mask[:, :old_max_number_of_ops] = old_op_mask
        self.op_mask = new_op_mask

        # op_waiting_time, op_remain_work
        new_op_waiting_time = np.zeros((self.number_of_envs, self.max_number_of_ops))
        new_op_waiting_time[:, :old_max_number_of_ops] = old_op_waiting_time
        self.op_waiting_time = new_op_waiting_time

        new_op_remain_work = np.zeros((self.number_of_envs, self.max_number_of_ops))
        new_op_remain_work[:, :old_max_number_of_ops] = old_op_remain_work
        self.op_remain_work = new_op_remain_work

        # 更新 unscheduled_op_nums
        new_unscheduled_op_nums = old_unscheduled_op_nums.copy()
        for env_idx in range(self.number_of_envs):
            new_unscheduled_op_nums[env_idx] += np.sum(new_job_length_list[env_idx])  # 新增操作数
        self.unscheduled_op_nums = new_unscheduled_op_nums

        # ---------------------------
        # 5. 扩展 job_first_op_id, job_last_op_id
        #    需要考虑新作业对应的第一道和最后一道工序编号
        # ---------------------------
        new_job_first_op_id = np.zeros((self.number_of_envs, self.number_of_jobs), dtype=int)
        new_job_last_op_id = np.zeros((self.number_of_envs, self.number_of_jobs), dtype=int)

        # 保留之前的
        new_job_first_op_id[:, :old_number_of_jobs] = self.job_first_op_id
        new_job_last_op_id[:, :old_number_of_jobs] = self.job_last_op_id

        for env_idx in range(self.number_of_envs):
            # 旧作业部分保持不变
            # 从 old_number_of_jobs 之后开始给新作业分配工序 ID
            start_op_id = old_job_last_op_id[env_idx, -1] + 1  # 旧最后一道工序ID+1，即新作业起始工序ID
            for i, length_val in enumerate(new_job_length_list[env_idx]):
                job_idx = old_number_of_jobs + i
                new_job_first_op_id[env_idx, job_idx] = start_op_id
                new_job_last_op_id[env_idx, job_idx] = start_op_id + length_val - 1
                start_op_id += length_val

        self.job_first_op_id = new_job_first_op_id
        self.job_last_op_id = new_job_last_op_id

        # ---------------------------
        # 6. 扩展 candidate 和 mask 等 [E, N] 形状的属性
        # ---------------------------
        # candidate
        new_candidate = np.full((self.number_of_envs, self.number_of_jobs), -1, dtype=int)
        new_candidate[:, :old_number_of_jobs] = old_candidate
        # 对于新工件，其当前 candidate 即它的首道工序 ID
        for env_idx in range(self.number_of_envs):
            for i, length_val in enumerate(new_job_length_list[env_idx]):
                job_idx = old_number_of_jobs + i
                new_candidate[env_idx, job_idx] = self.job_first_op_id[env_idx, job_idx]
        self.candidate = new_candidate

        # mask
        new_mask = np.zeros((self.number_of_envs, self.number_of_jobs), dtype=bool)
        new_mask[:, :old_number_of_jobs] = old_mask
        # 新插入工件默认还未完成，mask = 0
        self.mask = new_mask

        # candidate_free_time
        new_candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))
        new_true_candidate_free_time = np.zeros((self.number_of_envs, self.number_of_jobs))
        new_candidate_free_time[:, :old_number_of_jobs] = old_candidate_free_time
        new_true_candidate_free_time[:, :old_number_of_jobs] = old_true_candidate_free_time
        # 新工件暂时可视为立即可排(初始时刻等同当前 next_schedule_time)，或者根据实际逻辑设置
        self.candidate_free_time = new_candidate_free_time
        self.true_candidate_free_time = new_true_candidate_free_time

        # candidate_pt, candidate_process_relation
        new_candidate_pt = np.zeros((self.number_of_envs, self.number_of_jobs))
        new_candidate_pt[:, :old_number_of_jobs] = old_candidate_pt
        self.candidate_pt = new_candidate_pt

        new_candidate_process_relation = np.ones((self.number_of_envs, self.number_of_jobs), dtype=bool)
        # 对旧作业的关系还原
        new_candidate_process_relation[:, :old_number_of_jobs] = old_candidate_process_relation
        # 新插入工件的首道工序暂定 relation=False(表示它可被机器加工，只要 machine 对应 pt 不为 0)
        # 因为与 remain_process_relation 配合时，需要把 candidate_process_relation 置为 false
        # 才表示这道工序可以调度
        for env_idx in range(self.number_of_envs):
            for i, length_val in enumerate(new_job_length_list[env_idx]):
                job_idx = old_number_of_jobs + i
                first_op_id = self.job_first_op_id[env_idx, job_idx]
                # 如果某台机器 pt>0，说明该操作可以加工
                # 这里的逻辑与 step 中 dynamic_pair_mask 相关，可以根据自身需要进行初始化
                if np.sum(self.true_op_pt[env_idx, first_op_id, :]) > 0:
                    new_candidate_process_relation[env_idx, job_idx] = False
        self.candidate_process_relation = new_candidate_process_relation

        # ---------------------------
        # 7. 扩展机器相关属性（如果需要）
        # ---------------------------
        # mch_queue [E, M, max_number_of_ops+1]
        new_mch_queue = np.full((self.number_of_envs, self.number_of_machines,
                                self.max_number_of_ops + 1), -99, dtype=int)
        new_mch_queue_len = np.zeros((self.number_of_envs, self.number_of_machines), dtype=int)
        new_mch_queue_last_op_id = np.zeros((self.number_of_envs, self.number_of_machines), dtype=int)
        # 保留旧数据
        new_mch_queue[:, :, :old_max_number_of_ops + 1] = old_mch_queue
        new_mch_queue_len[:, :] = old_mch_queue_len
        new_mch_queue_last_op_id[:, :] = old_mch_queue_last_op_id
        self.mch_queue = new_mch_queue
        self.mch_queue_len = new_mch_queue_len
        self.mch_queue_last_op_id = new_mch_queue_last_op_id

        # mch_free_time, true_mch_free_time, mch_remain_work
        new_mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))
        new_true_mch_free_time = np.zeros((self.number_of_envs, self.number_of_machines))
        new_mch_remain_work = np.zeros((self.number_of_envs, self.number_of_machines))
        new_mch_free_time[:, :] = old_mch_free_time
        new_true_mch_free_time[:, :] = old_true_mch_free_time
        new_mch_remain_work[:, :] = old_mch_remain_work

        self.mch_free_time = new_mch_free_time
        self.true_mch_free_time = new_true_mch_free_time
        self.mch_remain_work = new_mch_remain_work

        # mch_mask
        new_mch_mask = np.copy(old_mch_mask)  # 如果 mch_mask 不依赖操作数，可以不变
        self.mch_mask = new_mch_mask

        # ---------------------------
        # 8. 重新构造或更新特征相关的数据结构
        #    (fea_j, fea_m, fea_pairs, comp_idx, dynamic_pair_mask, 等)
        #    可以复用已有的构造函数，也可部分更新
        # ---------------------------
        self.update_op_mask()
        self.update_mch_mask()

        # 重新构造或部分更新特征
        self.construct_op_features()
        self.construct_mch_features()
        self.construct_pair_features()

        # comp_idx, dynamic_pair_mask
        # 由于 candidate_process_relation 重置了一部分，这里重新计算
        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)
        self.comp_idx = self.logic_operator(~self.dynamic_pair_mask)

        # 更新老状态 old_state
        # 扩展后的状态才能重新保存
        self.state.update(self.fea_j, self.op_mask,
                        self.fea_m, self.mch_mask,
                        self.dynamic_pair_mask, self.comp_idx, self.candidate,
                        self.fea_pairs)

        # 最后也可以把这份更新后的 state 存到 old_* 里，以免 reset() 等地方需要用到
        self.old_state = copy.deepcopy(self.state)

        # 重新计算新的 init_quality 或者保持旧值
        # 视需求而定，这里简单示例直接更新
        self.init_quality = np.max(self.op_ct_lb, axis=1)
        self.max_endTime = np.max(self.op_ct_lb, axis=1)

        print("新工件插入完毕，当前作业总数 = {}, 最大操作总数 = {}".format(
            self.number_of_jobs, self.max_number_of_ops
        ))

    def insert_new_jobs_ds(self, new_job_length_list, new_op_pt_list):
        # 合并新旧工件的 job_length 和 op_pt
        old_number_of_jobs = self.number_of_jobs
        self.job_length = np.concatenate([self.job_length, np.array(new_job_length_list)], axis=1)
        self.number_of_jobs = self.job_length.shape[1]

        # 合并每个环境的 op_pt，并更新 env_number_of_ops 和 max_number_of_ops
        new_env_op_pt_list = []
        for env_idx in range(self.number_of_envs):
            # 提取原始有效数据（去除填充）
            original_op_pt = self.true_op_pt[env_idx][:self.env_number_of_ops[env_idx], :]
            # 合并新工件的 op_pt
            combined_op_pt = np.concatenate([original_op_pt, new_op_pt_list[env_idx]], axis=0)
            new_env_op_pt_list.append(combined_op_pt)
        
        # 更新每个环境的总工序数
        self.env_number_of_ops = np.array([op_pt.shape[0] for op_pt in new_env_op_pt_list])
        self.max_number_of_ops = np.max(self.env_number_of_ops)
        
        # 对合并后的 op_pt 进行填充，使其形状为 [E, max_number_of_ops, M]
        padded_op_pt = []
        for env_idx in range(self.number_of_envs):
            pad_height = self.max_number_of_ops - new_env_op_pt_list[env_idx].shape[0]
            padded = np.pad(new_env_op_pt_list[env_idx], 
                            ((0, pad_height), (0, 0)), 
                            mode='constant', constant_values=0)
            padded_op_pt.append(padded)
        self.true_op_pt = np.array(padded_op_pt)
        
        # 重新归一化处理时间
        non_zero_mask = (self.true_op_pt > 0)
        normalized_op_pt = 1e-6 + (1 - 1e-6) * (self.true_op_pt - self.pt_lower_bound) / (self.pt_upper_bound - self.pt_lower_bound + 1e-8)
        self.op_pt = np.where(non_zero_mask, normalized_op_pt, 0)
        self.op_pt = ma.array(self.op_pt, mask=self.reverse_process_relation)

        # 重新生成静态属性（如 job_first_op_id）
        self.set_static_properties()

        # 扩展动态状态数组并初始化新工件的状态
        # 1. candidate：新工件的初始候选操作是其第一个工序
        new_candidate = np.zeros((self.number_of_envs, self.number_of_jobs), dtype=int)
        new_candidate[:, :old_number_of_jobs] = self.candidate  # 保留原有状态
        for env_idx in range(self.number_of_envs):
            for job_idx in range(old_number_of_jobs, self.number_of_jobs):
                # 新工件的第一个工序的ID是前面所有工序的总和
                new_candidate[env_idx, job_idx] = np.sum(self.job_length[env_idx, :job_idx])
        self.candidate = new_candidate

        # 2. mask：新工件的 mask 初始化为未完成状态
        new_mask = np.zeros((self.number_of_envs, self.number_of_jobs), dtype=bool)
        new_mask[:, :old_number_of_jobs] = self.mask
        for env_idx in range(self.number_of_envs):
            for job_idx in range(old_number_of_jobs, self.number_of_jobs):
                job_length = self.job_length[env_idx, job_idx]
                # 如果新工件只有一个工序，则调度后立即标记为完成
                new_mask[env_idx, job_idx] = (job_length == 0)  # 假设 job_length=0 表示无效
        self.mask = new_mask

        # 3. 扩展 op_ct、op_scheduled_flag 等数组
        new_op_ct = np.zeros((self.number_of_envs, self.max_number_of_ops))
        new_op_ct[:, :self.op_ct.shape[1]] = self.op_ct
        self.op_ct = new_op_ct

        new_op_scheduled_flag = np.zeros((self.number_of_envs, self.max_number_of_ops), dtype=bool)
        new_op_scheduled_flag[:, :self.op_scheduled_flag.shape[1]] = self.op_scheduled_flag
        self.op_scheduled_flag = new_op_scheduled_flag

        # 更新工序与工件的映射关系
        self.op_match_job_left_op_nums = np.array([
            np.repeat(self.job_length[env_idx], repeats=self.virtual_job_length[env_idx])
            for env_idx in range(self.number_of_envs)
        ])

        # 重新计算特征
        self.construct_op_features()
        self.construct_mch_features()
        self.construct_pair_features()

        # 更新环境状态
        self.state.update(self.fea_j, self.op_mask,
                        self.fea_m, self.mch_mask,
                        self.dynamic_pair_mask, self.comp_idx, self.candidate,
                        self.fea_pairs)
        
    def schedule_maintenance(self, machine_id, start_time, duration):
        """记录维护计划"""
        end_time = start_time + duration
        if machine_id not in self.maintenance_schedule:
            self.maintenance_schedule[machine_id] = []
        self.maintenance_schedule[machine_id].append((start_time, end_time))
    
    def check_maintenance(self, current_time):
        """检查当前时间是否需要维护"""
        # print(current_time, self.maintenance_schedule)
        for machine_id, periods in self.maintenance_schedule.items():
            print(machine_id, periods)
            for start, end in periods:
                if start <= current_time < end:
                    self.mask_machine(0, machine_id)
                    # print("已屏蔽机器 {} 的加工".format(machine_id))
                elif current_time >= end and hasattr(self, 'old_op_pt_list'):
                    self.unmask_machine()

    def mask_job(self, chosen_env_id, job_id):
        """屏蔽指定工件的所有工序"""
        # 记录旧的 process_relation
        self.recorded_candidate_process_relation = self.candidate_process_relation.copy()
        
        self.candidate_process_relation[chosen_env_id, job_id, :] = True

    def unmask_job(self, chosen_env_id, job_id):
        self.candidate_process_relation[chosen_env_id, job_id] = self.recorded_candidate_process_relation[chosen_env_id, job_id]
    