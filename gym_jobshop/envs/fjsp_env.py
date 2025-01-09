from gym_jobshop.envs.fjsp_env_base import FJSPEnvBase
import numpy as np
import numpy.ma as ma
from gymnasium import spaces

class FJSPEnv(FJSPEnvBase):
    def __init__(self, num_jobs, num_machines, **kwargs):
        super().__init__(num_jobs, num_machines, **kwargs)
        self.chosen_features = {"job_features":[
                                                "op_scheduled_flag",
                                                "op_ct_lb",
                                                "op_min_pt",
                                                "pt_span",
                                                "op_mean_pt",
                                                "op_waiting_time",
                                                "op_remain_work",
                                                "op_match_job_left_op_nums",
                                                "op_match_job_remain_work",
                                                "op_available_mch_nums"],
                                "mch_features":[
                                                "mch_current_available_jc_nums",
                                                "mch_current_available_op_nums",
                                                "mch_min_pt",
                                                "mch_mean_pt",
                                                "mch_waiting_time",
                                                "mch_remain_work",
                                                "mch_free_time",
                                                "mch_working_flag"],
                                "op_mask": None,
                                "mch_mask": None,
                                "candidate": None,
                                "pair_features": None,
                                "comp_idx": None,
                                "dynamic_pair_mask": None,
            }
        # 定义空间
        self.mch_fea_dim = len(self.chosen_features["mch_features"])
        self.op_fea_dim = len(self.chosen_features["job_features"])
        

    # 用户定义的初始化和更新方法
    def other_features_initialize(self):
        self.mch_features_initialize()
        self.op_features_initialize()
        self.pair_features_initialize()

    def other_features_update(self):
        chosen_job, chosen_op = self.chosen_job, self.chosen_op
        self.mch_features_update(chosen_job, chosen_op)
        self.op_features_update(chosen_job, chosen_op)
        self.pair_features_update()

    def op_features_initialize(self: FJSPEnvBase):

        self.compatible_op = np.sum(self.process_relation, 1)
        
        self.op_scheduled_flag = np.zeros((self.number_of_ops))
        self.op_max_pt = np.max(self.op_pt, axis=-1).data # static feature
        self.pt_span = self.op_max_pt - self.op_min_pt # static feature
        self.op_mean_pt = np.mean(self.op_pt, axis=1).data  # static feature
        self.op_available_mch_nums = np.copy(self.compatible_op) / self.number_of_machines  # static feature
        # self.mch_max_pt = np.max(self.op_pt, axis=1)
        self.op_waiting_time = np.zeros((self.number_of_ops))
        self.op_remain_work = np.zeros((self.number_of_ops))
        self.op_match_job_left_op_nums = np.array(np.repeat(self.job_length, repeats=self.job_length))
        self.job_remain_work = [np.sum(self.op_mean_pt[self.job_first_op_id[i]:self.job_last_op_id[i] + 1])
                for i in range(self.number_of_jobs)]

        self.op_match_job_remain_work = np.array(np.repeat(self.job_remain_work, repeats=self.job_length))

    def op_features_update(self: FJSPEnvBase, chosen_job, chosen_op):
        
        self.op_scheduled_flag[self.chosen_op] = 1
        # update op_waiting_time
        self.op_waiting_time = np.zeros((self.number_of_ops))
        self.op_waiting_time[self.candidate] = \
            (1 - self.mask) * np.maximum(np.expand_dims(self.next_schedule_time, axis=0)
                                    - self.candidate_free_time, 0) + self.mask * self.op_waiting_time[self.candidate]
        # update op_remain_work
        self.op_remain_work = np.maximum(self.op_ct -
                                            np.expand_dims(self.next_schedule_time, axis=0), 0)
        # update op_match_job_left_op_nums
        self.op_match_job_left_op_nums[self.job_first_op_id[chosen_job]:self.job_last_op_id[chosen_job] + 1] -= 1
        # update op_match_job_remain_work
        self.op_match_job_remain_work[
            self.job_first_op_id[chosen_job]:self.job_last_op_id[chosen_job] + 1] -= \
                self.op_mean_pt[chosen_op]

    def mch_features_initialize(self: FJSPEnvBase):
        self.compatible_mch = np.sum(self.process_relation, 0)

        self.mch_current_available_jc_nums = np.sum(~self.candidate_process_relation, axis=0)
        self.mch_current_available_op_nums = np.copy(self.compatible_mch)
        self.mch_min_pt = np.max(self.op_pt, axis=0).data # static feature
        # self.mch_max_pt = np.max(self.op_pt, axis=1)
        self.mch_mean_pt = np.mean(self.op_pt, axis=0).filled(0) # static feature
        self.mch_remain_work = np.zeros((self.number_of_machines))
        self.mch_waiting_time = np.zeros((self.number_of_machines))
        self.mch_working_flag = np.zeros((self.number_of_machines))

    def mch_features_update(self: FJSPEnvBase, chosen_job, chosen_op):
        self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask, axis=0)
        self.mch_current_available_op_nums -= self.process_relation[chosen_op]
        mch_free_duration = np.expand_dims(self.next_schedule_time, axis=0) - self.mch_free_time
        mch_free_flag = mch_free_duration < 0
        self.mch_working_flag = mch_free_flag + 0
        self.mch_waiting_time = (1 - mch_free_flag) * mch_free_duration
        self.mch_remain_work = np.maximum(-mch_free_duration, 0)

    def pair_features_initialize(self):
        self.pair_features_update()

    def pair_features_update(self):
        remain_op_pt = ma.array(self.op_pt, mask=~self.remain_process_relation)

        chosen_op_max_pt = np.expand_dims(self.op_max_pt[self.candidate], axis=-1)

        max_remain_op_pt = np.max(np.max(remain_op_pt, axis=0, keepdims=True), axis=1, keepdims=True) \
            .filled(0 + 1e-8)

        mch_max_remain_op_pt = np.max(remain_op_pt, axis=0, keepdims=True).filled(0 + 1e-8)

        pair_max_pt = np.max(np.max(self.candidate_pt, axis=0, keepdims=True),
                                axis=1, keepdims=True) + 1e-8

        mch_max_candidate_pt = np.max(self.candidate_pt, axis=0, keepdims=True) + 1e-8

        pair_wait_time = self.op_waiting_time[self.candidate][:,np.newaxis] + \
                                        self.mch_waiting_time[np.newaxis, :]

        chosen_job_remain_work = np.expand_dims(self.op_match_job_remain_work
                                                [self.candidate],
                                                axis=-1) + 1e-8

        self.pair_features = np.stack((self.candidate_pt,
                                    self.candidate_pt / chosen_op_max_pt,
                                    self.candidate_pt / mch_max_candidate_pt,
                                    self.candidate_pt / max_remain_op_pt,
                                    self.candidate_pt / mch_max_remain_op_pt,
                                    self.candidate_pt / pair_max_pt,
                                    self.candidate_pt / chosen_job_remain_work,
                                    pair_wait_time), axis=-1)


    def mask_machine(self, mch_id):
        op_pt_list = self.op_pt.data
        op_pt_list[:, mch_id] = 0
        zero_row_indices = np.where(np.all(op_pt_list == 0, axis=1))[0]
        # Determine which job each index belongs to
        #检查op_pt_list是否有全0的行，如果有，把这一行的id计算出来
        task_ids = []
        for idx in zero_row_indices:
            task_id = np.where((self.job_first_op_id <= idx) & (self.job_last_op_id >= idx))[0]
            self.candidate_process_relation[task_id, :] = 0
            self.process_relation[task_id, :] = 0
            self.remain_process_relation[task_id, :] = 0
            task_ids.append(task_id[0] if len(task_id) > 0 else None)

        self.process_relation[:, mch_id] = 0
        self.reverse_process_relation = ~self.process_relation
        self.op_pt = ma.array(op_pt_list, mask=self.reverse_process_relation)
        self.remain_process_relation[:, mch_id] = 0
        self.candidate_process_relation[:, mch_id] = 0
        # self.candidate_free_time
        self.mch_free_time[mch_id] = 1000

        return {"unavaliable job id":task_ids}