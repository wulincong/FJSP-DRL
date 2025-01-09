import numpy as np
import gymnasium as gym
from gymnasium import spaces
import plotly.graph_objects as go
from plotly.colors import qualitative
import numpy.ma as ma
import colorsys
import random


class FJSPEnvBase(gym.Env):
    def __init__(self, num_jobs, num_machines, **kwargs):
                
        super(FJSPEnvBase, self).__init__()
        self.chosen_features = None
        self.observation_space = None
        self.n_j = num_jobs  # 设置作业数量
        self.n_m = num_machines  # 设置机器数量
        self.fig = go.Figure()
        self.action_space = spaces.Discrete((self.n_j - 1) * self.n_m + self.n_m)

    def other_features_initialize(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def other_features_update(self):
        raise NotImplementedError("Subclass must implement abstract method")
    
    def set_initial_data(self, job_length_list, op_pt_list):
        
        self.job_length = np.array(job_length_list)
        self.number_of_machines = op_pt_list.shape[1]
        self.number_of_jobs = job_length_list.shape[0]
        self.number_of_ops = op_pt_list.shape[0]
        self.unscheduled_op_nums = self.number_of_ops
        self.max_ops = max(self.number_of_jobs * self.number_of_machines * 2, self.number_of_ops)
        self.observation_space = spaces.Dict({
            'job_features': spaces.Box(low=0, high=np.inf, shape=(self.max_ops, len(self.chosen_features["job_features"])), dtype=np.float32),
            'mch_features': spaces.Box(low=0, high=np.inf, shape=(self.n_m, len(self.chosen_features["mch_features"])), dtype=np.float32),
            'op_mask': spaces.Box(low=0, high=1, shape=(self.max_ops, 3), dtype=np.int32),
            'mch_mask': spaces.Box(low=0, high=1, shape = (self.n_m, self.n_m),dtype=np.int32),
            'candidate': spaces.Box(low=0, high=100000, shape=(self.n_j,), dtype=np.int32),
            'pair_features': spaces.Box(low=0, high=np.inf, shape=(self.n_j, self.n_m, 8), dtype=np.float32),
            'comp_idx': spaces.Box(low=0, high=1, shape = (self.n_m, self.n_m, self.n_j), dtype=np.int32),
            'dynamic_pair_mask': spaces.Box(low=0, high=1, shape = (self.n_j, self.n_m), dtype=np.int32),
        })

        self.process_relation = (op_pt_list != 0)
        self.reverse_process_relation = ~self.process_relation
        self.op_pt = ma.array(op_pt_list, mask=self.reverse_process_relation)
        self.mask = np.full(shape=(self.number_of_jobs, ), fill_value=0, dtype=bool)
        self.remain_process_relation = np.copy(self.process_relation)
        
        self.mch_queue = np.full(shape=(self.number_of_machines, self.number_of_ops + 1), fill_value=-99, dtype=int)
        self.mch_queue_len = np.zeros((self.number_of_machines, ), dtype=int)
        self.mch_free_time = np.zeros((self.number_of_machines))

        # 用于设置候选操作
        self.job_first_op_id = np.concatenate(([0], np.cumsum(self.job_length)[:-1])).astype(int)
        self.job_last_op_id = self.job_first_op_id + self.job_length - 1
        self.mch_queue_last_op_id = np.zeros((self.number_of_machines), dtype=int)
        self.op_min_pt = np.min(self.op_pt, axis=-1).data
        self.op_ct_lb = self.op_min_pt.copy()

        for i in range(self.number_of_jobs):
            self.op_ct_lb[self.job_first_op_id[i]:self.job_last_op_id[i] + 1] = np.cumsum(
                self.op_ct_lb[self.job_first_op_id[i]:self.job_last_op_id[i] + 1])
        self.op_ct = np.zeros((self.number_of_ops,))
        
        self.candidate = self.job_first_op_id.copy()
        self.candidate_pt = self.op_pt[self.candidate]
        self.candidate_pt = self.candidate_pt.data
        self.dynamic_pair_mask = (self.candidate_pt == 0)
        self.candidate_process_relation = np.copy(self.dynamic_pair_mask.data)
        self.mch_current_available_jc_nums = np.sum(~self.candidate_process_relation, axis=1)
        self.candidate_free_time = np.zeros((self.number_of_jobs,)) 
        self.multi_env_mch_diag = np.eye(self.number_of_machines, dtype=bool)
        self.comp_idx = self.logic_operator(~self.dynamic_pair_mask)
        self.current_makespan = -1000000
        self.max_endTime = np.max(self.op_ct_lb, axis=0)

        self._init_op_mask()
        self._init_mch_mask()
        self.step_count = 0
        # 其他特征初始化
        self.other_features_initialize()

        self.tasks_data = [{"Task": "Mch", "op":-1 ,"Station": f"Machine{i}", "Start": 0, "Duration": 0, "Width": 0.4} for i in range(self.number_of_machines-1, -1, -1) ]
        
        self.old_attributes = self.__dict__.copy()
        return self._get_observation()

    def reset(self, seed=None, options=None):
        # 处理seed
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        
        # print(self.op_pt.data)
        # if not hasattr(self, "old_attributes") or self.old_attributes is None:
        self.set_initial_data(self.job_length, self.op_pt.data)
        # else: self.__dict__ = self.old_attributes.copy()
        # print(f"Reset in self.mch_queue: {self.mch_queue}")
        # # 其他状态变量
        # self.current_time = 0
        # self.done = False

        # 返回单一的观测值和额外信息
        obs = self._get_observation()
        info = {"reset_seed": seed, "options": options}  # 或其他你需要的调试信息
        return obs, info

    def step(self, action):
        # 1. 处理输入动作 actions
        """
        action = job_idx * number_of_machines + machine_idx
        """
        # print(f"Action: {action}")
        chosen_job = action // self.number_of_machines 
        chosen_mch = action % self.number_of_machines   
        chosen_op = self.candidate[chosen_job]  # 当前工件的工序
        # print(f"Job {chosen_job} is scheduled to Machine {chosen_mch} to process Op {chosen_op}, candidate: {self.candidate}")
        self.chosen_job = chosen_job
        self.chosen_op  = chosen_op
        # 2. 验证动作合法性
        
        if (self.reverse_process_relation[chosen_op, chosen_mch]).any():
            reward = -10
            terminated = False
            truncated = True
            info = {'invalid_action': f'FJSP_Env.py Error from choosing action: Op {chosen_op} can\'t be processed by Mch {chosen_mch}'}
            return self._get_observation(), float(reward), terminated, truncated, info

        self.step_count += 1

        # 3. 更新调度状态
        candidate_add_flag = (chosen_op != self.job_last_op_id[chosen_job])  # 检查当前操作是否为最后工序索引
        self.candidate[chosen_job] += candidate_add_flag
        
        if self.mask[chosen_job]:  # 检查工件是否已完成调度
            reward = -10  # 非法动作的惩罚
            terminated = False  # 未自然终止
            truncated = True  # 因非法动作强制中止
            info = {
                'invalid_action': f'Action is invalid: Job {chosen_job} has already been fully scheduled.'
            }
            return self._get_observation(), float(reward), terminated, truncated, info
        self.mask[chosen_job] = (1 - candidate_add_flag) # 已完成的调度工序被标记为True

        # 4. 机器队列与工序时间更新
        self.mch_queue[chosen_mch, self.mch_queue_len[chosen_mch]] = chosen_op

        self.mch_queue_len[chosen_mch] += 1

        chosen_op_st = np.maximum(self.candidate_free_time[chosen_job], self.mch_free_time[chosen_mch])

        self.op_ct[chosen_op] = chosen_op_st + self.op_pt[chosen_op, chosen_mch]
        self.candidate_free_time[chosen_job] = self.op_ct[chosen_op] # 候选空闲时间更新为[0, 0, 15]
        self.mch_free_time[chosen_mch] = self.op_ct[chosen_op]  # 机器空闲时间更新为[0, 15]
        self.current_makespan = np.maximum(self.current_makespan, self.op_ct[chosen_op])

        # 利用这些更新的数据，计算下一步可以调度的工序和机器组合，确保时间约束的正确性。

        # 更新候选工序的加工时间和工序-机器合法关系

        if candidate_add_flag:
            self.candidate_pt[chosen_job] = self.op_pt[chosen_op + 1]
            self.candidate_process_relation[chosen_job] = self.reverse_process_relation[chosen_op + 1]
        else:
            self.candidate_process_relation[chosen_job] = 1

        # 将候选工序可用时间扩展到一个额外的维度
        candidate_expanded = self.candidate_free_time[:, np.newaxis]  # ( num_jobs, 1)
        mch_expanded = self.mch_free_time[np.newaxis, :]  # (1, num_machines)

        # 使用广播机制计算每个工序和机器的最早可用时间
        self.pair_free_time = np.maximum(candidate_expanded, mch_expanded)  # (num_jobs, num_machines) [[ 0., 15.], [ 0., 15.], [15., 15.]]

        schedule_matrix = ma.array(self.pair_free_time, mask=self.candidate_process_relation) # [[0, 15.], [0, --.], [15., --]]
        self.next_schedule_time = np.min(schedule_matrix.reshape(-1))

        self.remain_process_relation[chosen_op] = 0
        self.op_scheduled_flag[chosen_op] = 1

        self.deleted_op_nodes = np.logical_and((self.op_ct <= self.next_schedule_time[np.newaxis]),
                                self.op_scheduled_flag)

        self.delete_mask_fea_j = np.tile(self.deleted_op_nodes[:, np.newaxis],
                                         (1, self.op_fea_dim))

        self._update_op_mask()
        
        self.mch_queue_last_op_id[chosen_mch] = chosen_op
        self.unscheduled_op_nums -= 1

        diff = self.op_ct[chosen_op] - self.op_ct_lb[chosen_op]
        self.op_ct_lb[chosen_op:self.job_last_op_id[chosen_job] + 1] += diff.astype(self.op_ct_lb.dtype)

        self.dynamic_pair_mask = np.copy(self.candidate_process_relation)

        self.unavailable_pairs = np.array(self.pair_free_time > self.next_schedule_time)
        # 某个工件的空闲时间大于下次调度时间，说明该工件目前不可用，或者当前工件无法在机器上加工，也不可用
        self.dynamic_pair_mask = np.logical_or(self.dynamic_pair_mask, self.unavailable_pairs)

        self.comp_idx = self.logic_operator(~self.dynamic_pair_mask)
        self._update_mch_mask()
        
        # 更新用户特征
        self.other_features_update()

        # 更新状态

        self.tasks_data.append({"Task": f"Job{chosen_job}", 
                                "op": chosen_op,
                                "Station": f"Machine{chosen_mch}", 
                                "Start": chosen_op_st, 
                                "Duration": self.op_pt[chosen_op, chosen_mch], 
                                "Width": 0.4})
        reward = self.max_endTime - np.max(self.op_ct_lb, axis=0)
        self.max_endTime = np.max(self.op_ct_lb, axis=0)

        # 检查是否完成
        terminated = self.step_count >= self.number_of_ops
        truncated = False  # 非正常中止
        info = {"makespan": self.max_endTime, "next_schedule_time": self.next_schedule_time, "step_count": self.step_count, 
                "chosem_op": chosen_op, "chosem_mch": chosen_mch, "remain_works":self.remain_work_status()}
        return self._get_observation(), float(reward), terminated, truncated, info
    
    def remain_work_status(self):
        """
        获取当前时刻的环境状态，返回剩余工件操作数和 op_pt_dict。
        """
        # 计算每个工件剩余的待调度工序数
        remaining_ops = self.job_last_op_id - self.candidate

        op_pt = []

        for job_id in range(len(self.job_length)):
            cnt = 0
            for op_id in range(self.candidate[job_id], self.job_last_op_id[job_id] + 1):
                if self.op_scheduled_flag[op_id] == 0:
                    op_pt.append(list(self.op_pt[op_id].data))  # 将时间加入字典
                    cnt += 1
            remaining_ops[job_id] = cnt
        return remaining_ops, np.array(op_pt, dtype=np.float32)

    def _extent_features(self):
        """扩展到观测空间"""
        pass

    def _get_observation(self):
        """
        构建特征字典并归一化处理
        """
        self._extent_features()  # 扩展特征
        observation = {}

        def normalize(feature_array):
            """
            对特征进行归一化处理
            :param feature_array: numpy 数组
            :return: 归一化后的 numpy 数组
            """
            if np.all((feature_array == 0) | (feature_array == 1)):
                # 如果仅包含 0 和 1，则无需归一化
                return feature_array
            else:
                epsilon = 1e-6
                max_val = feature_array.max()
                min_val = feature_array.min()
                # 避免除以零的情况
                if max_val == min_val:
                    return np.full_like(feature_array, epsilon)
                return epsilon + (feature_array - min_val) / (max_val - min_val) * (1 - epsilon)

        for feature_name, feature_requests in self.chosen_features.items():
            if feature_requests is not None:
                # 对 feature_requests 中的每个特征进行归一化
                stacked_features = np.stack(
                    [normalize(getattr(self, kids_feature_name)) for kids_feature_name in feature_requests],
                    axis=-1
                )
                observation[feature_name] = stacked_features
            else:
                # 对直接的 feature_name 进行归一化
                observation[feature_name] = getattr(self, feature_name)

        # 扩展 job_features
        if "job_features" in observation:
            job_features = observation["job_features"]
            current_shape = job_features.shape
            padded_job_features = np.zeros((self.max_ops, current_shape[1]), dtype=job_features.dtype)
            padded_job_features[:current_shape[0], :] = job_features
            observation["job_features"] = padded_job_features

        # 扩展 op_mask
        if "op_mask" in observation:
            op_mask = observation["op_mask"]
            current_shape = op_mask.shape
            padded_op_mask = np.ones((self.max_ops, current_shape[1]), dtype=op_mask.dtype)  # 用 1 填充
            padded_op_mask[:current_shape[0], :] = op_mask
            observation["op_mask"] = padded_op_mask

        return observation

    def _get_job_id(self, chosen_job):
        cumulative = 0
        for job_id, length in enumerate(self.job_length):
            if chosen_job < cumulative + length:
                return job_id
            cumulative += length
        return None

    def _get_next_operation(self, job_id, current_op_idx):
        start_idx = sum(self.job_length[:job_id])
        current_op = current_op_idx - start_idx
        if current_op + 1 < self.job_length[job_id]:
            return start_idx + current_op + 1
        return None
    
    def _init_op_mask(self):
        self.op_mask = np.full(shape=(self.number_of_ops, 3),fill_value=0, dtype=np.float32)
        self.op_mask[self.job_first_op_id, 0] = 1
        self.op_mask[self.job_last_op_id, 2] = 1

    def _update_op_mask(self):
        object_mask = np.zeros_like(self.op_mask)
        object_mask[:, 2] = self.deleted_op_nodes
        object_mask[1:, 0] = self.deleted_op_nodes[:-1]
        self.op_mask = np.logical_or(object_mask, self.op_mask).astype(np.float32)
    
    def _update_mch_mask(self):

        self.mch_mask = self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        self.delete_mask_fea_m = np.tile(~(np.sum(self.mch_mask, keepdims=True, axis=-1).astype(bool)),
                                         (1, self.mch_fea_dim))
        self.mch_mask[self.multi_env_mch_diag] = 1

    def _init_mch_mask(self):

        self.mch_mask = self.logic_operator(self.remain_process_relation).sum(axis=-1).astype(bool)
        self.delete_mask_fea_m = np.tile(~(np.sum(self.mch_mask, keepdims=True, axis=-1).astype(bool)),
                                         (1, self.mch_fea_dim))
        self.mch_mask[self.multi_env_mch_diag] = 1

    def logic_operator(self, x, flagT=True):
        if flagT:
            x = x.transpose(1, 0)
        d1 = np.expand_dims(x, 1)
        d2 = np.expand_dims(x, 0)

        return np.logical_and(d1, d2).astype(np.float32)
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def generate_action(self):
        # 假设 action_space 大小为 (self.n_j - 1) * self.n_m + self.n_m
        action_space_size = (self.n_j - 1) * self.n_m + self.n_m

        # 生成一个合法的随机概率分布
        random_distribution = np.random.rand(action_space_size)
        
        # 获取 dynamic_pair_mask，假设是一个 (self.n_j, self.n_m) 的二维数组
        dynamic_pair_mask = ~self.dynamic_pair_mask

        # 将 dynamic_pair_mask 展平为一维数组
        flat_mask = dynamic_pair_mask.flatten()  # 形状变为 (n_j * n_m)

        # 屏蔽非法的动作：合法动作保持原概率，非法动作的概率设置为 0
        random_distribution = random_distribution * flat_mask

        # 选择最大值对应的动作（通过直接索引选出最大值的位置）
        action = np.argmax(random_distribution)  # 选择最大值的索引

        return action

    def render(self):
        '''tasks_data = [
        {"Task": "Job1-Task1", "Station": "Machine1", "Start": 0, "Duration": 4, "Width": 0.4},
        {"Task": "Job2-Task1", "Station": "Machine2", "Start": 5, "Duration": 3, "Width": 0.4},
        {"Task": "Job3-Task1", "Station": "Machine3", "Start": 9, "Duration": 2, "Width": 0.4},
        ]   '''
        # 获取唯一的Job名称列表
        unique_jobs = list(set(task['Task'] for task in self.tasks_data))

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
            title="Gantt Chart",
            xaxis_title="time",
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
