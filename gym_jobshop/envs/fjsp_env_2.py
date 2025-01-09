import numpy as np
import simpy
from gymnasium import spaces
import plotly.graph_objects as go

from gym_jobshop.envs.fjsp_env import FJSPEnv

FJSPEnv = FJSPEnv

class FJSPEnvSimPy:
    def __init__(self, n_envs, n_jobs, n_machines, processing_times, job_lengths):
        """
        Initialize the FJSP environment using SimPy.
        :param n_envs: Number of environments.
        :param n_jobs: Number of jobs.
        :param n_machines: Number of machines.
        :param processing_times: Processing times matrix [N, M].
        :param job_lengths: List of job lengths per environment.
        """
        self.n_envs = n_envs
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.n_ops = processing_times.shape[0]
        self.processing_times = processing_times
        self.job_lengths = job_lengths

        # Gym spaces for compatibility
        self.action_space = spaces.Discrete(self.n_ops * self.n_machines)
        self.observation_space = spaces.Dict({
            "machine_status": spaces.Box(low=0, high=1, shape=(n_envs, n_machines), dtype=np.float32),
            "job_status": spaces.Box(low=0, high=1, shape=(n_envs, n_jobs), dtype=np.float32),
            "current_time": spaces.Box(low=0, high=np.inf, shape=(n_envs,), dtype=np.float32),
        })

        # Initialize environment and states
        self.envs = [simpy.Environment() for _ in range(self.n_envs)]
        self.machines = [
            {m: simpy.Resource(env, capacity=1) for m in range(n_machines)}
            for env in self.envs
        ]
        self.tasks_data = []
        self.current_time = np.zeros(n_envs)
        self.step_count = 0
        self.initialize_vars()

    def initialize_vars(self):
        """
        Initialize variables for scheduling simulation.
        """
        self.current_makespan = np.full(self.n_envs, float("-inf"))
        self.previous_C_max = np.full(self.n_envs, float("inf"))  # 上一个时间步的 C_max
        self.current_C_max = np.full(self.n_envs, float("-inf"))  # 当前时间步的 C_max
        self.mch_free_time = np.zeros((self.n_envs, self.n_machines))
        self.candidate_free_time = np.zeros((self.n_envs, self.n_jobs))
        self.op_ct = np.zeros((self.n_envs, self.n_ops))
        self.op_scheduled = np.zeros((self.n_envs, self.n_ops), dtype=bool)  # 是否已调度的标志
        self.op_waiting_time = np.zeros((self.n_envs, self.n_ops))
        self.op_remain_work = np.zeros((self.n_envs, self.n_ops))
        self.mch_remain_work = np.zeros((self.n_envs, self.n_machines))
        self.mask = np.zeros((self.n_envs, self.n_jobs), dtype=bool)

        self.tasks_data = [[
            {"Task": "Job", "Station": f"Machine{i}", "Start": 0, "Duration": 0, "Width": 0.4}
                for i in range(self.n_machines)
            ] for _ in range(self.n_envs)]
    
    def calculate_predicted_C_max(self, env_idx):
        """
        计算环境中所有操作的预计完工时间，并获取 C_max。
        :param env_idx: 环境索引
        :return: 当前环境的预计 C_max
        """
        predicted_completion_times = np.zeros(self.n_ops)
        for op_idx in range(self.n_ops):
            if self.op_scheduled[env_idx, op_idx]:
                # 已调度的操作，使用已有完工时间
                predicted_completion_times[op_idx] = self.op_ct[env_idx, op_idx]
            else:
                # 未调度的操作，计算预计完工时间
                # 找出该操作可用的所有机器及其空闲时间
                available_machines = np.where(self.processing_times[op_idx] > 0)[0]
                predicted_times = [
                    self.mch_free_time[env_idx, mch_idx] + self.processing_times[op_idx, mch_idx]
                    for mch_idx in available_machines
                ]
                # 选择最小的预计完工时间
                predicted_completion_times[op_idx] = min(predicted_times)
        
        # 返回所有操作的最大预计完工时间（C_max）
        return max(predicted_completion_times)

    def step(self, actions):
        """
        Execute scheduling actions and return updated states and rewards.
        :param actions: Array of actions, one per environment.
        :return: observation, reward, done
        """
        rewards = np.zeros(self.n_envs)
        for env_idx, action in enumerate(actions):
            chosen_job = action // self.n_machines
            chosen_mch = action % self.n_machines
            chosen_op = chosen_job  # Placeholder for candidate operation

            # Validate the action
            if self.processing_times[chosen_op, chosen_mch] == 0:
                print(f"Invalid action: Operation {chosen_op} cannot be processed on Machine {chosen_mch}")
                continue

            # Simulate processing the task
            duration = self.processing_times[chosen_op, chosen_mch]
            task_start_time = max(
                self.mch_free_time[env_idx, chosen_mch],  # 机器空闲时间
                self.op_ct[env_idx, chosen_op]  # 前序任务完工时间（如果有前序任务）
            )
            task_end_time = task_start_time + duration

            # 更新完工时间
            self.op_ct[env_idx, chosen_op] = task_end_time
            self.mch_free_time[env_idx, chosen_mch] = task_end_time
            self.op_scheduled[env_idx, chosen_op] = True

            # Add task to task data
            self.tasks_data[env_idx].append({
                "Task": f"Job{chosen_job}",
                "Station": f"Machine{chosen_mch}",
                "Start": task_start_time,
                "Duration": duration,
                "Width": 0.4
            })

        # Compute rewards based on predicted C_max reduction
        for env_idx in range(self.n_envs):
            # 计算预计 C_max
            self.current_C_max[env_idx] = self.calculate_predicted_C_max(env_idx)
            # 奖励为 C_max 的减少量
            reward = self.previous_C_max[env_idx] - self.current_C_max[env_idx]
            rewards[env_idx] = reward
            # 更新 previous_C_max
            self.previous_C_max[env_idx] = self.current_C_max[env_idx]

        # Check termination
        done = np.all(self.op_scheduled, axis=1).all()
        return self._get_observation(), rewards, done


    def _process_task(self, env_idx, chosen_op, chosen_mch, duration):
        """
        SimPy process to simulate task execution.
        """
        with self.machines[env_idx][chosen_mch].request() as request:
            yield request
            yield self.envs[env_idx].timeout(duration)

    def _get_observation(self):
        """
        Return the current state of the environment.
        """
        return {
            "machine_status": (self.mch_free_time > 0).astype(float),
            "job_status": (~self.mask).astype(float),
            "current_time": self.current_time,
        }

    def plot_gantt_chart(self, tasks_data=None):
        """
        Generate a Gantt chart using plotly.graph_objects.
        :param tasks_data: List of dictionaries containing task information.
                        Each task should have keys: 'Task', 'Station', 'Start', 'Duration', 'Width'.
        """
        if tasks_data is None: tasks_data = self.tasks_data[0]
        fig = go.Figure()

        for task in tasks_data:
            fig.add_trace(go.Bar(
                x=[task["Duration"]],
                y=[task["Station"]],
                base=task["Start"],
                width=[task["Width"]],
                orientation='h',
                name=task["Task"],
                text=f"{task['Task']} ({task['Duration']})",
                hoverinfo="text"
            ))

        # Customize layout
        fig.update_layout(
            title="Gantt Chart for Task Scheduling",
            xaxis_title="Time",
            yaxis_title="Machines",
            barmode="stack",
            template="plotly_white",
            height=600
        )
        
        # Show chart
        fig.show()

