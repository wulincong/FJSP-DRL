from gym_jobshop.envs.fjsp_env import FJSPEnv
import numpy as np
import numpy.ma as ma
from gymnasium import spaces


class FJSPEnvShutdown(FJSPEnv):

    def __init__(self, num_jobs, num_machines, **kwargs):
        super().__init__(num_jobs, num_machines, **kwargs)

        self.mch_working_power = np.random.uniform(0.3, 1, size=self.n_m)
        self.mch_idle_power = np.random.uniform(0.1, 0.2, size=s elf.n_m)

    def calculate_idle_periods(self, tasks_data=None):
        """
        计算每台机器的空闲时间段。
        
        :param tasks_data: list，包含任务信息的字典列表，每个字典包括以下字段：
                        - Task: 任务名称
                        - Station: 机器名称
                        - Start: 任务开始时间
                        - Duration: 任务持续时间
                        - Width: 占位宽度（可选）
        :return: list，包含每台机器的空闲时间段，每个空闲时间段是一个字典，包含以下字段：
                - Station: 机器名称
                - IdleStart: 空闲开始时间
                - IdleEnd: 空闲结束时间
        """
        # 按机器分组
        if tasks_data is None:
            tasks_data = self.tasks_data
        machines = {}
        for task in tasks_data:
            machine = task["Station"]
            if machine not in machines:
                machines[machine] = []
            machines[machine].append(task)
        
        # 计算空闲时间段
        idle_periods = []
        for machine, tasks in machines.items():
            # 按开始时间排序任务
            tasks.sort(key=lambda x: x["Start"])
            
            # 遍历任务计算空闲时间
            for i in range(len(tasks) - 1):
                current_task = tasks[i]
                next_task = tasks[i + 1]
                idle_start = current_task["Start"] + current_task["Duration"]
                idle_end = next_task["Start"]
                
                # 如果存在空闲时间，记录
                if idle_start < idle_end:
                    idle_periods.append({
                        "Station": machine,
                        "IdleStart": idle_start,
                        "IdleEnd": idle_end
                    })
        
        return idle_periods

    def decide_idle_mode(self, idle_periods, mch_idle_power, shutdown_energy_cost, idle_threshold):
        """
        判断每台机器的空闲时间是否选择停机或待机。

        :param idle_periods: list，包含每台机器的空闲时间段，格式为：
                            [{"Station": 机器名称, "IdleStart": 空闲开始时间, "IdleEnd": 空闲结束时间}, ...]
        :param mch_idle_power: list，每台机器的待机功率（按索引与机器一一对应）。
        :param shutdown_energy_cost: float，停机和开机的总能耗（固定值）。
        :param idle_threshold: float，空闲时间低于该值时选择待机。
        :return: list，每个空闲时间段的决策结果，格式为：
                [{"Station": 机器名称, "IdleStart": 空闲开始时间, "IdleEnd": 空闲结束时间, "Decision": "Shutdown" or "Idle"}, ...]
        """
        decisions = []
        
        for period in idle_periods:
            machine_index = int(period["Station"][-1]) - 1  # 假设机器名称如 "Machine1", "Machine2"
            idle_start = period["IdleStart"]
            idle_end = period["IdleEnd"]
            idle_duration = idle_end - idle_start
            
            # 待机能耗计算
            idle_energy = mch_idle_power[machine_index] * idle_duration
            
            # 决策：若待机能耗高于开关机能耗，或空闲时间较长，则选择停机
            if idle_duration >= idle_threshold and idle_energy > shutdown_energy_cost:
                decision = "Shutdown"
            else:
                decision = "Idle"
            
            # 记录结果
            decisions.append({
                "Station": period["Station"],
                "IdleStart": idle_start,
                "IdleEnd": idle_end,
                "Decision": decision
            })
        
        return decisions
    
    def add_shutdown_tasks(self, tasks_data, shutdown_energy_cost, idle_threshold):
        """
        在 tasks_data 中添加需要关机的时间段，关机任务的 Task 标记为 "Shutdown"。

        :param tasks_data: list，包含任务信息的字典列表，每个字典包括以下字段：
                        - Task: 任务名称
                        - Station: 机器名称
                        - Start: 任务开始时间
                        - Duration: 任务持续时间
        :param shutdown_energy_cost: float，停机和开机的总能耗（固定值）。
        :param idle_threshold: float，空闲时间低于该值时选择待机。
        :return: list，包含新增关机任务的完整任务数据。
        """
        # 按机器分组任务
        mch_idle_power = self.mch_idle_power[0]
        machines = {}
        for task in tasks_data:
            machine = task["Station"]
            if machine not in machines:
                machines[machine] = []
            machines[machine].append(task)

        # 处理每台机器的空闲时间
        updated_tasks = tasks_data.copy()
        for machine, tasks in machines.items():
            # 按开始时间排序任务
            tasks.sort(key=lambda x: x["Start"])
            
            for i in range(len(tasks) - 1):
                current_task = tasks[i]
                next_task = tasks[i + 1]
                
                # 计算空闲时间
                idle_start = current_task["Start"] + current_task["Duration"]
                idle_end = next_task["Start"]
                idle_duration = idle_end - idle_start
                
                if idle_duration > 0:  # 有空闲时间
                    # 获取机器索引
                    machine_index = int(machine[-1]) - 1  # 假设机器名如 "Machine1", "Machine2"
                    
                    # 待机能耗计算
                    idle_energy = mch_idle_power[machine_index] * idle_duration
                    
                    # 判断是否需要关机
                    if idle_duration >= idle_threshold and idle_energy > shutdown_energy_cost:
                        # 添加关机任务
                        shutdown_task = {
                            "Task": "Shutdown",
                            "Station": machine,
                            "Start": idle_start,
                            "Duration": idle_duration,
                            "Width": 0.4  # 关机任务的默认宽度
                        }
                        updated_tasks.append(shutdown_task)
                        self.tasks_data.append(shutdown_task)
        # 返回更新后的任务数据
        return sorted(updated_tasks, key=lambda x: (x["Station"], x["Start"]))

