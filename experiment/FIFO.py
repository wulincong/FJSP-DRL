from data_utils import SD2_instance_generator_EM
import time
import numpy as np



def FIFO_scheduling(job_length, op_pt, mch_working_power, mch_idle_power):
    """
    :param job_length: 每个作业的操作数 (shape [J])
    :param op_pt: 加工时间矩阵 (shape [N, M])
    :param mch_working_power: 每台机器的工作功率 (shape [M])
    :param mch_idle_power: 每台机器的待机功率 (shape [M])
    :return: 总待机能耗，总加工能耗，调度表
    """
    n_jobs = len(job_length)
    n_machines = len(mch_working_power)
    n_operations = len(op_pt)

    # 初始化调度表，每个操作的开始和结束时间，以及所选机器
    schedule = np.full((n_operations, 3), -1)  # [start_time, end_time, machine]

    # 初始化每台机器的时间表，用于追踪何时可用
    machine_times = np.zeros(n_machines)
    
    # 初始化每台机器的工作时间
    machine_working_times = np.zeros(n_machines)

    # 作业的操作指针，初始时每个作业的第一个操作为0
    job_indices = np.zeros(n_jobs, dtype=int)

    # 初始化能耗
    total_idle_energy = 0
    total_working_energy = 0

    # 当前时间
    current_time = 0

    while np.sum(job_indices) < n_operations:
        for j in range(n_jobs):
            if job_indices[j] >= job_length[j]:
                continue  # 当前作业的所有操作已经调度完毕

            op_index = sum(job_length[:j]) + job_indices[j]  # 当前操作的全局索引

            # 获取当前操作的加工时间和可用的机器
            available_machines = np.where(op_pt[op_index] > 0)[0]

            # 使用FIFO策略选择最早可用的机器
            earliest_machine = available_machines[np.argmin(machine_times[available_machines])]

            # 计算操作的开始时间和结束时间
            start_time = max(current_time, machine_times[earliest_machine])
            end_time = start_time + op_pt[op_index][earliest_machine]

            # 更新调度表
            schedule[op_index] = [start_time, end_time, earliest_machine]

            # 更新机器的下一个可用时间
            machine_times[earliest_machine] = end_time

            # 计算该机器的工作时间
            working_time = end_time - start_time
            machine_working_times[earliest_machine] += working_time
            
            # 累加工作能耗
            total_working_energy += working_time * mch_working_power[earliest_machine]
            
            # 更新当前时间到最早操作结束时间
            current_time = end_time

            # 将作业的操作指针向前移动
            job_indices[j] += 1

    # 计算总待机能耗
    max_end_time = max(machine_times)  # 获取整个调度的最大结束时间
    for m in range(n_machines):
        idle_time = max_end_time - machine_working_times[m]  # 计算每台机器的待机时间
        total_idle_energy += idle_time * mch_idle_power[m]  # 累加待机能耗

    return total_idle_energy, total_working_energy, schedule


# 主程序部分
if __name__ == "__main__":
    # 配置生成FJSP实例
    class Config:
        def __init__(self, n_j, n_m, op_per_job, low, high, data_suffix, op_per_mch_min=1, op_per_mch_max=3):
            self.n_j = n_j
            self.n_m = n_m
            self.op_per_job = op_per_job
            self.low = low
            self.high = high
            self.data_suffix = data_suffix
            self.op_per_mch_min = op_per_mch_min
            self.op_per_mch_max = op_per_mch_max

    # 定义实例数量
    num_instances = 100

    # 初始化总能耗
    total_energy = 0

    # 记录程序开始时间
    total_start_time = time.time()

    for _ in range(num_instances):
        # 实例生成
        config = Config(n_j=20, n_m=10, op_per_job=10, low=1, high=99, data_suffix='mix')
        job_length, op_pt, op_per_mch, mch_working_power, mch_idle_power = SD2_instance_generator_EM(config)

        # 运行FIFO调度算法
        idle_energy, working_energy, schedule = FIFO_scheduling(job_length, op_pt, mch_working_power, mch_idle_power)
        
        # 累加总能耗
        total_energy += (idle_energy + working_energy)

    # 记录程序结束时间
    total_end_time = time.time()

    # 计算平均总能耗
    average_energy = total_energy / num_instances

    # 计算总运行时间
    total_run_time = total_end_time - total_start_time

    # 输出平均总能耗和总运行时间
    print(f"10个实例的平均总功耗: {average_energy:.2f}")
    print(f"总运行时间: {total_run_time:.4f} 秒")



