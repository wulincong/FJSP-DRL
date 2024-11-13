import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt
from params import configs
from data_utils import SD2_instance_generator_EM

os.environ['ON_PY'] = "1"

# 实例参数设置
num_jobs = 10
num_machines = 5
num_operations = 5

# 生成FJSP实例
job_length, op_pt, op_per_mch, mch_working_power, mch_idle_power = SD2_instance_generator_EM(configs, num_jobs, num_machines, num_operations)

def initialize_population(population_size):
    # 初始化种群，每个个体是一个调度方案
    population = []
    for _ in range(population_size):
        individual = []
        for j in range(num_jobs):
            job_schedule = np.random.permutation(np.arange(job_length[j]))  # 每个作业的操作顺序
            individual.append(job_schedule)
        population.append(individual)
    return population

def decode_individual(individual):
    # 解码个体为调度顺序和机器分配
    schedule = []
    machine_assignment = []
    for job in individual:
        job_schedule = []
        job_machines = []
        for op in job:
            compatible_machines = np.where(op_pt[op] > 0)[0]
            selected_machine = random.choice(compatible_machines)  # 随机选择一个兼容机器
            job_schedule.append(op)
            job_machines.append(selected_machine)
        schedule.append(job_schedule)
        machine_assignment.append(job_machines)
    return schedule, machine_assignment

def calculate_energy(schedule, machine_assignment):
    # 计算给定调度的能耗
    total_energy = 0
    machine_time = [0] * num_machines
    for i, job in enumerate(schedule):
        for op_idx, operation in enumerate(job):
            machine = machine_assignment[i][op_idx]
            processing_time = op_pt[operation][machine]
            start_time = max(machine_time[machine], 0)  # 机器可用时间与操作开始时间取最大
            end_time = start_time + processing_time
            machine_time[machine] = end_time
            total_energy += processing_time * mch_working_power[machine]  # 计算工作能耗
    # 计算空闲能耗
    max_time = max(machine_time)
    for machine in range(num_machines):
        total_energy += (max_time - machine_time[machine]) * mch_idle_power[machine]  # 计算空闲能耗
    return total_energy

def fitness(individual):
    # 适应度函数，能耗的负数，因为我们希望能耗最小化
    schedule, machine_assignment = decode_individual(individual)
    energy = calculate_energy(schedule, machine_assignment)
    return -energy

def selection(population, fitness_scores):
    # 选择操作（轮盘赌选择）
    total_fitness = sum(fitness_scores)
    probabilities = [f / total_fitness for f in fitness_scores]
    selected_indices = np.random.choice(len(population), size=2, p=probabilities)
    return [population[i] for i in selected_indices]

def crossover(parent1, parent2):
    # 单点交叉
    crossover_point = random.randint(0, num_jobs - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate):
    # 突变操作
    for i in range(num_jobs):
        if random.random() < mutation_rate:
            swap_idx = np.random.choice(len(individual[i]), 2, replace=False)
            # 交换两个操作
            individual[i][swap_idx[0]], individual[i][swap_idx[1]] = individual[i][swap_idx[1]], individual[i][swap_idx[0]]
    return individual

def genetic_algorithm(population_size, mutation_rate, generations):
    # 遗传算法主流程
    population = initialize_population(population_size)
    best_solution = None
    best_fitness = float('-inf')
    fitness_over_time = []

    for generation in range(generations):
        fitness_scores = [fitness(individual) for individual in population]
        
        # 更新最佳解
        current_best_fitness = max(fitness_scores)
        current_best_solution = population[np.argmax(fitness_scores)]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = current_best_solution
        
        fitness_over_time.append(-best_fitness)  # 存储使能耗最小的适应度
        
        # 选择、交叉和变异
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = selection(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))
        
        population = new_population
    
    return best_solution, -best_fitness, fitness_over_time

# 设置遗传算法参数
population_size = 50
mutation_rate = 0.1
generations = 500

# 运行遗传算法
t1 = time.time()
best_solution, best_fitness, fitness_over_time = genetic_algorithm(population_size, mutation_rate, generations)
t2 = time.time()

print(f"Time: {t2 - t1}")
print(f"Best solution: {best_solution}")
print(f"Best fitness (energy consumption): {best_fitness}")

# 可视化迭代过程中的能耗
plt.plot(fitness_over_time)
plt.title('Genetic Algorithm: Energy Consumption over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Energy Consumption')
plt.grid(True)

# 保存图像为文件
plt.savefig('genetic_algorithm_energy_consumption.png')  # 保存为文件

plt.show()
