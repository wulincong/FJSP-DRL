import numpy as np
import random

# 初始化种群
class GA():
    def __init__(self, job_length, op_pt):
        self.num_jobs = job_length.shape[0]
        self.num_machines = op_pt.shape[1]
        self.num_operations = job_length[0]
        self.processing_times = np.array(op_pt)

    def initialize_population(self, pop_size):
        population = []
        for _ in range(pop_size):
            individual = []
            for job in range(self.num_jobs):
                for op in range(self.num_operations):
                    machines = [m for m in range(self.num_machines) if self.processing_times[job * self.num_operations + op, m] > 0]
                    machine_choice = random.choice(machines)
                    individual.append((job, op, machine_choice))
            random.shuffle(individual)
            population.append(individual)
        return population

    # 计算适应度
    def calculate_fitness(self, individual):
        machine_time = np.zeros(self.num_machines)
        job_time = np.zeros(self.num_jobs)
        
        for job, op, machine in individual:
            start_time = max(machine_time[machine], job_time[job])
            processing_time = self.processing_times[job * self.num_operations + op, machine]
            machine_time[machine] = start_time + processing_time
            job_time[job] = start_time + processing_time
        
        return max(machine_time)

    # 选择
    def selection(self, population, fitness):
        sorted_population = [x for _, x in sorted(zip(fitness, population))]
        return sorted_population[:len(population)//2]

    # 交叉
    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1) - 2)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    # 变异
    def mutate(self, individual, mutation_rate):
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]

    # 遗传算法
    # 遗传算法
    def genetic_algorithm(self, pop_size, mutation_rate, generations):
        population = self.initialize_population(pop_size)
        best_fitness_over_time = []
        
        for _ in range(generations):
            fitness = [self.calculate_fitness(ind) for ind in population]
            best_fitness_over_time.append(min(fitness))
            population = self.selection(population, fitness)
            
            next_generation = []
            while len(next_generation) < pop_size:
                parent1, parent2 = random.sample(population, 2)
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1, mutation_rate)
                self.mutate(child2, mutation_rate)
                next_generation.extend([child1, child2])
            
            population = next_generation
        
        fitness = [self.calculate_fitness(ind) for ind in population]
        best_individual = population[np.argmin(fitness)]
        best_fitness_over_time.append(min(fitness))
        
        return best_individual, min(fitness), best_fitness_over_time


