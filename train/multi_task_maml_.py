from train.base import *
import numpy as np

class ConvergenceChecker:
    def __init__(self, window_size=20, threshold=0.01):
        self.window_size = window_size
        self.threshold = threshold
        self.data = []

    def add_data(self, value):
        self.data.append(value/1000.0)
        if len(self.data) > self.window_size:
            self.data.pop(0)

    def is_converged(self):
        if len(self.data) < self.window_size:
            return False
        std_dev = np.std(self.data)
        print(std_dev)
        return std_dev < self.threshold

    def clear(self):
        self.data = []

# 示例使用
convergence_checker = ConvergenceChecker(window_size=20, threshold=0.006)

class MultiTaskTrainer(Trainer):
    def __init__(self, config) -> None:
        self.env = None
        super().__init__(config)
        self.ppo = PPO_initialize()

    def train(self):
        setup_seed(self.seed_train)
        self.log = []
        self.record = float('inf')
        self.validation_log = []
        self.train_st = time.time()
        change_env = True
        step_count = 0
        step_counts = []
        self.history_problem = []
        for iteration in range(self.meta_iterations):
            ep_st = time.time()
            if change_env:
                max_try = 1000
                for i in range(max_try):
                    if i + 1 == max_try:
                        print(self.history_problem)
                        print(step_counts)
                        exit(f"所有问题训练完成，iteration={iteration}")
                        
                    n_j = random.choice([8, 11, 13, 16, 17])
                    n_m = random.choice([4, 8, 12])
                    problem = (n_j, n_m)

                    if problem not in self.history_problem: 
                        self.history_problem.append(problem)
                        self.n_j , self.n_m = problem
                        break

                print(f"{self.n_j}x{self.n_m}")

                self.envs = [FJSPEnvForSameOpNums(self.n_j, self.n_m) for _ in range(self.num_tasks)]
                for env in self.envs:
                    dataset_job_length, dataset_op_pt = self.sample_training_instances()
                    state = env.set_initial_data(dataset_job_length, dataset_op_pt)
                change_env = False
                step_count = 0

            inner_ppos = []
            for task in range(self.num_tasks):
                env = self.envs[task]
                state = env.reset()
                inner_ppo = deepcopy(self.ppo)
                ep_rewards = self.memory_generate(env, state, inner_ppo)

                loss, v_loss = inner_ppo.update(self.memory)
                mean_rewards_all_env = np.mean(ep_rewards)
                mean_makespan_all_env = np.mean(env.current_makespan)
                inner_ppos.append(inner_ppo)

            makespans = []

            for i in range(self.num_tasks):
                env = self.envs[i]
                inner_ppo = inner_ppos[i]
                state = env.reset()
                ep_rewards = self.memory_generate(env, state, inner_ppo)
                loss, _ = self.ppo.update(self.memory)
                makespans.append(np.mean(env.current_makespan))

            ep_et = time.time()
            makespan = np.mean(makespans)

            if iteration < 2: self.record = vali_result = makespan 

            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    iteration + 1, mean_rewards_all_env, makespan, loss, ep_et - ep_st))
            
            convergence_checker.add_data(makespan)

            if convergence_checker.is_converged():
                print("Converged at step:", step_count)
                step_counts.append(step_count)
                change_env = True

            if (iteration + 1) % self.validate_timestep == 0:
                vali_result = self.valid_model()
                tqdm.write(f'The validation quality is: {vali_result} (best : {self.record})')
            del inner_ppos
            scalars={
                'Loss/train': loss
                ,'makespan_train':makespan
                ,'makespan_validate':vali_result
            }
            self.iter_log(iteration, scalars)
            step_count += 1
        
        print(step_counts)
if __name__ == "__main__":
    trainer = MultiTaskTrainer(configs)
    trainer.train()
