#不同规模问题的maml，iter间

from train.base import *
import numpy as np


# 示例使用
convergence_checker = ConvergenceChecker(window_size=20, threshold=0.01)

class MultiTaskTrainer(Trainer):
    def __init__(self, config) -> None:
        self.env = None
        super().__init__(config)
        self.ppo = PPO_initialize()
        self.meta_optimizer = torch.optim.Adam(self.ppo.policy.parameters(), self.meta_lr)

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
                probs = [(random.choice([8, 11, 13, 16, 17]), random.choice([4, 8, 12])) for _ in range(self.num_tasks)]
                print(probs)
                self.envs = [FJSPEnvForSameOpNums(probs[_][0], probs[_][1]) for _ in range(self.num_tasks)]
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
                inner_ppos.append(inner_ppo)

            makespans = []
            loss_sum = 0
            for i in range(self.num_tasks):
                env = self.envs[i]
                inner_ppo = inner_ppos[i]
                state = env.reset()
                ep_rewards = self.memory_generate(env, state, inner_ppo)
                loss, _ = self.ppo.compute_loss(self.memory)
                loss_sum += loss
                makespans.append(np.mean(env.current_makespan))
            
            mean_loss = loss_sum / self.num_tasks
            self.meta_optimizer.zero_grad()
            mean_loss.backward()
            self.meta_optimizer.step()
            ep_et = time.time()
            makespan = np.mean(makespans)

            if iteration < 2: self.record = vali_result = makespan 

            tqdm.write(
                'Episode {}\t reward: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    iteration + 1, mean_rewards_all_env, mean_loss, ep_et - ep_st))
            print(makespans)
            convergence_checker.add_data(mean_loss)

            if convergence_checker.is_converged():
                print("Converged at step:", step_count)
                step_counts.append(step_count)
                convergence_checker.clear()
                change_env = True

            if (iteration + 1) % self.validate_timestep == 0:
                vali_result = self.valid_model()
                tqdm.write(f'The validation quality is: {vali_result} (best : {self.record})')
            del inner_ppos
            
            scalars={
                'Loss/train': loss
                ,'makespan_train':makespan
                ,'makespan_validate':vali_result
                ,"loss_std":convergence_checker.std_dev()
            }
            self.iter_log(iteration, scalars)
            step_count += 1
        
        print(step_counts)
if __name__ == "__main__":
    trainer = MultiTaskTrainer(configs)
    trainer.train()
