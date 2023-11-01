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
            if iteration % self.reset_env_timestep == 0:
                probs = [(random.choice([10, 15, 20]), random.choice([5, 10])) for _ in range(self.num_tasks)]
                print(probs)
                self.envs = [FJSPEnvForSameOpNums(probs[_][0], probs[_][1]) for _ in range(self.num_tasks)]
                for env in self.envs:
                    dataset_job_length, dataset_op_pt = self.sample_training_instances()
                    state = env.set_initial_data(dataset_job_length, dataset_op_pt)
                change_env = False
                step_count = 0

            inner_ppos = []
            data_primes = []
            makespans = [[] for _ in range(self.num_tasks)]
            loss_sum = 0

            for task in range(self.num_tasks):
                env = self.envs[task]
                state = env.reset()
                ep_rewards = self.memory_generate(env, state, self.ppo)

                theta_prime = self.ppo.inner_update(self.memory)
                state = env.reset()                
                ep_rewards_ = self.memory_generate(env, state, self.ppo, params=theta_prime)
                data_prime = deepcopy(self.memory)
                data_primes.append(data_prime)
                mean_rewards_all_env = np.mean(ep_rewards)
                makespans[task].append(env.current_makespan)


            for i in range(self.num_tasks):
                loss, _ = self.ppo.compute_loss(data_primes[i])
                loss_sum += loss

            mean_loss = loss_sum / self.num_tasks
            self.meta_optimizer.zero_grad()
            mean_loss.backward()
            # 查看哪些参数受到loss的影响
            # for name, param in self.ppo.policy.named_parameters():
            #     if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
            #         print(name, "受到了loss的影响")
            #     else:
            #         print(name, "没有受到loss的影响")
            
            self.meta_optimizer.step()

            ep_et = time.time()
            makespans_mean = [np.mean(lst) for lst in makespans]
            print(makespans_mean)
            if iteration < 2: self.record = vali_result = makespans_mean[0] 

            tqdm.write(
                'Episode {}\t reward: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    iteration + 1, mean_rewards_all_env, mean_loss, ep_et - ep_st))
            convergence_checker.add_data(float(mean_loss))
            # print(mean_loss.device)

            if (iteration + 1) % self.validate_timestep == 0:
                vali_result = self.valid_model()
                tqdm.write(f'The validation quality is: {vali_result} (best : {self.record})')
            del inner_ppos
            
            scalars={
                'Loss/train': loss
                ,'makespan_train':np.mean(makespans_mean)
                ,'makespan_validate':vali_result
                ,"loss_std":convergence_checker.std_dev()
            }
            self.iter_log(iteration, scalars)
            step_count += 1
        
        print(step_counts)


if __name__ == "__main__":
    trainer = MultiTaskTrainer(configs)
    trainer.train()

