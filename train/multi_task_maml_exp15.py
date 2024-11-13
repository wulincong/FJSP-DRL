'''
exp15 在固定的几个问题之间进行MAML学习，工件的operjob。
'''
from train.base import *
import numpy as np


class MultiTaskTrainer(Trainer):
    def __init__(self, config) -> None:
        self.env = None
        super().__init__(config)
        self.ppo = PPO_initialize()
        self.meta_optimizer = torch.optim.Adam(self.ppo.policy.parameters(), self.meta_lr)
        self.op_per_job_options = config.op_per_job_options
        print("self.op_per_job_options", self.op_per_job_options)
        self.num_tasks=len(self.op_per_job_options)

    def train(self):
        setup_seed(self.seed_train)
        self.log = []
        self.record = float('inf')
        self.train_st = time.time()
        step_count = 0
        self.history_problem = []
        for iteration in range(self.meta_iterations):
            ep_st = time.time()
            
            if iteration % self.reset_env_timestep == 0:
                self.envs = []
                for _ in range(self.num_tasks):
                    env = FJSPEnvForSameOpNums(self.n_j, self.n_m)
                    env.set_initial_data(*self.sample_training_instances(op_per_job=self.op_per_job_options[_]))
                    self.envs.append(env)
                step_count = 0

            inner_ppos = []
            makespans = [[] for _ in range(self.num_tasks)]
            loss_sum = 0

            for task in range(self.num_tasks):
                env = self.envs[task]
                state = env.reset()
                ep_rewards = self.memory_generate(env, state, self.ppo)

                theta_prime = self.ppo.inner_update(self.memory, 1, 0.001)
                state = env.reset()
                
                self.memory_generate(env, state, self.ppo, params=theta_prime)

                loss, _ = self.ppo.compute_loss(self.memory)
                loss_sum += loss
                mean_rewards_all_env = np.mean(ep_rewards)
                makespans[task].append(env.current_makespan)

            mean_loss = loss_sum / self.num_tasks
            mean_loss.backward()

            self.meta_optimizer.step()
            self.meta_optimizer.zero_grad()

            ep_et = time.time()
            makespans_mean = [np.mean(lst) for lst in makespans]
            print(makespans_mean)
            if iteration < 2: self.record = vali_result = makespans_mean[0] 

            tqdm.write(
                'Episode {}\t reward: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    iteration + 1, mean_rewards_all_env, loss, ep_et - ep_st))

            if (iteration + 1) % self.validate_timestep == 0:
                self.save_model()

            scalars={
                'Loss/train': loss
                ,'makespan_train':np.mean(makespans_mean)
                ,'makespan_validate':vali_result
            }
            self.iter_log(iteration, scalars)
            step_count += 1
            del inner_ppos
            del makespans
            self.memory.clear_memory()



if __name__ == "__main__":
    trainer = MultiTaskTrainer(configs)
    trainer.train()

