'''
exp17 在固定的几个问题之间进行MAML学习，不再随机问题。
机器数、工件数、op_per_job可以同时变化
'''
from train.base import *
import numpy as np

class MultiTaskTrainer(Trainer):
    def __init__(self, config) -> None:

        super().__init__(config)
        
        self.env = None
        self.ppo = PPO_initialize()
        self.meta_optimizer = torch.optim.Adam(self.ppo.policy.parameters(), self.meta_lr)
        
        self.n_j_options = config.n_j_options        
        self.n_m_options = config.n_m_options
        self.op_per_job_options = config.op_per_job_options
        
        print("self.n_js: ", self.n_j_options)
        print("self.n_m_options: ",self.n_m_options)
        print("self.op_per_job_options: ",self.op_per_job_options)

    def train(self):
        setup_seed(self.seed_train)
        self.log = []
        self.record = float('inf')
        self.train_st = time.time()

        assert self.num_tasks == len(self.n_j_options)

        for iteration in range(self.meta_iterations):
            ep_st = time.time()

            #重置环境
            if iteration % self.reset_env_timestep == 0:
                self.envs = []
                for _ in range(self.num_tasks):
                    env = FJSPEnvForSameOpNums(self.n_j_options[_], self.n_m_options[_])
                    env.set_initial_data(*self.sample_training_instances(
                        n_j=self.n_j_options[_], 
                        n_m=self.n_m_options[_],
                        op_per_job=self.op_per_job_options[_]
                        ))
                    self.envs.append(env)

            makespans = [[] for _ in range(self.num_tasks)]
            loss_sum = 0

            for task in range(self.num_tasks):
                env = self.envs[task]
                state = env.reset()
                theta_prime = None

                ep_rewards = self.memory_generate(env, state, self.ppo, theta_prime)
                theta_prime = self.ppo.inner_update(self.memory, 1, self.task_lr)
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
            makespans_mean = np.max([np.mean(lst) for lst in makespans])
            print(makespans_mean)
            if iteration < 2: self.record = makespan_min = makespans_mean

            tqdm.write(
                'Episode {}\t reward: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    iteration + 1, mean_rewards_all_env, loss, ep_et - ep_st))

            if makespans_mean< makespan_min:
                makespan_min = makespans_mean
                self.save_model()
            #     tqdm.write(f'The validation quality is: {vali_result} (best : {self.record})')

            scalars={
                'Loss/train': loss
                ,'makespan_train':np.mean(makespans_mean)
                # ,'makespan_validate':vali_result
            }
            self.iter_log(iteration, scalars)
            del makespans
            self.memory.clear_memory()



if __name__ == "__main__":
    trainer = MultiTaskTrainer(configs)
    trainer.train()

