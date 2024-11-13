#改进maml训练

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
        self.n_j_options = config.n_j_options
        print("self.n_js", self.n_j_options)
        self.n_m_options = config.n_m_options
        print(self.n_m_options)

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
                assert self.num_tasks == len(self.n_j_options)
                probs = [(self.n_j_options[_], self.n_m_options[_]) for _ in range(self.num_tasks)]
                # probs = [(random.choice(self.n_j_options), random.choice(self.n_m_options)) for _ in range(self.num_tasks)]
                print(probs)

                self.envs = []
                for _ in range(self.num_tasks):
                    self.n_j, self.n_m = probs[_]
                    env = FJSPEnvForSameOpNums(probs[_][0], probs[_][1])
                    env.set_initial_data(*self.sample_training_instances(n_j=probs[_][0], n_m=probs[_][1]))
                    self.envs.append(env)
                step_count = 0

            makespans = [[] for _ in range(self.num_tasks)]
            loss_sum = 0

            for task in range(self.num_tasks):
                env = self.envs[task]
               
                theta_prime = None
                for i in range(3):
                    state = env.reset()
                    ep_rewards = self.memory_generate(env, state, self.ppo, params=theta_prime)
                    theta_prime = self.ppo.inner_update(self.memory,num_steps=1, inner_lr=0.01, params=theta_prime)
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
                ,"loss_std":convergence_checker.std_dev()
            }
            self.iter_log(iteration, scalars)
            step_count += 1

        print(step_counts)


if __name__ == "__main__":
    trainer = MultiTaskTrainer(configs)
    trainer.train()

