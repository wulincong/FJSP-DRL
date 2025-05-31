###从头训练
from train.Trainer import *
from model.VariVAEPPO import VariVAEPPO

class VariVAEMultiTaskTrainer(MultiTaskTrainer):
    def __init__(self, config):
        """
        Initialize the VariVAE Multi-task Trainer.
        :param config: Configuration parameters for the training process.
        """
        super().__init__(config)
        self.config = config
        self.env = None
        self.ppo = VariVAEPPO(config)  # Use VariVAEPPO as the policy optimizer
        self.meta_optimizer = torch.optim.Adam(self.ppo.policy.parameters(), lr=config.meta_lr)

        # Task-specific configurations
        self.num_tasks = config.num_tasks
        self.n_j_options = config.n_j_options
        self.n_m_options = config.n_m_options
        self.op_per_job_options = config.op_per_job_options
        self.valid_memorys = [Memory(gamma=config.gamma, gae_lambda=config.gae_lambda) for _ in range(self.num_tasks)]
        self.train_param_list = []

        print("n_j_options: ", self.n_j_options)
        print("n_m_options: ", self.n_m_options)

    def reset_env_tasks(self, use_candidate = False):
        """
        Reset the environments for each task.
        """
        self.tasks = []
        candidate_j = [11, 23, 23]
        candidate_m = [8, 8, 17]
        for _ in range(self.num_tasks):
            n_j = n_m = 0
            if use_candidate:
                n_j = candidate_j[_]
                n_m = candidate_m[_]
            else:
                while n_j <= n_m:
                    n_j = random.randint(5, 20)
                    n_m = random.randint(5, 15)
            print(f"generate env task n_j, n_m = {n_j, n_m}")
            if self.data_source == 'SD1':
                env = FJSPEnvForVariousOpNums(n_j, n_m)
            elif self.data_source == 'SD2':
                env = FJSPEnvForSameOpNums(n_j, n_m)
            else:
                print("Invalid data source.")
            env.set_initial_data(*self.sample_training_instances(n_j, n_m))
            self.tasks.append(env)

    def train(self):
        """
        Train the VariVAE model using MAML-based meta-learning.
        """
        setup_seed(self.config.seed_train)
        self.log = []
        self.record = float('inf')
        self.train_st = time.time()

        self.makespans = [[] for _ in range(self.num_tasks)]
        self.meta_loss = []
        self.mean_rewards = []
        self.reset_env_tasks()
        for iteration in range(self.config.meta_iterations):
            ep_st = time.time()

            # Reset environment tasks periodically
            # if iteration % self.config.reset_env_timestep == 0:
            #     self.reset_env_tasks()

            iteration_policies = []
            mean_rewards_all_env_list = []

            for task_idx in range(self.num_tasks):
                # Step 1: Train on support set
                env = self.tasks[task_idx]
                state = env.reset()
                self.memory = Memory(gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)
                ep_rewards = self.memory_generate(env, state, self.ppo)

                # Inner update to get task-specific parameters (theta_prime)
                # theta_prime = self.ppo.inner_update(self.memory, self.config.k_epochs, self.config.task_lr)
                epoch_loss, v_loss, vae_loss = self.ppo.update(self.memory)
                # Step 2: Evaluate on query set
                # ep_rewards = self.memory_generate(env, state, self.ppo, params=theta_prime,
                #                                   memory=self.valid_memorys[task_idx])  # Collect query set

                mean_rewards_all_env = np.mean(ep_rewards)
                mean_rewards_all_env_list.append(mean_rewards_all_env)
                self.makespans[task_idx].append(np.mean(env.current_makespan))
                # iteration_policies.append(theta_prime)
                state = env.reset()

            mean_reward_iter = np.mean(mean_rewards_all_env_list)

            # Step 3: Compute meta loss and optimize
            # loss = self.ppo.meta_optimize(self.valid_memorys, iteration_policies)
            # self.meta_loss.append(float(loss))

            ep_et = time.time()
            makespan_sum = np.sum([np.mean(lst) for lst in self.makespans])

            tqdm.write(
                f'Episode {iteration + 1} reward: {mean_reward_iter:.4f} '
                f'Meta loss: {epoch_loss:.8f}, VAE loss: {vae_loss:.8f} '
                f'Training time: {ep_et - ep_st:.2f}s Makespan:{makespan_sum}'
            )
            self.mean_rewards.append(mean_reward_iter)
            self.meta_loss.append(epoch_loss)
            # Save model periodically
            if makespan_sum < self.record:
                self.record = makespan_sum
                self.save_model()

            # Periodically log parameters for analysis
            # if iteration % 10 == 0:
            #     self.train_param_list.append([p.clone().detach() for p in self.ppo.policy.actor.parameters()])

            scalars = {
                'Loss/train': epoch_loss,
                'makespan_train': np.mean(makespan_sum),
            }
            # 添加模型结构

            self.iter_log(iteration, scalars)
            self.memory.clear_memory()

        # Save logs
        self.save_logs()


    def save_logs(self):
        """
        Save training logs to files.
        """
        with open(f"./train_log/makespans{self.log_timestamp}.txt", "w") as f:
            print(self.makespans, file=f)
        with open(f"./train_log/mean_rewards{self.log_timestamp}.txt", "w") as f:
            print(self.mean_rewards, file=f)
        with open(f"./train_log/meta_loss{self.log_timestamp}.txt", "w") as f:
            print(self.meta_loss, file=f)

    def save_model(self):
        """
        Save the VariVAE model.
        """
        torch.save(self.ppo.policy.state_dict(), f"./trained_network/vari_vae_{self.data_source}_{self.log_timestamp}.pth")


def main():
    trainer = VariVAEMultiTaskTrainer(configs)
    trainer.train()


if __name__ == '__main__':
    main()

