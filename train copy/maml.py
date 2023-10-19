from train.base import *

class MAMLTrainer(Trainer):
    
    def __init__(self, config) -> None:
        self.envs = [FJSPEnvForSameOpNums(self.n_j, self.n_m) for _ in range(self.num_tasks)]
        super().__init__(config)
        self.ppo = PPO_initialize()
    
    def train(self):
        setup_seed(self.seed_train)

        for iteration in range(self.meta_iterations):
            for task in range(self.num_tasks):
                env = self.envs[task]
                if iteration % self.reset_env_timestep == 0:
                    dataset_job_length, dataset_op_pt = self.sample_training_instances()
                    state = env.set_initial_data(dataset_job_length, dataset_op_pt)
                else:
                    state = env.reset()
                original_parameters = deepcopy(self.ppo.policy.parameters())
                self.memory_generate(env, state, self.ppo)
                self.ppo.inner_update(self.memory)
                # 任务适应评估

                # 计算元更新损失

                # 执行元更新


