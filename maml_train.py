from train_base import *

class SDTrainer(Trainer):

    def __init__(self, config) -> None:
        super().__init__(config)
        self.ppo = PPO_initialize()

    def train(self):
        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"vali data :{self.vali_data_path}")
        print("\n")
        setup_seed(self.seed_train)
        self.log = []
        self.record = float('inf')
        self.validation_log = []
        self.train_st = time.time()
        

        for iteration in tqdm(range(self.meta_iterations), file=sys.stdout, desc="progress"):

            ep_st = time.time()

            inner_ppos = []

            for task in range(self.num_tasks):
                env = self.envs[task]
                if iteration % self.reset_env_timestep == 0:
                    dataset_job_length, dataset_op_pt = self.sample_training_instances()
                    state = env.set_initial_data(dataset_job_length, dataset_op_pt)
                else:
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
            
            if iteration < 2: vali_result = makespan 

            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    iteration + 1, mean_rewards_all_env, makespan, loss, ep_et - ep_st))

            if (iteration + 1) % self.validate_timestep == 0:
                vali_result = self.valid_model()
                tqdm.write(f'The validation quality is: {vali_result} (best : {self.record})')

            self.iter_log(iteration, loss, makespan, vali_result)

    def fast_adapt_valid_model(self):
        
        if self.data_source == "SD1":
            vali_result = self.validate_envs_with_various_op_nums().mean()
        else:
            vali_result = self.validate_envs_with_same_op_nums().mean()

        if vali_result < self.record:
            self.save_model()
            self.record = vali_result

        self.validation_log.append(vali_result)
        self.save_validation_log()
        return vali_result



if __name__ == "__main__":
    trainer = SDTrainer(configs)
    trainer.train()


