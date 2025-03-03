from train.base import *

class VAETrainer(Trainer):

    def __init__(self, config):
        super().__init__(config)
        self.ppo = PPO_initialize(config)
        if config.data_source == "SD1":
            self.env = FJSPEnvForVariousOpNums(self.n_j, self.n_m)
        else:
            self.env = FJSPEnvForSameOpNums(self.n_j, self.n_m)
        
    def train(self):
        setup_seed(self.seed_train)
        self.record = float('inf')
        self.makespans = []
        self.meta_loss = []
        self.mean_rewards = []
        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"vali data :{self.vali_data_path}")
        print("\n")
        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            ep_st = time.time()

            # resampling the training data
            if i_update % self.reset_env_timestep == 0:
                dataset_job_length, dataset_op_pt = self.sample_training_instances()
                state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)
            else:
                state = self.env.reset()
            
            ep_rewards = - deepcopy(self.env.init_quality)

            while True:
                self.memory.push(state)

                with torch.no_grad():
                    pi_envs, vals_envs, *ohers = self.ppo.policy_old(fea_j=state.fea_j_tensor,
                                                             op_mask=state.op_mask_tensor,
                                                             candidate=state.candidate_tensor,
                                                             fea_m=state.fea_m_tensor,
                                                             mch_mask=state.mch_mask_tensor,
                                                             comp_idx=state.comp_idx_tensor,
                                                             dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                                                             fea_pairs=state.fea_pairs_tensor)
                action_envs, action_logprob_envs = sample_action(pi_envs)
                # state transition
                state, reward, done = self.env.step(actions=action_envs.cpu().numpy())
                ep_rewards += reward
                reward = torch.from_numpy(reward).to(device)
                # collect the transition
                self.memory.done_seq.append(torch.from_numpy(done).to(device))
                self.memory.reward_seq.append(reward)
                self.memory.action_seq.append(action_envs)
                self.memory.log_probs.append(action_logprob_envs)
                self.memory.val_seq.append(vals_envs.squeeze(1))
                if done.all():
                    break

            loss, v_loss, vae_loss = self.ppo.update(self.memory)
            self.memory.clear_memory()

            mean_rewards_all_env = np.mean(ep_rewards)
            mean_makespan_all_env = np.mean(self.env.current_makespan)
            self.mean_rewards.append(mean_rewards_all_env)
            self.makespans.append(mean_makespan_all_env)

            if (i_update + 1) % self.validate_timestep == 0:
                self.save_model()

            ep_et = time.time()
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    i_update + 1, mean_rewards_all_env, mean_makespan_all_env, loss, ep_et - ep_st))

            scalars={
                'Loss/train': loss
                ,'makespan_train':mean_makespan_all_env
                # ,'makespan_validate':vali_result
            }
            
            self.iter_log(i_update, scalars)


    def save_logs(self):
        
        with open(f"./train_log/makespans{self.log_timestamp}.txt", "w") as f:
            print(self.makespans, file=f)
        with open(f"./train_log/mean_rewards{self.log_timestamp}.txt", "w") as f:
            print(self.mean_rewards, file=f)


if __name__ == '__main__':
    trainer = VAETrainer(configs)
    trainer.train()