###从头训练
from train.base import *

class DANTrainer(Trainer):
    def __init__(self, config):

        super().__init__(config)
        self.ppo = PPO_initialize(config)
        self.env = FJSPEnvForSameOpNums(self.n_j, self.n_m)


    def train(self):
        """
            train the model following the config
        """
        setup_seed(self.seed_train)
        self.log = []
        self.validation_log = []
        self.record = float('inf')

        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"vali data :{self.vali_data_path}")
        print("\n")

        self.train_st = time.time()
        self.train_params = []
        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            ep_st = time.time()

            # resampling the training data
            if i_update % self.reset_env_timestep == 0:
                dataset_job_length, dataset_op_pt = self.sample_training_instances()
                state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)
            else:
                if i_update == 10:
                    self.env.render_gantt_chart()
                state = self.env.reset()

            ep_rewards = - deepcopy(self.env.init_quality)

            while True:

                # state store
                self.memory.push(state)
                with torch.no_grad():

                    pi_envs, vals_envs = self.ppo.policy_old(fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                                                             op_mask=state.op_mask_tensor,  # [sz_b, N, N]
                                                             candidate=state.candidate_tensor,  # [sz_b, J]
                                                             fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                                                             mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                                                             comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                                                             dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                                                             fea_pairs=state.fea_pairs_tensor)  # [sz_b, J, M]

                # sample the action
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

            loss, v_loss = self.ppo.update(self.memory)
            self.memory.clear_memory()

            mean_rewards_all_env = np.mean(ep_rewards)
            mean_makespan_all_env = np.mean(self.env.current_makespan)
            if i_update < 2: vali_result = mean_makespan_all_env 

            # save the mean rewards of all instances in current training data
            self.log.append([i_update, mean_rewards_all_env])

            # validate the trained model
            if (i_update + 1) % self.validate_timestep == 0:
                self.train_params.append(list(self.ppo.policy.parameters()))
                self.save_model()

                # if self.data_source == "SD1":
                #     vali_result = self.validate_envs_with_various_op_nums().mean()
                # else:
                #     vali_result = self.validate_envs_with_same_op_nums().mean()

                # if vali_result < self.record:
                #     self.save_model()
                #     self.record = vali_result

                # self.validation_log.append(vali_result)
                # self.save_validation_log()
                # tqdm.write(f'The validation quality is: {vali_result} (best : {self.record})')

            ep_et = time.time()
            # print the reward, makespan, loss and training time of the current episode
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    i_update + 1, mean_rewards_all_env, mean_makespan_all_env, loss, ep_et - ep_st))
            
            scalars={
                'Loss/train': loss
                ,'makespan_train':mean_makespan_all_env
                # ,'makespan_validate':vali_result
            }
            
            self.iter_log(i_update, scalars)

        self.train_et = time.time()

        # log results
        self.save_training_log()



def main():
    trainer = DANTrainer(configs)
    trainer.train()


if __name__ == '__main__':
    main()

