#从已有模型finetuning
from train.base import *

class DANTrainer(Trainer):
    def __init__(self, config):

        super().__init__(config)
        self.env = FJSPEnvForSameOpNums(self.n_j, self.n_m)
        self.finetuning_model = f'./trained_network/{config.model_source}/{config.finetuning_model}.pth'
        self.ppo = PPO_initialize()
        self.ppo.policy.load_state_dict(torch.load(self.finetuning_model, map_location='cuda'))
        self.ppo.policy_old = deepcopy(self.ppo.policy)

        print(f"finetuning model name :{self.finetuning_model}")

    def check_file_count(self):
        # 列出 instance_dir 目录中的所有文件
        files = [f for f in os.listdir(self.instance_dir) if os.path.isfile(os.path.join(self.instance_dir, f))]

        # 计算文件数量
        file_count = len(files)

        # 比较文件数量和 num_envs
        return file_count == self.num_envs

    def train(self):
        """
            train the model following the config
        """
        setup_seed(self.seed_train)
        self.log = []
        self.validation_log = []
        self.record = float('inf')

        self.makespan_log = []
        self.loss_log = []

        self.train_st = time.time()

        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            ep_st = time.time()

            # resampling the training data
            if i_update  == 0:
                data_path = f"{self.instance_dir}/"

                if self.check_file_count():
                    dataset_job_length, dataset_op_pt = load_data_from_files(data_path)
                else:
                    import logging
                    logging.info("实例数量与环境数不同，将重新生成实例！")

                    dataset_job_length, dataset_op_pt = self.sample_training_instances()
                
                state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)
            else:
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
            # print(self.env.current_makespan)
            if i_update < 2: vali_result = mean_makespan_all_env 

            # save the mean rewards of all instances in current training data
            self.log.append([i_update, mean_rewards_all_env])
            self.makespan_log.append(list(self.env.current_makespan))
            self.loss_log.append(loss)

            ep_et = time.time()
            # print the reward, makespan, loss and training time of the current episode
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    i_update + 1, mean_rewards_all_env, mean_makespan_all_env, loss, ep_et - ep_st))
            scalars = {f"makespan_{i}":m  for i, m in zip(range(self.num_envs), self.env.current_makespan)}
            scalars.update({
                'Loss/train': loss
                ,'makespan_train':mean_makespan_all_env
                ,'makespan_validate':vali_result
            })
            
            self.iter_log(i_update, scalars)

        self.train_et = time.time()

        # log results
        # self.save_training_log()
        self.save_finetuning_log()


def main():
    trainer = DANTrainer(configs)
    trainer.train()


if __name__ == '__main__':
    main()

