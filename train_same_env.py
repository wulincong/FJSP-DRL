from train_base import *

writer_ = SummaryWriter(configs.logdir+"_")  # 创建一个SummaryWriter对象，用于记录日志

class SameEnvTrainer(Trainer):

    def __init__(self, config) -> None:
        super().__init__(config)
        self.ppo = PPO_initialize()
        self.ppo_ = PPO_initialize()

    def train(self):
        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"vali data :{self.vali_data_path}")
        print("\n")

        self.log = []
        self.record = float('inf')
        self.record_ = float('inf')
        self.validation_log = []
        self.train_st = time.time()
        iter_ = 0
        for iteration in tqdm(range(self.meta_iterations), file=sys.stdout, desc="progress"):

            ep_st = time.time()

            inner_ppos = []

            for task in range(self.num_tasks):
                # meta子模型训练
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
                
                # ppo_训练
                state = env.reset()
                ep_rewards = self.memory_generate(env, state, self.ppo_)
                loss_, v_loss_ = self.ppo_.update(self.memory)
                mean_rewards_all_env_ = np.mean(ep_rewards)
                mean_makespan_all_env_ = np.mean(env.current_makespan)
                iter_ += 1

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
            if iteration < 2: vali_result = vali_result_ = makespan 
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t makespan_:{:.2f} Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    iteration + 1, mean_rewards_all_env, makespan, mean_makespan_all_env_, loss, ep_et - ep_st))

            if (iteration + 1) % self.validate_timestep == 0:
                vali_result = self.valid_model()
                tqdm.write(f'The validation quality is: {vali_result} (best : {self.record})')
            
            # validate the trained model of ppo_
            if (iteration + 1) % self.validate_timestep == 0:
                if self.data_source == "SD1":
                    vali_result_ = self.validate_envs_with_various_op_nums_(self.ppo_).mean()
                else:
                    vali_result_ = self.validate_envs_with_same_op_nums_(self.ppo_).mean()

                if vali_result_ < self.record_:
                    self.save_model_(self.ppo_)
                    self.record_ = vali_result_

                tqdm.write(f'The validation quality of ppo_ is: {vali_result} (best : {self.record_})')

            self.iter_log(iteration, loss, makespan, vali_result)
            self.iter_log(iteration, loss_, mean_makespan_all_env_, vali_result_, writer_)

    def validate_envs_with_various_op_nums_(self, model):
        model.policy.eval()
        state = self.vali_env.reset()
        while True:
            with torch.no_grad():
                batch_idx = ~torch.from_numpy(self.vali_env.done_flag)
                pi, _ = model.policy(fea_j=state.fea_j_tensor[batch_idx],  # [sz_b, N, 8]
                                        op_mask=state.op_mask_tensor[batch_idx],
                                        candidate=state.candidate_tensor[batch_idx],  # [sz_b, J]
                                        fea_m=state.fea_m_tensor[batch_idx],  # [sz_b, M, 6]
                                        mch_mask=state.mch_mask_tensor[batch_idx],  # [sz_b, M, M]
                                        comp_idx=state.comp_idx_tensor[batch_idx],  # [sz_b, M, M, J]
                                        dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx],  # [sz_b, J, M]
                                        fea_pairs=state.fea_pairs_tensor[batch_idx])  # [sz_b, J, M]
                action = greedy_select_action(pi)
                state, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break
        model.policy.train()
        return self.vali_env.current_makespan


    def validate_envs_with_same_op_nums_(self, model):
        model.policy.eval()
        state = self.vali_env.reset()

        while True:

            with torch.no_grad():
                
                pi, _ = model.policy(fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                                        op_mask=state.op_mask_tensor,
                                        candidate=state.candidate_tensor,  # [sz_b, J]
                                        fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                                        mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                                        comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                                        dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                                        fea_pairs=state.fea_pairs_tensor)  # [sz_b, J, M]

            action = greedy_select_action(pi)
            state, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        model.policy.train()
        return self.vali_env.current_makespan

    def save_model_(self, model):
        """
            save the model
        """
        torch.save(model.policy.state_dict(), f'./trained_network/{self.data_source}'
                                                 f'/{self.model_name}_.pth')


    def iter_log(self, iteration, loss, makespan_train, makespan_validate, writer=writer):
        writer.add_scalar('Loss/train', loss, iteration)
        writer.add_scalar('makespan_train', makespan_train, iteration)
        writer.add_scalar('makespan_validate', makespan_validate, iteration)


if __name__ == "__main__":
    trainer = SameEnvTrainer(configs)
    trainer.train()


