from train.base import *
import numpy as np



class MultiTaskTrainer(Trainer):
    def __init__(self, config, ) -> None:

        super().__init__(config)
        
        self.env = None
        self.ppo = PPO_initialize(config)
        self.meta_optimizer = torch.optim.Adam(self.ppo.policy.parameters(), self.meta_lr)
        
        self.n_j_options = config.n_j_options
        self.n_m_options = config.n_m_options
        self.op_per_job_options = config.op_per_job_options
        self.valid_memorys = [Memory(gamma=config.gamma, gae_lambda=config.gae_lambda) for _ in range(self.num_tasks)]
        self.train_param_list = []
        print("self.n_js: ", self.n_j_options)
        print("self.n_m_options: ",self.n_m_options)
        print("self.op_per_job_options: ",self.op_per_job_options)

    def train(self):
        setup_seed(self.seed_train)
        self.log = []
        self.record = float('inf')
        self.train_st = time.time()
        assert self.num_tasks == len(self.n_j_options)

        self.makespans = [[] for _ in range(self.num_tasks)]
        self.current_EC_record = [[] for _ in range(self.num_tasks)]
        self.mean_rewards = []
        self.meta_loss = []

        for iteration in range(self.meta_iterations):
            ep_st = time.time()

            #重置环境

            # if iteration % self.reset_env_timestep == 0:
            #     self.tasks = []
            #     for _ in range(self.num_tasks):
            #         env = FJSPEnvForSameOpNums(self.n_j_options[_], self.n_m_options[_])
            #         env.set_initial_data(*self.sample_training_instances(
            #             n_j=self.n_j_options[_], 
            #             n_m=self.n_m_options[_],
            #             op_per_job=self.op_per_job_options[_]
            #             ))
            #         self.tasks.append(env)

            if iteration % self.reset_env_timestep == 0:
                self.tasks = []
                for _ in range(self.num_tasks):
                    n_j = n_m = 0
                    while n_j <= n_m: 
                        n_j = random.randint(5, 20)
                        n_m = random.randint(5, 20)
                    print(n_j, n_m)
                    env = FJSPEnvForSameOpNums(n_j, n_m)
                    env.set_initial_data(*self.sample_training_instances(n_j, n_m))
                    self.tasks.append(env)

            iteration_policies = []
            mean_rewards_all_env_list = []
            for task_idx in range(self.num_tasks):
                env = self.tasks[task_idx]
                state = env.reset()

                ep_rewards = self.memory_generate(env, state, self.ppo)
                theta_prime = self.ppo.inner_update(self.memory, 4, self.task_lr)
                state = env.reset()

                ep_rewards = self.memory_generate(env, state, self.ppo, params=theta_prime, memory=self.valid_memorys[task_idx]) #搜集query set

                mean_rewards_all_env = np.mean(ep_rewards)
                mean_rewards_all_env_list.append(mean_rewards_all_env)
                self.makespans[task_idx].append(np.mean(env.current_makespan))
                # self.current_EC_record[task_idx].append(np.mean(env.current_EC))
                iteration_policies.append(theta_prime)
            mean_reward_iter = np.mean(mean_rewards_all_env_list)
            #计算meta损失
            loss = self.ppo.meta_optimize(self.valid_memorys, iteration_policies)
            self.meta_loss.append(float(loss))
            # self.meta_optimizer.step()
            # self.meta_optimizer.zero_grad()

            ep_et = time.time()
            makespan_sum = np.sum([np.mean(lst) for lst in self.makespans])
            # print(makespan_sum)
            if iteration < 2: self.record = makespan_min = makespan_sum

            tqdm.write(
                'Episode {} reward: {} Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    iteration + 1, mean_reward_iter, loss, ep_et - ep_st))
            self.mean_rewards.append(mean_reward_iter)
            if makespan_sum < makespan_min:
                makespan_min = makespan_sum
            self.save_model()
            #     tqdm.write(f'The validation quality is: {vali_result} (best : {self.record})')
            if iteration % 10 == 0: self.train_param_list.append([p.clone().detach() for p in self.ppo.policy.actor.parameters()])
            scalars={
                'Loss/train': loss
                ,'makespan_train':np.mean(makespan_sum)
                # ,'makespan_validate':vali_result
            }
            self.iter_log(iteration, scalars)
            self.memory.clear_memory()

        with open(f"./train_log/makespans{self.log_timestamp}.txt", "w") as f: print(self.makespans, file=f)
        with open(f"./train_log/mean_rewards{self.log_timestamp}.txt", "w") as f: print(self.mean_rewards, file=f)
        with open(f"./train_log/meta_loss{self.log_timestamp}.txt", "w") as f: print(self.meta_loss, file=f)
        # with open(f"./train_log/EC{int(time.time())}.txt", "w") as f: print(self.current_EC_record, file=f)


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

        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            ep_st = time.time()

            # resampling the training data
            if i_update % self.reset_env_timestep == 0:
                dataset_job_length, dataset_op_pt = self.sample_training_instances()
                state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)
            else:
                state = self.env.reset()

            ep_rewards = self.memory_generate(self.env, state, self.ppo)

            loss, v_loss = self.ppo.update(self.memory)
            self.memory.clear_memory()

            mean_rewards_all_env = np.mean(ep_rewards)
            mean_makespan_all_env = np.mean(self.env.current_makespan)
            if i_update < 2: vali_result = mean_makespan_all_env 

            # save the mean rewards of all instances in current training data
            self.log.append([i_update, mean_rewards_all_env])

            # validate the trained model
            if (i_update + 1) % self.validate_timestep == 0:
                # self.save_model()

                if self.data_source == "SD1":
                    vali_result = self.validate_envs_with_various_op_nums().mean()
                else:
                    vali_result = self.validate_envs_with_same_op_nums().mean()

                if vali_result < self.record:
                    self.save_model()
                    self.record = vali_result

                self.validation_log.append(vali_result)
                self.save_validation_log()
                tqdm.write(f'The validation quality is: {vali_result} (best : {self.record})')

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


class PretrainTrainer(MultiTaskTrainer):

    def train(self, ):
        setup_seed(self.seed_train)
        for iteration in range(self.meta_iterations):
            ep_st = time.time()

            #重置环境
            if iteration % self.reset_env_timestep == 0:
                self.tasks = []
                for _ in range(self.num_tasks):
                    
                    env = FJSPEnvForSameOpNums(self.n_j_options[_], self.n_m_options[_])
                    env.set_initial_data(*self.sample_training_instances(
                        n_j=random.randint(5, 26), 
                        n_m=random.randint(5, 26),
                        op_per_job=10
                        ))
                    self.tasks.append(env)

            for task_idx in range(self.num_tasks):
                env = self.tasks[task_idx]
                state = env.reset()
                ep_rewards = self.memory_generate(env, state, self.ppo)
                loss, v_loss = self.ppo.update(self.memory)

            self.save_model()
            ep_et = time.time()
            tqdm.write(
                'Episode {}\n  Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    iteration + 1, loss, ep_et - ep_st))

class FinetuningTrainer(Trainer):
    
    def __init__(self, config):

        super().__init__(config)
        self.env = FJSPEnvForSameOpNums(self.n_j, self.n_m)
        self.finetuning_model = f'./trained_network/{config.model_source}/{config.finetuning_model}.pth'
        self.ppo = PPO_initialize(config)
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

            ep_rewards = self.memory_generate(self.env, state, self.ppo)

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

class FinetuningTest(Trainer):
    
    def __init__(self, config):

        super().__init__(config)
        self.env = FJSPEnvForSameOpNums(self.n_j, self.n_m)
        self.finetuning_model = f'./trained_network/{config.model_source}/{config.finetuning_model}.pth'
        self.ppo = PPO_initialize(config)
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

            ep_rewards = self.memory_generate(self.env, state, self.ppo)

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


class DANMkEcTrainer(Trainer):
    def __init__(self, config):

        super().__init__(config)
        self.env = FJSPEnvForSameOpNumsEnergy(self.n_j, self.n_m, self.config.factor_Mk, self.config.factor_Ec)
        # self.finetuning_model = f'../trained_network/SD2/10x5+mix.pth'
        self.ppo = PPO_initialize(config)
        # print(self.finetuning_model)


    def sample_training_instances(self, n_j=None,n_m=None, op_per_job=None ):
        """
            sample training instances following the config, 
            the sampling process of SD1 data is imported from "songwenas12/fjsp-drl" 
        :return: new training instances
        """
        if n_j is None: n_j = self.n_j
        if n_m is None: n_m = self.n_m
        if op_per_job is None: op_per_job = self.config.op_per_job
        prepare_JobLength = [random.randint(self.op_per_job_min, self.op_per_job_max) for _ in range(n_j)]
        dataset_JobLength = [] 
        dataset_OpPT = []
        dataset_mch_working_power = []
        dataset_mch_idle_power = []
        
        for i in range(self.num_envs): # 20

            JobLength, OpPT, _, mch_working_power, mch_idle_power = SD2_instance_generator_EMconflict(config=self.config, n_j = n_j, n_m = n_m, op_per_job=op_per_job)  

            dataset_JobLength.append(JobLength)
            dataset_OpPT.append(OpPT)
            dataset_mch_working_power.append(mch_working_power)
            dataset_mch_idle_power.append(mch_idle_power)

        # print("len of sample_training_instances/dataset_OpPT:", len(dataset_OpPT))
        return dataset_JobLength, dataset_OpPT, dataset_mch_working_power, dataset_mch_idle_power


    def train(self):
        """
            train the model following the config
        """
        setup_seed(self.seed_train)
        self.log = []
        self.validation_log = []
        self.record = float('inf')
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
    
        # print(f"model name :{self.finetuning_model}")
        print(f"vali data :{self.vali_data_path}")
        print("\n")

        self.train_st = time.time()
        current_EC_record = []
        mean_rewards_all_env_record = []
        mean_makespan_all_env_record = []
        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            ep_st = time.time()
            if i_update == 990: 
                print(990)
            # if i_update % self.reset_env_timestep == 0:
            if i_update == 0:
                dataset_job_length, dataset_op_pt, dataset_mch_working_power, dataset_mch_idle_power = self.sample_training_instances()
                state = self.env.set_initial_data(dataset_job_length, dataset_op_pt, dataset_mch_working_power, dataset_mch_idle_power )
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
                # self.env.render_pygame()
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
            # mean_rewards_all_env_record.append(mean_rewards_all_env)
            mean_makespan_all_env = np.mean(self.env.current_makespan)
            # mean_makespan_all_env_record.append(mean_makespan_all_env)
            current_makespan_normal = np.mean(self.env.current_makespan_normal)
            current_EC = np.mean(self.env.current_EC)
            # current_EC_record.append(current_EC)
            # print(self.env.current_makespan)
            if i_update < 2: vali_result = mean_makespan_all_env 

            # save the mean rewards of all instances in current training data
            # self.log.append([i_update, mean_rewards_all_env.mean()])

            ep_et = time.time()
            # print the reward, makespan, loss and training time of the current episode
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}, current EC: {:.5f}'.format(
                    i_update + 1, mean_rewards_all_env, mean_makespan_all_env, loss, ep_et - ep_st, current_EC))
            # scalars = {f"makespan_{i}":m  for i, m in zip(range(self.num_envs), self.env.current_makespan)}
            scalars = {}
            scalars.update({
                'Loss/train': loss
                ,'makespan_train':mean_makespan_all_env
                # ,'makespan_validate':vali_result
                ,'current_EC/current_EC': current_EC
                ,'current_EC/min_EC': self.env.total_min_energy.mean()
                ,"EC_MK": current_EC
                ,"rewards": mean_rewards_all_env
            })
            
            self.iter_log(i_update, scalars)
            # mk_normal_scalars = {"EC_MK": current_makespan_normal}
            # mk_writer = SummaryWriter(self.logdir+"mk")
            # self.iter_log(i_update, mk_normal_scalars, mk_writer)
            self.save_model()

        self.train_et = time.time()

        # log results
        # self.save_training_log()

        ##draw
        with open(f"./train_log/makespans{int(time.time())}.txt", "w") as f: print(mean_makespan_all_env_record, file=f)
        with open(f"./train_log/mean_rewards{int(time.time())}.txt", "w") as f: print(mean_rewards_all_env_record, file=f)
        # with open(f"./train_log/meta_loss{int(time.time())}.txt", "w") as f: print(self.meta_loss, file=f)
        with open(f"./train_log/EC{int(time.time())}.txt", "w") as f: print(current_EC_record, file=f)

class MultiTaskTrainerEc(DANMkEcTrainer):
    def __init__(self, config, ) -> None:

        super().__init__(config)
        
        self.env = None
        self.ppo = PPO_initialize(config)
        self.meta_optimizer = torch.optim.Adam(self.ppo.policy.parameters(), self.meta_lr)
        
        self.n_j_options = config.n_j_options        
        self.n_m_options = config.n_m_options
        self.op_per_job_options = config.op_per_job_options
        self.factor_Mk = config.factor_Mk
        self.factor_Ec = config.factor_Ec
        self.valid_memorys = [Memory(gamma=config.gamma, gae_lambda=config.gae_lambda) for _ in range(self.num_tasks)]
        print("self.n_js: ", self.n_j_options)
        print("self.n_m_options: ",self.n_m_options)
        print("self.op_per_job_options: ",self.op_per_job_options)


    def train(self):
        setup_seed(self.seed_train)
        self.log = []
        self.record = float('inf')
        self.train_st = time.time()
        # time.sleep(1000)
        assert self.num_tasks == len(self.n_j_options)

        self.makespans = [[] for _ in range(self.num_tasks)]
        self.Ec_records = [[] for _ in range(self.num_tasks)]
        self.mean_rewards = []
        self.meta_loss = []
        max_mean_rewards_all_task = -1e10
        for iteration in range(self.meta_iterations):
            ep_st = time.time()

            #重置环境
            # if iteration % self.reset_env_timestep == 0:
            if iteration == 0:
                self.tasks = []
                for _ in range(self.num_tasks):
                    env = FJSPEnvForSameOpNumsEnergy(self.n_j_options[_], self.n_m_options[_], 
                                                     self.factor_Mk, self.factor_Ec)
                    env.set_initial_data(*self.sample_training_instances(
                        n_j=self.n_j_options[_], 
                        n_m=self.n_m_options[_],
                        op_per_job=self.op_per_job_options[_]
                        ))
                    self.tasks.append(env)

            loss_sum = 0
            iteration_policies = []
            mean_rewards_all_env_list = []
            span_rewards_list = []
            for task_idx in range(self.num_tasks):
                env = self.tasks[task_idx]
                state = env.reset()
                theta_prime = None

                ep_rewards1 = self.memory_generate(env, state, self.ppo, theta_prime)
                theta_prime = self.ppo.inner_update(self.memory, 4, self.task_lr)
                state = env.reset()

                ep_rewards2 = self.memory_generate(env, state, self.ppo, params=theta_prime, memory=self.valid_memorys[task_idx]) #搜集query set

                mean_rewards_all_env = np.mean(ep_rewards2)
                mean_rewards_all_env_list.append(mean_rewards_all_env)
                self.makespans[task_idx].append(np.mean(env.current_makespan))
                self.Ec_records[task_idx].append(np.mean(env.current_EC))
                iteration_policies.append(theta_prime)
                span_rewards_list.append(np.mean(ep_rewards2 - ep_rewards1))
            #计算meta损失
            loss = self.ppo.meta_optimize(self.valid_memorys, iteration_policies)
            self.meta_loss.append(float(loss))
            self.mean_rewards.append(np.mean(mean_rewards_all_env_list))

            ep_et = time.time()

            tqdm.write(
                'Episode {}  reward: {} Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    iteration + 1, np.mean(mean_rewards_all_env_list), loss, ep_et - ep_st))
            if np.mean(mean_rewards_all_env_list) > max_mean_rewards_all_task:
                self.save_model()
                max_mean_rewards_all_task = np.mean(mean_rewards_all_env_list)
                print(f"iteration {iteration} saved model")
            self.memory.clear_memory()

        with open(f"./train_log/makespans{self.log_timestamp}.txt", "w") as f: print(self.makespans, file=f)
        with open(f"./train_log/mean_rewards{self.log_timestamp}.txt", "w") as f: print(self.mean_rewards, file=f)
        with open(f"./train_log/meta_loss{self.log_timestamp}.txt", "w") as f: print(self.meta_loss, file=f)
        with open(f"./train_log/EC{self.log_timestamp}.txt", "w") as f: print(self.Ec_records, file=f)


class PretrainTrainerEc(MultiTaskTrainerEc):

    def train(self, ):
        setup_seed(self.seed_train)
        self.n_j_options = self.n_j_options[::-1]
        self.n_m_options = self.n_m_options[::-1]
        for iteration in range(self.meta_iterations):
            ep_st = time.time()

            #重置环境
            if iteration % self.reset_env_timestep == 0:
                self.tasks = []
                for _ in range(self.num_tasks):
                    
                    env = FJSPEnvForSameOpNumsEnergy(self.n_j_options[_], self.n_m_options[_], 
                                                     self.factor_Mk, self.factor_Ec)
                    env.set_initial_data(*self.sample_training_instances(
                        n_j=self.n_j_options[_], 
                        n_m=self.n_m_options[_],
                        op_per_job=self.op_per_job_options[_]
                        ))
                    self.tasks.append(env)

            for task_idx in range(self.num_tasks):
                env = self.tasks[task_idx]
                state = env.reset()
                ep_rewards = self.memory_generate(env, state, self.ppo)
                loss, v_loss = self.ppo.update(self.memory)

            self.save_model()
            ep_et = time.time()
            tqdm.write(
                'Episode {}\n  Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    iteration + 1, loss, ep_et - ep_st))


if __name__ == "__main__":
    trainer = DANTrainer(configs)
    trainer.train()

