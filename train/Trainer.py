from train.base import *
import numpy as np

class MultiTaskTrainer(Trainer):
    def __init__(self, config) -> None:

        super().__init__(config)
        
        self.env = None
        self.ppo = PPO_initialize(config)
        self.meta_optimizer = torch.optim.Adam(self.ppo.policy.parameters(), self.meta_lr)
        
        self.n_j_options = config.n_j_options        
        self.n_m_options = config.n_m_options
        self.op_per_job_options = config.op_per_job_options
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
        self.current_EC_record = [[] for _ in range(self.num_tasks)]
        self.mean_rewards = []
        self.meta_loss = []

        for iteration in range(self.meta_iterations):
            ep_st = time.time()

            #重置环境
            if iteration % self.reset_env_timestep == 0:
                self.tasks = []
                for _ in range(self.num_tasks):
                    env = FJSPEnvForSameOpNums(self.n_j_options[_], self.n_m_options[_])
                    env.set_initial_data(*self.sample_training_instances(
                        n_j=self.n_j_options[_], 
                        n_m=self.n_m_options[_],
                        op_per_job=self.op_per_job_options[_]
                        ))
                    self.tasks.append(env)

            loss_sum = 0
            iteration_policies = []
            mean_rewards_all_env_list = []
            for task_idx in range(self.num_tasks):
                env = self.tasks[task_idx]
                state = env.reset()
                theta_prime = None

                ep_rewards = self.memory_generate(env, state, self.ppo, theta_prime)
                theta_prime = self.ppo.inner_update(self.memory, 1, self.task_lr)
                state = env.reset()

                ep_rewards = self.memory_generate(env, state, self.ppo, params=theta_prime, memory=self.valid_memorys[task_idx]) #搜集query set

                mean_rewards_all_env = np.mean(ep_rewards)
                mean_rewards_all_env_list.append(str(mean_rewards_all_env))
                self.makespans[task_idx].append(np.mean(env.current_makespan))
                # self.current_EC_record[task_idx].append(np.mean(env.current_EC))
                iteration_policies.append(theta_prime)

            #计算meta损失
            loss = self.ppo.meta_optimize(self.valid_memorys, iteration_policies)
            self.meta_loss.append(loss)
            # self.meta_optimizer.step()
            # self.meta_optimizer.zero_grad()

            ep_et = time.time()
            makespan_sum = np.sum([np.mean(lst) for lst in self.makespans])
            print(makespan_sum)
            if iteration < 2: self.record = makespan_min = makespan_sum

            tqdm.write(
                'Episode {}\n reward: {}\n Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    iteration + 1, " ".join(mean_rewards_all_env_list), loss, ep_et - ep_st))
            self.mean_rewards.append(mean_rewards_all_env_list)
            if makespan_sum < makespan_min:
                makespan_min = makespan_sum
                self.save_model()
            #     tqdm.write(f'The validation quality is: {vali_result} (best : {self.record})')

            scalars={
                'Loss/train': loss
                ,'makespan_train':np.mean(makespan_sum)
                # ,'makespan_validate':vali_result
            }
            self.iter_log(iteration, scalars)
            self.memory.clear_memory()
        with open(f"./train_log/makespans{int(time.time())}.txt", "w") as f: print(self.makespans, file=f)
        with open(f"./train_log/mean_rewards{int(time.time())}.txt", "w") as f: print(self.mean_rewards, file=f)
        with open(f"./train_log/meta_loss{int(time.time())}.txt", "w") as f: print(self.meta_loss, file=f)
        # with open(f"./train_log/EC{int(time.time())}.txt", "w") as f: print(self.current_EC_record, file=f)

class MultiTaskECTrainer(Trainer):
    def __init__(self, config) -> None:

        super().__init__(config)
        
        self.env = None
        self.ppo = PPO_initialize(config)
        self.meta_optimizer = torch.optim.Adam(self.ppo.policy.parameters(), self.meta_lr)
        
        self.n_j_options = config.n_j_options        
        self.n_m_options = config.n_m_options
        self.op_per_job_options = config.op_per_job_options
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
        self.current_EC_record = [[] for _ in range(self.num_tasks)]
        self.mean_rewards = []
        self.meta_loss = []

        for iteration in range(self.meta_iterations):
            ep_st = time.time()

            #重置环境
            if iteration % self.reset_env_timestep == 0:
                self.tasks = []
                for _ in range(self.num_tasks):
                    env = FJSPEnvForSameOpNumsEnergy(self.n_j_options[_], self.n_m_options[_])
                    env.set_initial_data(*self.sample_training_instances(
                        n_j=self.n_j_options[_], 
                        n_m=self.n_m_options[_],
                        op_per_job=self.op_per_job_options[_]
                        ))
                    self.tasks.append(env)

            loss_sum = 0
            iteration_policies = []
            mean_rewards_all_env_list = []
            for task_idx in range(self.num_tasks):
                env = self.tasks[task_idx]
                state = env.reset()
                theta_prime = None

                ep_rewards = self.memory_generate(env, state, self.ppo, theta_prime)
                theta_prime = self.ppo.inner_update(self.memory, 1, self.task_lr)
                state = env.reset()

                ep_rewards = self.memory_generate(env, state, self.ppo, params=theta_prime, memory=self.valid_memorys[task_idx]) #搜集query set

                mean_rewards_all_env = np.mean(ep_rewards)
                mean_rewards_all_env_list.append(str(mean_rewards_all_env))
                self.makespans[task_idx].append(np.mean(env.current_makespan))
                self.current_EC_record[task_idx].append(np.mean(env.current_EC))
                iteration_policies.append(theta_prime)

            #计算meta损失
            loss = self.ppo.meta_optimize(self.valid_memorys, iteration_policies)
            self.meta_loss.append(loss)
            # self.meta_optimizer.step()
            # self.meta_optimizer.zero_grad()

            ep_et = time.time()
            makespan_sum = np.sum([np.mean(lst) for lst in self.makespans])
            print(makespan_sum)
            if iteration < 2: self.record = makespan_min = makespan_sum

            tqdm.write(
                'Episode {}\n reward: {}\n Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    iteration + 1, " ".join(mean_rewards_all_env_list), loss, ep_et - ep_st))
            self.mean_rewards.append(mean_rewards_all_env_list)
            if makespan_sum < makespan_min:
                makespan_min = makespan_sum
                self.save_model()
            #     tqdm.write(f'The validation quality is: {vali_result} (best : {self.record})')

            scalars={
                'Loss/train': loss
                ,'makespan_train':np.mean(makespan_sum)
                # ,'makespan_validate':vali_result
            }
            self.iter_log(iteration, scalars)
            self.memory.clear_memory()
        with open(f"./train_log/makespans{int(time.time())}.txt", "w") as f: print(self.makespans, file=f)
        with open(f"./train_log/mean_rewards{int(time.time())}.txt", "w") as f: print(self.mean_rewards, file=f)
        with open(f"./train_log/meta_loss{int(time.time())}.txt", "w") as f: print(self.meta_loss, file=f)
        with open(f"./train_log/EC{int(time.time())}.txt", "w") as f: print(self.current_EC_record, file=f)

class MultiTaskTrainerCustomize(Trainer):
    def __init__(self, config) -> None:

        super().__init__(config)
        
        self.env = None
        self.ppo = PPO_initialize(config)
        self.meta_optimizer = torch.optim.Adam(self.ppo.policy.parameters(), self.meta_lr)
        
        self.n_j_options = config.n_j_options
        self.n_m_options = config.n_m_options
        self.op_per_job_options = config.op_per_job_options
        self.valid_memorys = [Memory(gamma=config.gamma, gae_lambda=config.gae_lambda) for _ in range(self.num_tasks)]
        print("self.n_js: ", self.n_j_options)
        print("self.n_m_options: ",self.n_m_options)
        print("self.op_per_job_options: ",self.op_per_job_options)

    def train(self, inner_update_params_startswith):  
        '''
        inner_update_params_startswith content [ 'feature_exact', 'actor', 'critic']
        '''
        setup_seed(self.seed_train)
        self.log = []
        self.record = float('inf')
        self.train_st = time.time()
        # time.sleep(1000)
        assert self.num_tasks == len(self.n_j_options)

        self.makespans = [[] for _ in range(self.num_tasks)]
        self.mean_rewards = []
        self.meta_loss = []

        for iteration in range(self.meta_iterations):
            ep_st = time.time()

            #重置环境
            if iteration % self.reset_env_timestep == 0:
                self.tasks = []
                for _ in range(self.num_tasks):
                    env = FJSPEnvForSameOpNums(self.n_j_options[_], self.n_m_options[_])
                    env.set_initial_data(*self.sample_training_instances(
                        n_j=self.n_j_options[_], 
                        n_m=self.n_m_options[_],
                        op_per_job=self.op_per_job_options[_]
                        ))
                    self.tasks.append(env)

            loss_sum = 0
            iteration_policies = []
            mean_rewards_all_env_list = []
            vaild_mk = 0.0
            for task_idx in range(self.num_tasks):
                env = self.tasks[task_idx]
                state = env.reset()
                theta_prime = None

                ep_rewards = self.memory_generate(env, state, self.ppo, theta_prime)
                theta_prime = self.ppo.inner_update(self.memory, 4, self.task_lr)
                # print(theta_prime)
                # 遍历 self.ppo.policy 网络中的所有参数
                for name, param in self.ppo.policy.named_parameters():                    
                    # print(name)
                    if any(name.startswith(prefix) for prefix in inner_update_params_startswith):
                        # 如果 theta_prime 中存在该参数名，则更新参数
                        param.data = theta_prime[name].data

                state = env.reset()

                ep_rewards = self.memory_generate(env, state, self.ppo, params=theta_prime, memory=self.valid_memorys[task_idx]) #搜集query set

                mean_rewards_all_env = np.mean(ep_rewards)
                mean_rewards_all_env_list.append(str(mean_rewards_all_env))
                self.makespans[task_idx].append(np.mean(env.current_makespan))
                vaild_mk += np.mean(env.current_makespan)
                iteration_policies.append(theta_prime)

            #计算meta损失
            
            meta_update_params_startswith = set(['feature_exact', 'actor', 'critic']) - set(inner_update_params_startswith)
            for name, param in self.ppo.policy.named_parameters():
                if any(name.startswith(prefix) for prefix in inner_update_params_startswith):
                    param.requires_grad = False
            loss = self.ppo.meta_optimize_customize(self.valid_memorys, iteration_policies, meta_update_params_startswith)
            self.meta_loss.append(loss)
            # self.meta_optimizer.step()
            # self.meta_optimizer.zero_grad()
            for name, param in self.ppo.policy.named_parameters():
                if any(name.startswith(prefix) for prefix in inner_update_params_startswith):
                    param.requires_grad = True
            ep_et = time.time()
            print(vaild_mk)
            if iteration < 2: self.record = makespan_min = vaild_mk

            tqdm.write(
                'Episode {}\n reward: {}\n Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    iteration + 1, " ".join(mean_rewards_all_env_list), loss, ep_et - ep_st))
            self.mean_rewards.append(mean_rewards_all_env_list)
            if vaild_mk < makespan_min:
                makespan_min = vaild_mk
                self.save_model()
            #     tqdm.write(f'The validation quality is: {vali_result} (best : {self.record})')

            scalars={
                'Loss/train': loss
                ,'makespan_train':vaild_mk
                # ,'makespan_validate':vali_result
            }
            self.iter_log(iteration, scalars)
            self.memory.clear_memory()
        with open(f"./train_log/makespans{int(time.time())}.txt", "w") as f: print(self.makespans, file=f)
        with open(f"./train_log/mean_rewards{int(time.time())}.txt", "w") as f: print(self.mean_rewards, file=f)
        with open(f"./train_log/meta_loss{int(time.time())}.txt", "w") as f: print(self.meta_loss, file=f)


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


if __name__ == "__main__":
    trainer = DANTrainer(configs)
    trainer.train()

