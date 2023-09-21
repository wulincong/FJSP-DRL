from common_utils import *
from params import configs
from tqdm import tqdm
from data_utils import load_data_from_files, CaseGenerator, SD2_instance_generator
from common_utils import strToSuffix, setup_seed
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from copy import deepcopy
import os
import random
import time
import sys
from model.PPO import PPO_initialize
from model.PPO import Memory
from model.sub_layers import *

str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

device = torch.device(configs.device)


class Trainer:
    def __init__(self, config, actor, critic, task):

        self.n_j = config.n_j # Number of jobs of the instance
        self.n_m = config.n_m # Number of machines of the instance
        self.low = config.low # Lower Bound of processing time(PT)
        self.high = config.high # Upper Bound of processing time
        self.op_per_job_min = int(0.8 * self.n_m) 
        self.op_per_job_max = int(1.2 * self.n_m)
        self.data_source = config.data_source # Suffix of test data 测试数据的后缀
        self.config = config
        self.max_updates = config.max_updates # No. of episodes of each env for training
        self.reset_env_timestep = config.reset_env_timestep
        self.validate_timestep = config.validate_timestep
        self.num_envs = config.num_envs # Batch size for training environments

        if not os.path.exists(f'./trained_network/{self.data_source}'):
            os.makedirs(f'./trained_network/{self.data_source}')
        if not os.path.exists(f'./train_log/{self.data_source}'):
            os.makedirs(f'./train_log/{self.data_source}')

        if device.type == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if self.data_source == 'SD1':
            self.data_name = f'{self.n_j}x{self.n_m}'
        elif self.data_source == 'SD2':
            self.data_name = f'{self.n_j}x{self.n_m}{strToSuffix(config.data_suffix)}'

        self.vali_data_path = f'./data/data_train_vali/{self.data_source}/{self.data_name}'
        self.test_data_path = f'./data/{self.data_source}/{self.data_name}'
        self.model_name = f'{self.data_name}{strToSuffix(config.model_suffix)}'

        # seed
        self.seed_train = config.seed_train
        self.seed_test = config.seed_test
        setup_seed(self.seed_train)

        self.env = FJSPEnvForSameOpNums(self.n_j, self.n_m)
        self.test_data = load_data_from_files(self.test_data_path)
        # validation data set
        vali_data = load_data_from_files(self.vali_data_path)

        if self.data_source == 'SD1':
            self.vali_env = FJSPEnvForVariousOpNums(self.n_j, self.n_m)
        elif self.data_source == 'SD2':
            self.vali_env = FJSPEnvForSameOpNums(self.n_j, self.n_m)

        self.vali_env.set_initial_data(vali_data[0], vali_data[1])

        self.ppo = PPO_initialize( actor, critic)
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)

    def train(self):
        """
            train the model following the config
        """
        setup_seed(self.seed_train)
        self.log = []
        self.validation_log = []
        self.record = float('inf')

        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'): # 1000
            ep_st = time.time()

            # resampling the training data
            if i_update % self.reset_env_timestep == 0:
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

            # save the mean rewards of all instances in current training data
            self.log.append([i_update, mean_rewards_all_env])



    def save_training_log(self):
        """
            save reward data & validation makespan data (during training) and the entire training time
        """
        file_writing_obj = open(f'./train_log/{self.data_source}/' + 'reward_' + self.model_name + '.txt', 'a')
        file_writing_obj.write(str(self.log))

        file_writing_obj1 = open(f'./train_log/{self.data_source}/' + 'valiquality_' + self.model_name + '.txt', 'a')
        file_writing_obj1.write(str(self.validation_log))

        file_writing_obj3 = open(f'./train_time.txt', 'a')
        file_writing_obj3.write(
            f'model path: ./DANIEL_FJSP/trained_network/{self.data_source}/{self.model_name}\t\ttraining time: '
            f'{round((self.train_et - self.train_st), 2)}\t\t local time: {str_time}\n')

    def save_validation_log(self):
        """
            save the results of validation
        """
        file_writing_obj1 = open(f'./train_log/{self.data_source}/' + 'valiquality_' + self.model_name + '.txt', 'w')
        file_writing_obj1.write(str(self.validation_log))

    def sample_training_instances(self):
        """
            sample training instances following the config, 
            the sampling process of SD1 data is imported from "songwenas12/fjsp-drl" 
        :return: new training instances
        """
        prepare_JobLength = [random.randint(self.op_per_job_min, self.op_per_job_max) for _ in range(self.n_j)]
        # [6, 5, 5, 4, 5, 5, 6, 5, 6, 6]
        dataset_JobLength = []  # array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
        dataset_OpPT = []
        for i in range(self.num_envs): # 20
            if self.data_source == 'SD1':
                case = CaseGenerator(self.n_j, self.n_m, self.op_per_job_min, self.op_per_job_max,
                                     nums_ope=prepare_JobLength, path='./test', flag_doc=False)
                JobLength, OpPT, _ = case.get_case(i)
              
            else:
                JobLength, OpPT, _ = SD2_instance_generator(config=self.config)  
                # JobLength: array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
                # _ = 2.66
                '''OpPt: array([[22, 93,  0, 58,  3],
                                    .. (50, 5) ..
                                [56,  0,  0,  0,  0]])'''
            dataset_JobLength.append(JobLength)
            # [array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5]), array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5]), array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])]
            dataset_OpPT.append(OpPT)
            # array list [(50, 5), (50, 5), ...]
        return dataset_JobLength, dataset_OpPT

    def validate_envs_with_same_op_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have the same number of operations
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        state = self.vali_env.reset()

        while True:

            with torch.no_grad():
                pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
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

        self.ppo.policy.train()
        return self.vali_env.current_makespan

    def validate_envs_with_various_op_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have various number of operations
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        state = self.vali_env.reset()

        while True:

            with torch.no_grad():
                batch_idx = ~torch.from_numpy(self.vali_env.done_flag)
                pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor[batch_idx],  # [sz_b, N, 8]
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

        self.ppo.policy.train()
        return self.vali_env.current_makespan

    def save_model(self):
        """
            save the model
        """
        torch.save(self.ppo.policy.state_dict(), f'./trained_network/{self.data_source}'
                                                 f'/{self.model_name}.pth')

    def load_model(self):
        """
            load the trained model
        """
        model_path = f'./trained_network/{self.data_source}/{self.model_name}.pth'
        self.ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))


def main():







    actor_parameters = trainer.ppo.policy.actor.parameters()
    critic_parameters = trainer.ppo.policy.critic.parameters()

    # 初始化元参数 theta

    actor = Actor(configs.num_mlp_layers_actor, 4 * configs.layer_fea_output_dim[-1] + 8,
                           configs.hidden_dim_actor, 1).to(device)
    critic = Critic(configs.num_mlp_layers_critic, 2 * configs.layer_fea_output_dim[-1], configs.hidden_dim_critic,
                         1).to(device)
    
    # MAML 元训练循环
    for meta_iteration in range(configs.meta_iterations):
        meta_grads = []
        
        # 对每个任务进行训练
        for task in tasks:

            trainer = Trainer(configs, actor, critic, task)

            trainer.train()
            # 从元参数复制新参数
            phi = copy(theta)
            
            # 在当前任务上使用 PPO 更新几步
            for step in range(ppo_steps):
                data = collect_data(task, phi)
                grads = compute_ppo_gradients(data, phi)
                phi = update_parameters(phi, grads)
            
            # 计算元梯度
            meta_data = collect_data(task, phi)
            meta_grad = compute_meta_gradient(meta_data, phi)
            meta_grads.append(meta_grad)
        
        # 使用元梯度更新元参数
        theta = update_parameters(theta, meta_grads)



if __name__ == '__main__':
    main()




    def train(self):
        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"vali data :{self.vali_data_path}")
        print("\n")

        self.train_st = time.time()

        for iteration in tqdm(range(self.meta_iterations), file=sys.stdout, desc="progress", colour="blue"):
            
            ep_st = time.time()
            meta_grads_actor = []  # 用于存储所有任务的Actor梯度
            meta_grads_critic = []  # 用于存储所有任务的Critic梯度


            for task in range(self.num_tasks):

                if iteration % self.reset_env_timestep == 0:
                    dataset_job_length, dataset_op_pt = self.sample_training_instances()
                    state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)
                else:
                    state = self.env.reset()

                self.memory.generate()
                
                actor = Actor(configs.num_mlp_layers_actor, 4 * configs.layer_fea_output_dim[-1] + 8,
                        configs.hidden_dim_actor, 1).to(device)
                
                critic = Critic(configs.num_mlp_layers_critic, 2 * configs.layer_fea_output_dim[-1], configs.hidden_dim_critic,
                                    1).to(device)
                inner_ppo = PPO_initialize(actor, critic)
                inner_ppo.policy.load_state_dict(self.ppo.state_dict())

                p_loss, v_loss, *_ = inner_ppo.update(self.memory)

                actor_grad = inner_ppo.policy.actor.named_parameters()
                critic_grad = inner_ppo.policy.critic.named_parameters()
                meta_grads_actor.append(actor_grad)
                meta_grads_critic.append(critic_grad)

                self.memory.clear_memory()

    def train(self):
        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"vali data :{self.vali_data_path}")
        print("\n")

        self.train_st = time.time()

        for iteration in tqdm(range(self.meta_iterations), file=sys.stdout, desc="progress", colour="blue"):
            
            ep_st = time.time()
            meta_grads_actor = OrderedDict()  # 用于存储所有任务的Actor梯度
            meta_grads_critic = OrderedDict()  # 用于存储所有任务的Critic梯度
            

            for task in range(self.num_tasks):

                if iteration % self.reset_env_timestep == 0:
                    dataset_job_length, dataset_op_pt = self.sample_training_instances()
                    state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)
                else:
                    state = self.env.reset()
                ep_rewards = - deepcopy(self.env.init_quality)



                actor = Actor(configs.num_mlp_layers_actor, 4 * configs.layer_fea_output_dim[-1] + 8,
                        configs.hidden_dim_actor, 1).to(device)
                
                critic = Critic(configs.num_mlp_layers_critic, 2 * configs.layer_fea_output_dim[-1], configs.hidden_dim_critic,
                                    1).to(device)
                inner_ppo = PPO_initialize(actor, critic)
                inner_ppo.policy.load_state_dict(self.ppo.policy.state_dict())
                
                self.memory.generate()
                # p_loss, v_loss, *_ = inner_ppo.update(self.memory)
                loss, v_loss = inner_ppo.update(self.memory)
                mean_rewards_all_env = np.mean(ep_rewards)
                mean_makespan_all_env = np.mean(self.env.current_makespan)

                            # 记录任务内（inner-loop）的梯度
                for name, param in inner_ppo.policy.actor.named_parameters():
                    if param.requires_grad:
                        if name not in meta_grads_actor:
                            meta_grads_actor[name] = []
                        meta_grads_actor[name].append(param.grad.clone())
                        
                for name, param in inner_ppo.policy.critic.named_parameters():
                    if param.requires_grad:
                        if name not in meta_grads_critic:
                            meta_grads_critic[name] = []
                        meta_grads_critic[name].append(param.grad.clone())

                self.memory.clear_memory()

            # 对于Actor进行元更新
            for name, param in self.ppo.policy.actor.named_parameters():
                if param.requires_grad:
                    mean_grad = torch.stack(meta_grads_actor[name]).mean(dim=0)
                    param.grad = mean_grad
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()

            # 对于Critic进行元更新
            for name, param in self.ppo.policy.critic.named_parameters():
                if param.requires_grad:
                    mean_grad = torch.stack(meta_grads_critic[name]).mean(dim=0)
                    param.grad = mean_grad
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()
            if iteration % 100 == 0:
                for name, param in self.ppo.policy.critic.named_parameters():
                    tqdm.write(param.__str__())
            ep_et = time.time()
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    iteration + 1, mean_rewards_all_env, mean_makespan_all_env, loss, ep_et - ep_st))


    def train(self):
        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"vali data :{self.vali_data_path}")
        print("\n")

        self.train_st = time.time()

        for iteration in tqdm(range(self.meta_iterations), file=sys.stdout, desc="progress", colour="blue"):

            ep_st = time.time()
            meta_grads_actor = OrderedDict()  # 用于存储所有任务的Actor梯度
            meta_grads_critic = OrderedDict()  # 用于存储所有任务的Critic梯度

            inner_ppos = []

            for task in range(self.num_tasks):

                if iteration % self.reset_env_timestep == 0:
                    dataset_job_length, dataset_op_pt = self.sample_training_instances()
                    state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)
                else:
                    state = self.env.reset()
                

                inner_ppo = deepcopy(self.ppo.policy)

                ep_rewards = self.memory_generate(state, inner_ppo)
                loss, v_loss = inner_ppo.update(self.memory)
                mean_rewards_all_env = np.mean(ep_rewards)
                mean_makespan_all_env = np.mean(self.env.current_makespan)
                inner_ppos.append(inner_ppo)

            loss_list = []
            for i in range(self.num_tasks):
                dataset_job_length, dataset_op_pt = self.sample_training_instances()
                state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)

                self.memory_generate(state, inner_ppos[i])

                loss, _ = inner_ppos[i].compute_loss(self.memory)
                loss_list.append(loss)

            self.optimizer.zero_grad()  
            loss_list.mean().backward()
            self.optimizer.step()

            ep_et = time.time()
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    iteration + 1, mean_rewards_all_env, mean_makespan_all_env, loss, ep_et - ep_st))