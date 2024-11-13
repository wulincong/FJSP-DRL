from common_utils import *
from params import configs
from tqdm import tqdm
from data_utils import load_data_from_files, CaseGenerator, SD2_instance_generator, matrix_to_text
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
import higher
from model.sub_layers import *
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter


str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

device = torch.device(configs.device)

class Trainer:
    def __init__(self, config) -> None:
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
        self.meta_iterations = config.meta_iterations
        self.inner_updates = config.inner_updates
        self.meta_lr = config.meta_lr
        self.task_lr = config.task_lr
        self.num_tasks = config.num_tasks
        self.adapt_lr = config.adapt_lr
        self.adapt_nums = config.adapt_nums
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
        print("save model name: ",self.model_name)

        # seed
        self.seed_train = config.seed_train
        self.seed_test = config.seed_test
        setup_seed(self.seed_train)

        self.test_data = load_data_from_files(self.test_data_path)
        # validation data set
        vali_data = load_data_from_files(self.vali_data_path)

        if self.data_source == 'SD1':
            self.vali_env = FJSPEnvForVariousOpNums(self.n_j, self.n_m)
        elif self.data_source == 'SD2':
            self.vali_env = FJSPEnvForSameOpNums(self.n_j, self.n_m)

        self.vali_env.set_initial_data(vali_data[0], vali_data[1])

        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)
        self.writer = SummaryWriter(configs.logdir)  # 创建一个SummaryWriter对象，用于记录日志

    def train(self):
        '''定义训练过程'''
        raise NotImplementedError


    def iter_log(self, iteration, scalars:dict=None, writer=None):
        if writer is None:
            writer=self.writer
        for key in scalars.keys():
            writer.add_scalar(key, scalars[key], iteration)



    def valid_model(self,):
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

    def memory_generate(self, env, state, inner_ppo):
        '''根据环境生成轨迹'''
        ep_rewards = - deepcopy(env.init_quality)
        self.memory.clear_memory()
        while True: # 解决一个FJSP问题的过程
                    # state store
            self.memory.push(state)
            with torch.no_grad():
                pi_envs, vals_envs = inner_ppo.policy_old(fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
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
            state, reward, done = env.step(actions=action_envs.cpu().numpy())
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

        return ep_rewards

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
                JobLength, OpPT, _ = SD2_instance_generator(config=self.config,n_j = self.n_j, n_m = self.n_m)  
                lines_doc = matrix_to_text(JobLength, OpPT, _)
                print("\n".join(lines_doc))

                # JobLength: array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
                # _ = 2.66
                '''OpPt: array([[22, 93,  0, 58,  3],
                                    .. (50, 5) ..
                                [56,  0,  0,  0,  0]])'''
            dataset_JobLength.append(JobLength)
            # [array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5]), array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5]), array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])]
            dataset_OpPT.append(OpPT)
            # array list [(50, 5), (50, 5), ...]
        # print("len of sample_training_instances/dataset_OpPT:", len(dataset_OpPT))
        return dataset_JobLength, dataset_OpPT

    def validate_envs_with_same_op_nums(self, model=None, policy = None):
        """
            validate the policy using the greedy strategy
            where the validation instances have the same number of operations
        :return: the makespan of the validation set
        """
        if policy is None: policy = self.ppo.policy
        if model: policy = model.policy
        policy.eval()
        state = self.vali_env.reset()
        
        while True:

            with torch.no_grad():
                pi, _ = policy(fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
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

        policy.train()
        return self.vali_env.current_makespan

    def validate_envs_with_various_op_nums(self, model = None, policy = None):
        """
            validate the policy using the greedy strategy
            where the validation instances have various number of operations
        :return: the makespan of the validation set
        """
        if policy is None: policy = self.ppo.policy
        if model: policy = model.policy
        policy.eval()
        state = self.vali_env.reset()

        while True:

            with torch.no_grad():
                batch_idx = ~torch.from_numpy(self.vali_env.done_flag)
                pi, _ = policy(fea_j=state.fea_j_tensor[batch_idx],  # [sz_b, N, 8]
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

        policy.train()
        return self.vali_env.current_makespan

    def save_model(self, model=None):
        """
            save the model
        """
        if model is None: model = self.ppo
        torch.save(model.policy.state_dict(), f'./trained_network/{self.data_source}'
                                                 f'/{self.model_name}.pth')

    def load_model(self, model=None):
        """
            load the trained model
        """
        if model is None: model = self.ppo
        model_path = f'./trained_network/{self.data_source}/{self.model_name}.pth'
        model.policy.load_state_dict(torch.load(model_path, map_location='cuda'))
    
    def fast_adapt_valid_model(self, model=None):

        if model is None: model = self.ppo
        
        adapt_policy = model.clone_policy()

        # fast adapt
        for _ in range(self.adapt_nums):
            state = self.vali_env.reset()
            self.memory_generate(self.vali_env, state, model)
            _, _, adapt_policy = model.fast_adapt(self.memory, adapt_policy)



        if self.data_source == "SD1":
            vali_result = self.validate_envs_with_various_op_nums(policy=adapt_policy).mean()
        else:
            vali_result = self.validate_envs_with_same_op_nums(policy=adapt_policy).mean()

        if vali_result < self.record:
            self.save_model()
            self.record = vali_result

        
        
        self.validation_log.append(vali_result)
        self.save_validation_log()
        return vali_result
