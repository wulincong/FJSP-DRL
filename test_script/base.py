import time
import os
from common_utils import *
from params import configs
from tqdm import tqdm
from data_utils import pack_data_from_config
from model.PPO import PPO_initialize
from common_utils import setup_seed
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums, FJSPEnvForSameOpNumsEnergy, FJSPEnvForSameOpNumsMakespanEnergy
from fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from copy import deepcopy
from model.PPO import Memory
from train.base import ConvergenceChecker

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

device = torch.device(configs.device)

test_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
convergence_checker = ConvergenceChecker(window_size=20, threshold=0.01)

class Test:

    def __init__(self, config, data_set, model_path) -> None:

        self.config = config
        self.data_set = data_set
        self.adapt_nums=config.adapt_nums
        self.model_path = model_path
        setup_seed(config.seed_test)
        self.ppo=PPO_initialize(config)
        self.ppo.policy.load_state_dict(torch.load(self.model_path, map_location='cuda',weights_only=True))
        n_j = data_set[0][0].shape[0]
        n_op, n_m = data_set[1][0].shape
        # if config.data_source.startswith("SD2EC0"):
        #     self.env = FJSPEnvForSameOpNumsMakespanEnergy(n_j, n_m, config.factor_Mk, config.factor_Ec)
        if config.data_source.startswith("SD2EC"):
            self.env = FJSPEnvForSameOpNumsEnergy(n_j, n_m, config.factor_Mk, config.factor_Ec)
        elif config.data_source == "SD2": self.env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
        else : self.env = FJSPEnvForVariousOpNums(n_j, n_m)

        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)
        self.finetuning_makespan = []
        self.finetuning_ec = []        #
        self.adapt_param_list = []  # self.ppo.policy.actor.parameters
        self.test_result_list = []  # (makespan, ec, time)
    
    def greedy_strategy(self, policy=None, finetuning = False):
        """
            使用贪婪策略在给定的数据上测试模型。
        :param data_set: 测试数据
        :param model_path: 模型文件的路径
        :param seed: 用于测试的种子值

        :return: 测试结果，包括完成时间和时间信息
        """
        
        if policy is None: policy=self.ppo.policy
        test_result = []
        policy.eval()
        for i in range(len(self.data_set[0])):
            
            if self.config.data_source.startswith("SD2EC"): 
                state = self.env.set_initial_data([self.data_set[0][i]], [self.data_set[1][i]], [self.data_set[2][i]], [self.data_set[3][i]])
            else: state = self.env.set_initial_data([self.data_set[0][i]], [self.data_set[1][i]], )
            ep_st = time.time()
            ep_rewards = -self.env.init_quality
            t1 = time.time()
            state = self.env.reset()
            while True:
                with torch.no_grad():
                    pi, _ = policy(fea_j=state.fea_j_tensor,  # [1, N, 8]
                                    op_mask=state.op_mask_tensor,  # [1, N, N]
                                    candidate=state.candidate_tensor,  # [1, J]
                                    fea_m=state.fea_m_tensor,  # [1, M, 6]
                                    mch_mask=state.mch_mask_tensor,  # [1, M, M]
                                    comp_idx=state.comp_idx_tensor,  # [1, M, M, J]
                                    dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                                    fea_pairs=state.fea_pairs_tensor)  # [1, J, M]

                action = greedy_select_action(pi)
                state, reward, done = self.env.step(actions=action.cpu().numpy())
                ep_rewards += reward
                if done:
                    break
            t2 = time.time()
            # tqdm.write(f"{cnt}, {ep_et - ep_st}")
            if self.config.data_source.startswith("SD2EC"):
                # print(self.env.current_EC)
                test_result.append([np.mean(self.env.current_EC), np.mean(self.env.total_real_energy_consumed), t2 - t1, np.mean(ep_rewards)])
            else: test_result.append([self.env.current_makespan[0], t2 - t1])
        test_result = np.array(test_result)
        self.test_result_list.append(test_result.mean(axis=0))
        return np.array(test_result)

    def finetuning(self, times = 5, job_length=None, OpPt=None):
        self.ppo.policy.load_state_dict(torch.load(self.model_path, map_location='cuda',weights_only=True))
        self.ppo.policy.train()
        self.finetuning_makespan = []
        self.finetuning_ec = []        #

        if self.config.data_source.startswith("SD2EC"): 
            state = self.env.set_initial_data(self.data_set[0], self.data_set[1], self.data_set[2], self.data_set[3])
        else: state = self.env.set_initial_data(self.data_set[0], self.data_set[1], )
        if job_length is not None: state = self.env.set_initial_data(job_length, OpPt)
        ep_reward_list = []
        
        for _ in range(times):
            state = self.env.reset()
            ep_reward = self.memory_generate(self.env, state, self.ppo.policy_old)
            ep_reward_list.append(ep_reward)
            loss, *_  = self.ppo.update(self.memory)
            self.adapt_param_list.append([p.clone().detach() for p in self.ppo.policy.actor.parameters()])
            self.finetuning_makespan.append(self.env.current_makespan[0])
            if self.config.data_source.startswith("SD2EC"): 
                self.finetuning_ec.append(self.env.current_EC[0])

        return {
            "reward":ep_reward_list,
            "makespan":self.finetuning_makespan
            ,"EC": self.finetuning_ec
        }

    def sampling_strategy(self, sample_times, policy=None):
        """
            使用抽样策略在给定的数据上测试模型。
        :param data_set: 测试数据
        :param model_path: 模型文件的路径
        :param seed: 用于测试的种子值
        :return: 测试结果，包括完成时间和时间信息
        """
        test_result_list = []
        if policy is None: policy = self.ppo.policy
        
        policy.eval()

        for i in range(len(self.data_set[0])):
            # copy the testing environment
            JobLength_dataset = np.tile(np.expand_dims(self.data_set[0][i], axis=0), (sample_times, 1))
            OpPT_dataset = np.tile(np.expand_dims(self.data_set[1][i], axis=0), (sample_times, 1, 1))

            state = self.env.set_initial_data(JobLength_dataset, OpPT_dataset)
            t1 = time.time()
            while True:

                with torch.no_grad():
                    pi, _ = policy(fea_j=state.fea_j_tensor,  # [100, N, 8]
                                    op_mask=state.op_mask_tensor,  # [100, N, N]
                                    candidate=state.candidate_tensor,  # [100, J]
                                    fea_m=state.fea_m_tensor,  # [100, M, 6]
                                    mch_mask=state.mch_mask_tensor,  # [100, M, M]
                                    comp_idx=state.comp_idx_tensor,  # [100, M, M, J]
                                    dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [100, J, M]
                                    fea_pairs=state.fea_pairs_tensor)  # [100, J, M]

                action_envs, _ = sample_action(pi)
                state, _, done = self.env.step(action_envs.cpu().numpy())
                if done.all():
                    break

            t2 = time.time()
            best_makespan = np.min(self.env.current_makespan)
            test_result_list.append([best_makespan, t2 - t1])
        
        return np.array(test_result_list)
    
    def memory_generate(self, env, state, policy):
        '''根据环境生成轨迹'''
        ep_rewards = - deepcopy(env.init_quality)
        self.memory.clear_memory()
        while True: # 解决一个FJSP问题的过程
                    # state store
            self.memory.push(state)
            with torch.no_grad():
                pi_envs, vals_envs, *others = policy(fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
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

    def fast_adapt(self):
        model = self.ppo
        adapt_policy = model.clone_policy()
        # fast adapt
        state = self.env.set_initial_data([self.data_set[0][0]], [self.data_set[1][0]])
        for _ in range(self.adapt_nums):
            state = self.env.reset()
            self.memory_generate(self.env, state, model)
            _, _, adapt_policy = model.fast_adapt(self.memory, adapt_policy)
        
        return adapt_policy
        


    def multi_operations(self, data_set, model_path, seed, job_nums:list):
        setup_seed(seed)
        test_result_list = []
        self.ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))
        self.ppo.policy.eval()
        for n_j in job_nums:
            ...

class FinetuningTest(Test):
    
    def greedy_strategy(self, policy=None, finetuning = False):
        """
            使用贪婪策略在给定的数据上测试模型。
        :param data_set: 测试数据
        :param model_path: 模型文件的路径
        :param seed: 用于测试的种子值

        :return: 测试结果，包括完成时间和时间信息
        """





def load_data_and_model(config):
    
    if not os.path.exists('./test_results'):
        os.makedirs('./test_results')

    # collect the path of test models
    test_model = []

    for model_name in config.test_model:
        test_model.append((f'./trained_network/{config.model_source}/{model_name}.pth', model_name))

    # collect the test data
    test_data = pack_data_from_config(config.data_source, config.test_data)

    return test_data, test_model













