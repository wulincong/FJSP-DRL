import time
import os
from common_utils import *
from params import configs
from tqdm import tqdm
from data_utils import pack_data_from_config
from model.PPO import PPO_initialize
from common_utils import setup_seed
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

device = torch.device(configs.device)


test_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

class TestBase:

    def __init__(self) -> None:
        self.ppo=PPO_initialize()

    def greedy_strategy(self, data_set, model_path, seed):
        """
            使用贪婪策略在给定的数据上测试模型。
        :param data_set: 测试数据
        :param model_path: 模型文件的路径
        :param seed: 用于测试的种子值

        :return: 测试结果，包括完成时间和时间信息
        """
        test_result_list = []
        setup_seed(seed)
        self.ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))
        self.ppo.policy.eval()

        n_j = data_set[0][0].shape[0]
        n_op, n_m = data_set[1][0].shape
        env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)

        for i in range(len(data_set[0])):

            state = env.set_initial_data([data_set[0][i]], [data_set[1][i]])
            t1 = time.time()
            while True:

                with torch.no_grad():
                    pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor,  # [1, N, 8]
                                    op_mask=state.op_mask_tensor,  # [1, N, N]
                                    candidate=state.candidate_tensor,  # [1, J]
                                    fea_m=state.fea_m_tensor,  # [1, M, 6]
                                    mch_mask=state.mch_mask_tensor,  # [1, M, M]
                                    comp_idx=state.comp_idx_tensor,  # [1, M, M, J]
                                    dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                                    fea_pairs=state.fea_pairs_tensor)  # [1, J, M]

                action = greedy_select_action(pi)
                state, reward, done = env.step(actions=action.cpu().numpy())
                if done:
                    break
            t2 = time.time()

            test_result_list.append([env.current_makespan[0], t2 - t1])

        return np.array(test_result_list)


    def sampling_strategy(self, data_set, model_path, sample_times, seed):
        """
            使用抽样策略在给定的数据上测试模型。
        :param data_set: 测试数据
        :param model_path: 模型文件的路径
        :param seed: 用于测试的种子值
        :return: 测试结果，包括完成时间和时间信息
        """
        setup_seed(seed)
        test_result_list = []
        self.ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))
        self.ppo.policy.eval()

        n_j = data_set[0][0].shape[0]
        n_op, n_m = data_set[1][0].shape
        from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
        env = FJSPEnvForSameOpNums(n_j, n_m)

        for i in range(len(data_set[0])):
            # copy the testing environment
            JobLength_dataset = np.tile(np.expand_dims(data_set[0][i], axis=0), (sample_times, 1))
            OpPT_dataset = np.tile(np.expand_dims(data_set[1][i], axis=0), (sample_times, 1, 1))

            state = env.set_initial_data(JobLength_dataset, OpPT_dataset)
            t1 = time.time()
            while True:

                with torch.no_grad():
                    pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor,  # [100, N, 8]
                                    op_mask=state.op_mask_tensor,  # [100, N, N]
                                    candidate=state.candidate_tensor,  # [100, J]
                                    fea_m=state.fea_m_tensor,  # [100, M, 6]
                                    mch_mask=state.mch_mask_tensor,  # [100, M, M]
                                    comp_idx=state.comp_idx_tensor,  # [100, M, M, J]
                                    dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [100, J, M]
                                    fea_pairs=state.fea_pairs_tensor)  # [100, J, M]

                action_envs, _ = sample_action(pi)
                state, _, done = env.step(action_envs.cpu().numpy())
                if done.all():
                    break

            t2 = time.time()
            best_makespan = np.min(env.current_makespan)
            test_result_list.append([best_makespan, t2 - t1])
        
        return np.array(test_result_list)


    def multi_operations(self, data_set, model_path, seed, job_nums:list):
        setup_seed(seed)
        test_result_list = []
        self.ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))
        self.ppo.policy.eval()
        for n_j in job_nums:
            ...
















