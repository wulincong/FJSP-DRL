import time
import os
from common_utils import *
from params import configs
from tqdm import tqdm
from data_utils import pack_data_from_config
from model.PPO import PPO_initialize
from common_utils import setup_seed
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from test_script.base import *

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

device = torch.device(configs.device)

ppo = PPO_initialize(configs)
test_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))



def main(config, flag_sample):
    """
        test the trained model following the config and save the results
    :param flag_sample: whether using the sampling strategy
    """
    assert config.data_source.startswith("SD2EC")
    setup_seed(config.seed_test)
    if not os.path.exists('./test_results'):
        os.makedirs('./test_results')

    # collect the path of test models
    test_model = []

    for model_name in config.test_model:
        test_model.append((f'./trained_network/{config.model_source}/{model_name}.pth', model_name))
    print(test_model)
    # collect the test data
    test_data = pack_data_from_config(config.data_source, config.test_data)

    for i in range(len(test_model)):
        model = test_model[i]
        data = test_data[i]
        print("-" * 25 + "Test Learned Model" + "-" * 25)
        print(f"test data name: {data[1]}")
        print(f"Model name : {model[1]}")
        finetuning = True if model[1].startswith("maml") else False

        result_5_times = []
        for j in range(5):
            test = Test(configs, data[0], model[0])
            result = test.greedy_strategy(finetuning=finetuning)
            result_5_times.append(result)
        result_5_times = np.array(result_5_times)
        save_result = np.min(result_5_times, axis=0)
        print("testing results:")
        print(f"EC(greedy): ", save_result[:, 0].mean())
        print(f"real EC(greedy): ", save_result[:, 1].mean())
        print(f"time:", save_result[:, 2].mean())
        print("="*100)


if __name__ == '__main__':
    main(configs, False)
    # main(configs, True)
