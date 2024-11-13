from .base import *

def main(config, flag_sample):
    test_data, test_model = load_data_and_model(config)

    if flag_sample:
        model_prefix = "DANIELS"
    else:
        model_prefix = "DANIELG"
    
    for data in test_data:
        print("-" * 25 + "Test Learned Model" + "-" * 25)
        print(f"test data name: {data[1]}")
        print(f"test mode: {model_prefix}")
        save_direc = f'./test_results/{config.data_source}/{data[1]}'
        if not os.path.exists(save_direc):
            os.makedirs(save_direc)

        for model in test_model:
            save_path = save_direc + f'/Result_{model_prefix}+{model[1]}_{data[1]}.npy'
            
            if (not os.path.exists(save_path)) or config.cover_flag:
                print(f"Model name : {model[1]}")
                print(f"data name: ./data/{config.data_source}/{data[1]}")

                if not flag_sample:
                    print("Test mode: Greedy")
                    result_5_times = []
                    # Greedy mode, test 5 times, record average time.
                    for j in range(5):
                        t = Test(config, data[0], model[0], config.seed_test)
                        result = t.fast_adapt_greedy_strategy()
                        result_5_times.append(result)
                    result_5_times = np.array(result_5_times)
                    save_result = np.mean(result_5_times, axis=0)
                    print("testing results:")
                    print(f"makespan(greedy): ", save_result[:, 0].mean())
                    print(f"time: ", save_result[:, 1].mean())
                
                np.save(save_path, save_result)