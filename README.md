# FJSP-DRL

This repository is the official implementation of the paper “[Flexible Job Shop Scheduling via Dual Attention Network Based Reinforcement Learning](https://doi.org/10.1109/TNNLS.2023.3306421)”. *IEEE Transactions on Neural Networks and Learning Systems*, 2023.

## Quick Start

### requirements

- python $=$ 3.7.11
- argparse $=$ 1.4.0
- numpy $=$ 1.21.6
- ortools $=$ 9.3.10497
- pandas $=$ 1.3.5
- torch $=$ 1.11.0+cu113
- torchaudio $=$ 0.11.0+cu113
- torchvision $=$ 0.12.0+cu113
- tqdm $=$ 4.64.0

### introduction

- `data` saves the instance files including testing instances (in the subfolder `BenchData`, `SD1` and `SD2`) and validation instances (in the subfolder `data_train_vali`) .
- `model` contains the implementation of the proposed framework.
- `or_solution` saves the results solved by Google OR-Tools.
- `test_results`saves the results solved by priority dispatching rules and DRL models.
- `train_log` saves the training log of models, including information of the reward and validation makespan.
- `trained_network` saves the trained models.
- `common_utils.py` contains some useful functions (including the implementation of priority dispatching rules mentioned in the paper) .
- `data_utils.py` is used for data generation, reading and format conversion.
- `fjsp_env_same_op_nums.py` and `fjsp_env_various_op_nums.py` are implementations of fjsp environments, describing fjsp instances with the same number of operations and different number of operations, respectively.
- `ortools_solver.py` is used for solving the instances by Google OR-Tools.
- `params.py` defines parameters settings.
- `print_test_result.py` is used for printing the experimental results into an Excel file.
- `test_heuristic.py` is used for solving the instances by priority dispatching rules.
- `test_trained_model.py` is used for evaluating the models.
- `train.py` is used for training.

### train

```python
python train.py # train the model on 10x5 FJSP instances using SD2

# options (Validation instances of corresponding size should be prepared in ./data/data_train_vali/{data_source})
python train.py 	--n_j 10		# number of jobs for training/validation instances
			--n_m 5			# number of machines for training/validation instances
    			--data_source SD2	# data source (SD1 / SD2)
        		--data_suffix mix	# mode for SD2 data generation
            					# 'mix' is thedefault mode as defined in the paper
                				# 'nf' means 'no flexibility' (generating JSP data) 
        		--model_suffix demo	# annotations for the model
```

### evaluate

```python
python test_trained_model.py # evaluate the model trained on '10x5+mix' of SD2 using the testing instances of the same size using the greedy strategy

# options (Model files should be prepared in ./trained_network/{model_source})
python test_trained_model.py 	--data_source SD2	# source of testing instances
				--model_source SD2	# source of instances that the model trained on 
    				--test_data 10x5+mix	# list of instance names for testing
        			--test_model 10x5+mix	# list of model names for testing
            			--test_mode False	# whether using the sampling strategy
                		--sample_times 100	# set the number of sampling times
```

## Cite the paper

```
@ARTICLE{10246328,
  author={Wang, Runqing and Wang, Gang and Sun, Jian and Deng, Fang and Chen, Jie},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Flexible Job Shop Scheduling via Dual Attention Network-Based Reinforcement Learning}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TNNLS.2023.3306421}
}
```

## Reference

- https://github.com/songwenas12/fjsp-drl/
- https://github.com/zcaicaros/L2D
- https://github.com/google/or-tools
- https://github.com/Diego999/pyGAT


## Memory数据结构
- 专门设计用于PPO的轨迹数据存储与管理。
#### 主要数据结构

|属性名|数据类型/维度|作用说明|
|---------|--------|--------|
|fea_j_seq	| [N, tensor[sz_b, N, 8]] |作业特征序列|
|op_mask_seq|	[N, tensor[sz_b, N, 3]]	|操作掩码，用于约束合法动作|
|fea_m_seq|	[N, tensor[sz_b, M, 6]]	|机器特征序列|
|mch_mask_seq|	[N, tensor[sz_b, M, M]]	|机器掩码，表示机器约束|
|dynamic_pair_mask_seq	|[N, tensor[sz_b, J, M]]	|动态机器-作业配对掩码|
|comp_idx_seq|	[N, tensor[sz_b, M, M, J]]|	竞争索引特征|
|candidate_seq|	[N, tensor[sz_b, J]]|	候选作业序列|
|fea_pairs_seq|	[N, tensor[sz_b, J]]|	特征配对信息|
|action_seq|	[N, tensor[sz_b]]|	动作索引|
|reward_seq|	[N, tensor[sz_b]]|	奖励值|
|val_seq|	[N, tensor[sz_b]]|	状态值|
|done_seq|	[N, tensor[sz_b]]|	结束标记，用于PPO更新时终止条件|
|log_probs|	[N, tensor[sz_b]]|	动作概率的对数值|


#### 优势估计：get_gae_advantages

广义优势估计通过时间差分误差递推得到，公式如下：

$$
A_t^{\text{GAE}(\lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}
$$
- 其中$\delta_t$是时间差分误差 
$$ \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) $$
- $A_t^{\text{GAE}(\lambda)}$ 在 𝜆 参数控制下的优势估计。
- γ 是折扣因子，控制未来奖励的衰减程度。
- 𝜆是平滑参数，平衡偏差和方差。
- 𝑉(𝑠𝑡)是状态 $𝑠_𝑡$的值函数估计。

为了更高效地计算，可以将 GAE 写成递推形式：

$$
A_t^{\text{GAE}(\lambda)} = \delta_t + \gamma \lambda A_{t+1}^{\text{GAE}(\lambda)}
$$
这表示优势估计可以通过未来的 GAE 值与当前时间差分误差逐步递推计算得到。

GAE提供了平滑的优势估计，适应于长时间步的策略优化。

## PPO代码

#### PPO的核心目标：
稳定地更新策略，限制重要性采样比率，防止策略过大变动。
引入值函数和熵损失，平衡探索与利用。

#### Clipped Objective：
通过 torch.clamp 限制策略更新幅度，防止策略“崩坏”。

#### GAE（广义优势估计）：
提高优势估计的稳定性，减少高方差的影响。

#### 软更新策略：
使用权重 tau 缓慢更新旧策略网络，增强学习稳定性。

## 训练过程
DANTrainer 类负责将 PPO 算法与 FJSP 环境结合，进行模型训练。

#### 训练流程
1. 初始化 PPO 算法和 FJSP 环境。
2. 环境交互：通过策略网络采样动作与环境交互，收集轨迹数据。
3. 策略更新：调用 PPO 更新网络参数，优化策略与值函数。
4. 结果记录：监控奖励、工期和损失，保存模型参数。


