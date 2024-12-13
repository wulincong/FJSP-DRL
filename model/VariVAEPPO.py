from model.PPO import *
class VariVAEPPO:
    def __init__(self, config):
        """
        The implementation of PPO with VAE integration for meta-learning in VariVAE model.
        :param config: a package of parameters
        """
        self.lr = config.lr  # 学习率
        self.meta_lr = config.meta_lr
        self.gamma = config.gamma  # 折扣因子
        self.gae_lambda = config.gae_lambda  # GAE广义优势估计参数
        self.eps_clip = config.eps_clip  # PPO算法中的剪切范围
        self.k_epochs = config.k_epochs  # PPO算法中的迭代次数
        self.tau = config.tau  # 软更新时的更新权重
        # self.ppo_steps = config.ppo_steps

        # 损失系数
        self.ploss_coef = config.ploss_coef
        self.vloss_coef = config.vloss_coef
        self.entloss_coef = config.entloss_coef
        self.vae_loss_coef = config.vae_loss_coef  # VAE 损失系数
        self.minibatch_size = config.minibatch_size

        # 初始化 VariVAE 模型
        self.policy = VariVAE(config)
        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 创建优化器
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.vae_optimizer = torch.optim.Adam(self.policy.vae.parameters(), lr=self.lr)  # VAE 优化器
        self.meta_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.meta_lr)

        self.V_loss_2 = nn.MSELoss()
        self.device = torch.device(config.device)

    def vae_loss(self, fea_global, reconstructed_fea, mu, logvar):
        """
        计算 VAE 的重构损失和 KL 散度损失
        """
        # 如果fea_global是单一维度 (256)，我们可以通过增加维度来确保它匹配reconstructed_fea
        if fea_global.dim() == 1:
            fea_global = fea_global.unsqueeze(1).repeat(1, reconstructed_fea.size(1))  # 将fea_global扩展为[256, 8]

        recon_loss = F.mse_loss(reconstructed_fea, fea_global, reduction='mean')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / fea_global.size(0)
        return recon_loss + self.vae_loss_coef * kl_div

    def update(self, memory):
        '''
        更新 VariVAE 模型
        :param memory: PPO的经验数据
        :return: 总损失和价值损失
        '''
        # 获取转置后的训练数据
        t_data = memory.transpose_data()
        t_advantage_seq, v_target_seq = memory.get_gae_advantages()

        full_batch_size = len(t_data[-1])
        num_batch = np.ceil(full_batch_size / self.minibatch_size)

        loss_epochs = 0
        v_loss_epochs = 0
        vae_loss_epochs = 0

        for _ in range(self.k_epochs):
            for i in range(int(num_batch)):
                start_idx = i * self.minibatch_size
                end_idx = (i + 1) * self.minibatch_size if i + 1 < num_batch else full_batch_size

                # 前向传播
                pis, vals, mu_j, logvar_j, mu_m, logvar_m, reconstructed_fea_j, reconstructed_fea_m = self.policy(
                    fea_j=t_data[0][start_idx:end_idx],
                    op_mask=t_data[1][start_idx:end_idx],
                    candidate=t_data[6][start_idx:end_idx],
                    fea_m=t_data[2][start_idx:end_idx],
                    mch_mask=t_data[3][start_idx:end_idx],
                    comp_idx=t_data[5][start_idx:end_idx],
                    dynamic_pair_mask=t_data[4][start_idx:end_idx],
                    fea_pairs=t_data[7][start_idx:end_idx]
                )

                action_batch = t_data[8][start_idx: end_idx]
                logprobs, ent_loss = eval_actions(pis, action_batch)
                ratios = torch.exp(logprobs - t_data[12][start_idx: end_idx].detach())

                # 计算 PPO 损失
                advantages = t_advantage_seq[start_idx: end_idx]
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                v_loss = self.V_loss_2(vals.squeeze(1), v_target_seq[start_idx: end_idx])
                p_loss = -torch.min(surr1, surr2)
                ent_loss = -ent_loss.clone()
                ppo_loss = self.vloss_coef * v_loss + self.ploss_coef * p_loss + self.entloss_coef * ent_loss

                # 计算 VAE 损失
                vae_loss_j = self.vae_loss(t_data[9][start_idx:end_idx], reconstructed_fea_j, mu_j, logvar_j)
                vae_loss_m = self.vae_loss(t_data[10][start_idx:end_idx], reconstructed_fea_m, mu_m, logvar_m)
                total_vae_loss = vae_loss_j + vae_loss_m

                # 计算总损失
                loss = ppo_loss + self.vae_loss_coef * total_vae_loss

                # 反向传播与优化
                self.optimizer.zero_grad()
                # self.vae_optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                # self.vae_optimizer.step()

                # 记录损失
                loss_epochs += loss.mean().item()
                v_loss_epochs += v_loss.mean().item()
                vae_loss_epochs += total_vae_loss.mean().item()

        # 软更新策略
        for policy_old_params, policy_params in zip(self.policy_old.parameters(), self.policy.parameters()):
            policy_old_params.data.copy_(self.tau * policy_old_params.data + (1 - self.tau) * policy_params.data)

        return loss_epochs / self.k_epochs, v_loss_epochs / self.k_epochs, vae_loss_epochs / self.k_epochs

    def inner_update(self, memory, k_epochs, task_lr, params=None):
        # Clone the current policy parameters to start task-specific updates
        t_data = memory.transpose_data()  # Support set data
        t_advantage_seq, v_target_seq = memory.get_gae_advantages()
        full_batch_size = len(t_data[-1])
        num_batch = np.ceil(full_batch_size / self.minibatch_size)

        if params is None:
            updated_params = {name: param.clone() for name, param in self.policy.named_parameters()}
        else:
            updated_params = {name: param.clone() for name, param in params.items()}
        
        # Ensure updated_params are float32
        for name, param in updated_params.items():
            updated_params[name] = param.float()

        for i in range(int(num_batch)):
            start_idx = i * self.minibatch_size
            end_idx = min((i + 1) * self.minibatch_size, full_batch_size)
            
            pis, vals, *others = self.policy(fea_j=t_data[0][start_idx:end_idx],
                                            op_mask=t_data[1][start_idx:end_idx],
                                            candidate=t_data[6][start_idx:end_idx],
                                            fea_m=t_data[2][start_idx:end_idx],
                                            mch_mask=t_data[3][start_idx:end_idx],
                                            comp_idx=t_data[5][start_idx:end_idx],
                                            dynamic_pair_mask=t_data[4][start_idx:end_idx],
                                            fea_pairs=t_data[7][start_idx:end_idx],
                                            params=updated_params)
            
            # Ensure the advantage and value tensors are also float32
            advantages = t_advantage_seq[start_idx:end_idx].float()
            v_target_seq_batch = v_target_seq[start_idx:end_idx].float()

            action_batch = t_data[8][start_idx:end_idx]
            logprobs, _ = eval_actions(pis, action_batch)
            ratios = torch.exp(logprobs - t_data[12][start_idx:end_idx].detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            v_loss = self.V_loss_2(vals.squeeze(1), v_target_seq_batch)
            p_loss = -torch.min(surr1, surr2).mean()

            # Total loss for inner loop update
            inner_loss = self.vloss_coef * v_loss + self.ploss_coef * p_loss

            # Ensure the loss is float32
            inner_loss = inner_loss.float()

            # Compute gradients for current parameters
            grads = torch.autograd.grad(inner_loss.mean(), updated_params.values(), allow_unused=True)
            
            # Update parameters using task_lr
            params = {name: param - task_lr * grad for (name, param), grad in zip(updated_params.items(), grads) if grad is not None}

        return params
    
    def meta_optimize(self, valid_memorys, iteration_policies):
        """
        Perform meta-gradient update for VariVAE using the query set.
        :param valid_memorys: Query set memory for each task.
        :param iteration_policies: Task-specific parameters (theta') from inner loop.
        :return: Meta-loss for logging.
        """
        meta_loss = 0.0
        self.meta_optimizer.zero_grad()

        for task_idx, memory in enumerate(valid_memorys):
            # Get task-specific policy parameters (theta')
            params = iteration_policies[task_idx]

            # Transpose query set memory
            t_data = memory.transpose_data()
            t_advantage_seq, v_target_seq = memory.get_gae_advantages()

            full_batch_size = len(t_data[-1])
            num_batch = np.ceil(full_batch_size / self.minibatch_size)

            task_loss = 0.0  # Accumulate task loss over mini-batches

            # Ensure the data is float32
            t_data = [x.float() for x in t_data]
            t_advantage_seq = t_advantage_seq.float()
            v_target_seq = v_target_seq.float()

            for i in range(int(num_batch)):
                start_idx = i * self.minibatch_size
                end_idx = min((i + 1) * self.minibatch_size, full_batch_size)

                # Forward pass with task-specific parameters
                pis, vals, mu_j, logvar_j, mu_m, logvar_m, reconstructed_fea_j, reconstructed_fea_m = self.policy(
                    fea_j=t_data[0][start_idx:end_idx],
                    op_mask=t_data[1][start_idx:end_idx],
                    candidate=t_data[6][start_idx:end_idx],
                    fea_m=t_data[2][start_idx:end_idx],
                    mch_mask=t_data[3][start_idx:end_idx],
                    comp_idx=t_data[5][start_idx:end_idx],
                    dynamic_pair_mask=t_data[4][start_idx:end_idx],
                    fea_pairs=t_data[7][start_idx:end_idx],
                    params=params
                )

                # Compute PPO losses
                action_batch = t_data[8][start_idx:end_idx]
                logprobs, _ = eval_actions(pis, action_batch)
                ratios = torch.exp(logprobs - t_data[12][start_idx:end_idx].detach())
                advantages = t_advantage_seq[start_idx:end_idx]
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                v_loss = self.V_loss_2(vals.squeeze(1), v_target_seq[start_idx:end_idx])
                p_loss = -torch.min(surr1, surr2).mean()

                # Compute VAE losses
                vae_loss_j = self.vae_loss(t_data[9][start_idx:end_idx], reconstructed_fea_j, mu_j, logvar_j)
                vae_loss_m = self.vae_loss(t_data[10][start_idx:end_idx], reconstructed_fea_m, mu_m, logvar_m)

                # Total loss for the task (accumulated for mini-batches)
                task_loss += (
                    self.vloss_coef * v_loss +
                    self.ploss_coef * p_loss +
                    self.vae_loss_coef * (vae_loss_j + vae_loss_m)
                )

            # Average the task loss over the mini-batches
            task_loss /= num_batch

            meta_loss += task_loss

        # Average the meta-loss across tasks
        meta_loss /= len(valid_memorys)

        # Perform meta-gradient update
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()
