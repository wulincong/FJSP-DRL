from train.base import *

class DANTrainer(Trainer):
    def __init__(self, config):

        super().__init__(config)
        self.ppo = PPO_initialize()
        self.env = FJSPEnvForSameOpNums(self.n_j, self.n_m)

