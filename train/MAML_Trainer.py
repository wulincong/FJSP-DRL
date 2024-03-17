import time
import torch
import learn2learn as l2l
from common_utils import setup_seed

class MAMLTrainer():
    def __init__(self, config) -> None:
        self.config = config
        self.model_name =  f'maml{int(time.time())}'
        # seed
        self.seed_train = config.seed_train
        setup_seed(self.seed_train)
        print("save model name: ",self.model_name)


    def train():
        ...
    