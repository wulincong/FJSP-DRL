#测试maml模型的性能
from params import configs



#进行finetuning
from train.DAN_finetuning import DANTrainer

trainer = DANTrainer(configs)
trainer.train()



#对finetuning的模型进行测试


















