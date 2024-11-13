import torch
import torch.nn as nn

# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out

# 输入数据维度
input_size = 40
hidden_size = 64

# 创建模型实例
model = RNN(input_size, hidden_size)
print(model)

# 示例输入数据
inputs = torch.randn(20, 50, 40)  # (batch_size, sequence_length, input_size)

# 前向传播
outputs = model(inputs)
print(outputs.shape)  # 输出结果尺寸为(20, 50, 1)
