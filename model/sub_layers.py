import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """
            the implementation of multi layer perceptrons (refer to L2D)
        :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                            If num_layers=1, this reduces to linear model.
        :param input_dim: dimensionality of input features
        :param hidden_dim: dimensionality of hidden units at ALL layers
        :param output_dim:  number of classes for prediction
        """

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class Actor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):  # 3, 40, 64, 1
        """
            the implementation of Actor network (refer to L2D)
        :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                            If num_layers=1, this reduces to linear model.
        :param input_dim: dimensionality of input features
        :param hidden_dim: dimensionality of hidden units at ALL layers
        :param output_dim:  number of classes for prediction
        """
        super(Actor, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        self.activative = torch.tanh

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        if num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
            init.normal_(self.linear.weight, mean=0, std=0.1)
            init.zeros_(self.linear.bias)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            # 初始化权重和偏置
            for linear in self.linears:
                init.normal_(linear.weight, mean=0, std=0.1)
                init.zeros_(linear.bias)

    def forward(self, x):  #(20, 50, 40)
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.activative((self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class Critic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):  # 3, 16, 64, 1
        """
            the implementation of Critic network (refer to L2D)
        :param num_layers: number of layers in the neural networks (EXCLUDING the input layer).
                            If num_layers=1, this reduces to linear model.
        :param input_dim: dimensionality of input features
        :param hidden_dim: dimensionality of hidden units at ALL layers
        :param output_dim:  number of classes for prediction
        """
        super(Critic, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        self.activative = torch.tanh

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        if num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
            init.normal_(self.linear.weight, mean=0, std=0.1)
            init.zeros_(self.linear.bias)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            # 初始化权重和偏置
            for linear in self.linears:
                init.normal_(linear.weight, mean=0, std=0.1)
                init.zeros_(linear.bias)
    
    def forward(self, x): # 20, 16
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.activative((self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class ActorRNN(nn.Module):

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(ActorRNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.ht = None

        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        # self.activative = torch.tanh

    def forward(self, x, h0): #(20, 50, 40)
        # 初始化隐藏状态
        # h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        
        # 前向传播
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out, _


class CriticRNN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(CriticRNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # self.ht = nn.Parameter(torch.zeros(1, self.hidden_dim)).to('cuda')

        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        # self.activative = torch.tanh

    def forward(self, x): # 20, 16
        # 初始化隐藏状态
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        
        # 前向传播
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out
    
