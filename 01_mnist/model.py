import torch
from torch import nn


# 多层感知机（全连接网络）
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # define layers
        self.input_layer = nn.Linear(28 * 28, 256)
        self.hidden_layer1 = nn.Linear(256, 128)
        self.hidden_layer2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer1(x))
        x = torch.relu(self.hidden_layer2(x))
        x = self.output_layer(x)

        return x
