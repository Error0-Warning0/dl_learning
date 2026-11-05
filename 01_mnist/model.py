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


# CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def build_model(name):
    if name == "mlp":
        return MLP()
    if name == "cnn":
        return CNN()
    raise ValueError(f"未知的模型名：{name}")


# Check the model output shape
if __name__ == "__main__":
    model = CNN()
    x = torch.randn(4, 1, 28, 28)
    print(model(x).shape)
    print(sum(p.numel() for p in model.parameters()))
