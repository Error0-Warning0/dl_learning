import torch
from torch import nn


# CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def build_model(name):
    if name == "cnn":
        return CNN()
    raise ValueError(f"未知的模型名：{name}")


# Check the model output shape
if __name__ == "__main__":
    model = CNN()
    x = torch.randn(4, 3, 32, 32)
    print(model(x).shape)
    print(sum(p.numel() for p in model.parameters()))
