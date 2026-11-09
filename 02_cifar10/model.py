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


# CNN + BatchNorm（可选 Dropout）
class CNNBN(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# The basic building block of ResNet: two 3×3 convolutions + a shortcut connection
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_shortcut=True):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.use_shortcut = use_shortcut

        if use_shortcut and (stride != 1 or in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_shortcut:
            out = out + self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_blocks=3, use_shortcut=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self.make_layer(16, 16, num_blocks, 1, use_shortcut)
        self.layer2 = self.make_layer(16, 32, num_blocks, 2, use_shortcut)
        self.layer3 = self.make_layer(32, 64, num_blocks, 2, use_shortcut)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def make_layer(self, in_channels, out_channels, num_blocks, stride, use_shortcut):
        blocks = [BasicBlock(in_channels, out_channels, stride, use_shortcut)]
        for _ in range(num_blocks - 1):
            blocks.append(BasicBlock(out_channels, out_channels, 1, use_shortcut))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


def build_model(name):
    if name == "cnn":
        return CNN()
    if name == "cnn_bn":
        return CNNBN()
    if name == "cnn_bn_drop":
        return CNNBN(dropout=0.5)
    if name == "resnet20":
        return ResNet(num_blocks=3)
    if name == "plain20":
        return ResNet(num_blocks=3, use_shortcut=False)
    if name == "resnet56":
        return ResNet(num_blocks=9)
    if name == "plain56":
        return ResNet(num_blocks=9, use_shortcut=False)
    raise ValueError(f"未知的模型名：{name}")


# Check the model output shape
if __name__ == "__main__":
    x = torch.randn(4, 3, 32, 32)
    for name in ["resnet20", "plain20", "resnet56", "plain56"]:
        model = build_model(name)
        print(name, model(x).shape, sum(p.numel() for p in model.parameters()))
