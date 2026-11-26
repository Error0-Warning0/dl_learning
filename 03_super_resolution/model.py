import config
import torch
import torch.nn.functional as F
from torch import nn


class SRCNN(nn.Module):
    """先把低清图用双三次放大到目标尺寸，再用 3 层卷积修细节。"""

    def __init__(self):
        super().__init__()
        # 1) 9×9 大卷积：看足够大的范围，才能判断“这条边本来该是什么样”
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        # 2) 1×1 卷积：只在通道之间做组合，不改变空间尺寸
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        # 3) 5×5 卷积：输出 3 个通道（RGB），尺寸和输入一样
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, lr):
        # 低清图 (N,3,48,48) -> 双三次放大 -> (N,3,96,96)
        x = F.interpolate(
            lr, scale_factor=config.SCALE, mode="bicubic", antialias=False
        )
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)


def build_model(name):
    if name == "srcnn":
        return SRCNN()
    raise ValueError(f"未知的模型名：{name}")


if __name__ == "__main__":
    x = torch.randn(4, 3, 48, 48)
    model = build_model("srcnn")
    print(model(x).shape)
    print(sum(p.numel() for p in model.parameters()))
