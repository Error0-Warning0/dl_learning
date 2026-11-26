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


class ESPCN(nn.Module):
    """在低清图上卷积，最后用 PixelShuffle 一次放大。"""

    def __init__(self):
        super().__init__()
        # 每个低清像素要预测 SCALE×SCALE 个子像素，每个子像素 3 个颜色
        out_channels = 3 * config.SCALE**2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(config.SCALE)

    def forward(self, lr):
        x = F.relu(self.conv1(lr))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)  # (N, 3*SCALE**2, H, W)
        return self.pixel_shuffle(x)  # (N, 3, H*SCALE, W*SCALE)


def build_model(name):
    if name == "srcnn":
        return SRCNN()
    if name == "espcn":
        return ESPCN()
    raise ValueError(f"未知的模型名：{name}")


if __name__ == "__main__":
    x = torch.randn(4, 3, 48, 48)
    for name in ("srcnn", "espcn"):
        model = build_model(name)
        n_params = sum(p.numel() for p in model.parameters())
        print(name, model(x).shape, n_params)
