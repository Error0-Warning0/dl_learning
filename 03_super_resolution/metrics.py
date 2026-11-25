import torch
import torch.nn.functional as F


def rgb_to_y(x):
    """把 (N, 3, H, W) 的 RGB 图转成 (N, 1, H, W) 的亮度图，数值仍在 [0, 1]。"""
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def psnr(pred, target):
    """每张图一个 PSNR（单位 dB），只在亮度通道上算。"""
    pred_y = rgb_to_y(pred)
    target_y = rgb_to_y(target)

    # 每张图的平均平方误差，“.flatten(1)”把后面的 H、W 拉平
    mse = ((pred_y - target_y) ** 2).flatten(1).mean(dim=1)

    # 最大值是 1，所以 PSNR = 10 * log10(1 / MSE)
    # 两张图完全一样时 MSE 是 0，取对数会得到无穷大，所以先把它垫到 1e-10
    return 10 * torch.log10(1.0 / mse.clamp_min(1e-10))


def _gaussian_window(size=11, sigma=1.5, device=None):
    """生成 11×11 的高斯窗口，中间权重大、边缘权重小。"""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()  # 归一化，让权重加起来等于 1

    # 一维窗口相乘，得到二维窗口，形状是 (1, 1, size, size)
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


def ssim(pred, target):
    """每张图一个 SSIM，只在亮度通道上算，理论范围 -1～1，越大越好。"""
    pred_y = rgb_to_y(pred)
    target_y = rgb_to_y(target)
    window = _gaussian_window(device=pred_y.device)

    # 用高斯窗口做卷积，就是“以每个像素为中心算加权平均”
    mu1 = F.conv2d(pred_y, window)  # 预测图的局部平均亮度
    mu2 = F.conv2d(target_y, window)  # 答案图的局部平均亮度
    mu1_sq, mu2_sq = mu1**2, mu2**2

    # 局部方差和协方差
    sigma1_sq = F.conv2d(pred_y**2, window) - mu1_sq
    sigma2_sq = F.conv2d(target_y**2, window) - mu2_sq
    sigma12 = F.conv2d(pred_y * target_y, window) - mu1 * mu2

    # C1、C2 是防止分母为 0 的小常数（原论文里 K1=0.01，K2=0.03）
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    # 每张图把所有位置的 SSIM 平均成一个数
    return ssim_map.flatten(1).mean(dim=1)


# 用已知答案的小例子检查：完全一样应该是 1，差得远应该小于 1
if __name__ == "__main__":
    a = torch.rand(2, 3, 64, 64)
    b = torch.rand(2, 3, 64, 64)
    print("自己和自己比：PSNR", psnr(a, a), "SSIM", ssim(a, a))
    print("两张随机图比：PSNR", psnr(a, b), "SSIM", ssim(a, b))
