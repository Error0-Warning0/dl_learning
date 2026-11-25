import config
import torch
import torch.nn.functional as F
from data import build_full_val_set, build_loader, build_test_set
from matplotlib import pyplot as plt
from metrics import psnr, ssim


@torch.no_grad()
def baseline_metrics(loader, mode):
    """把低清图直接用某种插值方法放大，算平均 PSNR 和 SSIM。"""
    all_psnr, all_ssim = [], []
    for lr, hr in loader:
        lr, hr = lr.to(config.DEVICE), hr.to(config.DEVICE)

        # 放大到高清图的尺寸。放大时不需要 antialias
        # 最后裁到 [0, 1]：插值结果可能略微超出范围，标准评测也会先裁到合法像素范围
        up = F.interpolate(lr, size=hr.shape[-2:], mode=mode, antialias=False).clamp(
            0, 1
        )

        all_psnr.append(psnr(up, hr))
        all_ssim.append(ssim(up, hr))

    return torch.cat(all_psnr).mean().item(), torch.cat(all_ssim).mean().item()


@torch.no_grad()
def save_comparison(loader):
    """取 4 张测试图，把双三次放大结果和高清原图上下排好，存成图片。"""
    samples = []
    for lr, hr in loader:
        samples.append((lr, hr))
        if len(samples) == 4:
            break

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for i, (lr, hr) in enumerate(samples):
        up = F.interpolate(
            lr, size=hr.shape[-2:], mode="bicubic", antialias=False
        ).clamp(0, 1)

        axes[0, i].imshow(up[0].permute(1, 2, 0).cpu())
        score = psnr(up, hr).item()
        axes[0, i].set_title(f"bicubic / {score:.2f} dB", fontsize=9)
        axes[1, i].imshow(hr[0].permute(1, 2, 0).cpu())
        axes[1, i].set_title(f"HR {hr.shape[-2]}×{hr.shape[-1]}", fontsize=9)
        axes[0, i].axis("off")
        axes[1, i].axis("off")

    fig.tight_layout()
    config.MODEL_DIR.mkdir(exist_ok=True)
    fig.savefig(config.MODEL_DIR / "baseline_compare.png", dpi=150)
    print(f"对比图已保存到 {config.MODEL_DIR / 'baseline_compare.png'}")


def main():
    print(f"Using device: {config.DEVICE}")

    for name, dataset in [
        ("DIV2K valid", build_full_val_set()),
        ("Set5", build_test_set()),
    ]:
        loader = build_loader(dataset, shuffle=False, batch_size=1)
        print(f"\n{name}（{len(dataset)} 张）")
        for mode in ["nearest", "bilinear", "bicubic"]:
            score_psnr, score_ssim = baseline_metrics(loader, mode)
            print(f"{mode:8s}  PSNR {score_psnr:.2f} dB  SSIM {score_ssim:.4f}")

    save_comparison(build_loader(build_test_set(), shuffle=False, batch_size=1))


if __name__ == "__main__":
    main()
