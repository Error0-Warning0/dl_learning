import config
import torch
import torch.nn.functional as F
from data import build_loader, build_test_set
from matplotlib import pyplot as plt
from metrics import psnr, ssim
from model import build_model


@torch.no_grad()
def evaluate(model, loader):
    """同时在测试集上算模型和双三次基线的 PSNR / SSIM。"""
    model.eval()
    model_psnr, model_ssim = [], []
    bicubic_psnr, bicubic_ssim = [], []

    for lr, hr in loader:
        lr, hr = lr.to(config.DEVICE), hr.to(config.DEVICE)

        pred = model(lr).clamp(0, 1)
        up = F.interpolate(
            lr, size=hr.shape[-2:], mode="bicubic", antialias=False
        ).clamp(0, 1)

        model_psnr.append(psnr(pred, hr))
        model_ssim.append(ssim(pred, hr))
        bicubic_psnr.append(psnr(up, hr))
        bicubic_ssim.append(ssim(up, hr))

    return (
        torch.cat(bicubic_psnr).mean().item(),
        torch.cat(bicubic_ssim).mean().item(),
        torch.cat(model_psnr).mean().item(),
        torch.cat(model_ssim).mean().item(),
    )


@torch.no_grad()
def save_comparison(model, loader):
    """保存 4 张测试图：低清、双三次、模型输出、高清。"""
    samples = []
    for lr, hr in loader:
        samples.append((lr, hr))
        if len(samples) == 4:
            break

    row_names = ["LR (nearest ×2)", "bicubic", config.MODEL_NAME.upper(), "HR"]
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, (lr, hr) in enumerate(samples):
        lr_dev, hr_dev = lr.to(config.DEVICE), hr.to(config.DEVICE)

        pred = model(lr_dev).clamp(0, 1).cpu()
        up = (
            F.interpolate(
                lr_dev, size=hr_dev.shape[-2:], mode="bicubic", antialias=False
            )
            .clamp(0, 1)
            .cpu()
        )
        lr_show = F.interpolate(lr, scale_factor=config.SCALE, mode="nearest").clamp(
            0, 1
        )

        images = [lr_show[0], up[0], pred[0], hr[0]]
        for r, img in enumerate(images):
            axes[r, i].imshow(img.permute(1, 2, 0))
            axes[r, i].axis("off")

            if r in (1, 2):  # 只给双三次和模型标分数
                score = psnr(img[None], hr[0][None]).item()
                axes[r, i].set_title(f"{row_names[r]} / {score:.2f} dB", fontsize=8)
            elif i == 0:
                axes[r, i].set_title(row_names[r], fontsize=9)

    fig.tight_layout()
    config.MODEL_DIR.mkdir(exist_ok=True)
    fig.savefig(config.COMPARE_PATH, dpi=150)
    print(f"对比图已保存到 {config.COMPARE_PATH}")


def main():
    print(f"Using device: {config.DEVICE}")

    test_set = build_test_set()
    test_loader = build_loader(test_set, shuffle=False, batch_size=1)

    model = build_model(config.MODEL_NAME).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    print(f"已加载权重：{config.MODEL_PATH}")

    bicubic_psnr, bicubic_ssim, model_psnr, model_ssim = evaluate(model, test_loader)

    print()
    print(f"{'method':10s}  {'PSNR (dB)':>10s}  {'SSIM':>8s}")
    print(f"{'bicubic':10s}  {bicubic_psnr:10.2f}  {bicubic_ssim:8.4f}")
    print(f"{config.MODEL_NAME:10s}  {model_psnr:10.2f}  {model_ssim:8.4f}")
    print(f"提升：{model_psnr - bicubic_psnr:+.2f} dB")

    save_comparison(model, test_loader)


if __name__ == "__main__":
    main()
