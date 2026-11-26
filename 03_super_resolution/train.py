import config
import torch
from data import build_datasets, build_loader
from matplotlib import pyplot as plt
from metrics import psnr, ssim
from model import build_model
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

device = config.DEVICE


@torch.no_grad()
def evaluate_loader(model, loader):
    """在某个数据集上算平均 PSNR 和 SSIM。"""
    model.eval()
    all_psnr, all_ssim = [], []
    for lr, hr in loader:
        lr, hr = lr.to(device), hr.to(device)
        pred = model(lr).clamp(0, 1)  # 评测前裁到合法像素范围
        all_psnr.append(psnr(pred, hr))
        all_ssim.append(ssim(pred, hr))
    return torch.cat(all_psnr).mean().item(), torch.cat(all_ssim).mean().item()


def train_one_epoch(model, loader, loss_fn, optimizer, epoch):
    model.train()
    total_loss = 0.0
    for lr, hr in tqdm(loader, leave=False, desc=f"Epoch {epoch}"):
        lr, hr = lr.to(device), hr.to(device)
        pred = model(lr)
        loss = loss_fn(pred, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * lr.size(0)
    return total_loss / len(loader.dataset)


def main():
    print(f"Using device: {device}")
    config.MODEL_DIR.mkdir(exist_ok=True)

    train_set, val_set, _ = build_datasets()
    train_loader = build_loader(train_set, shuffle=True)
    val_loader = build_loader(val_set, shuffle=False, batch_size=1)

    model = build_model(config.MODEL_NAME).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    if config.USE_COSINE:
        scheduler = CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS, eta_min=config.LR_MIN
        )
    else:
        scheduler = None

    best_psnr = 0.0
    best_epoch = 0
    history_loss = []
    history_psnr = []
    history_ssim = []

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch)
        val_psnr, val_ssim = evaluate_loader(model, val_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        history_loss.append(train_loss)
        history_psnr.append(val_psnr)
        history_ssim.append(val_ssim)

        print(
            f"Epoch {epoch:2d}/{config.EPOCHS}  loss {train_loss:.5f}  "
            f"val PSNR {val_psnr:.2f} dB  val SSIM {val_ssim:.4f}  lr {current_lr:.1e}"
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            torch.save(model.state_dict(), config.MODEL_PATH)

        if scheduler is not None:
            scheduler.step()

    print(f"最好的验证 PSNR 是第 {best_epoch} 轮的 {best_psnr:.2f} dB")
    print(f"权重已保存到 {config.MODEL_PATH}")

    epochs = range(1, len(history_loss) + 1)
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss (MSE)")
    plt.title("Training Loss")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history_psnr)
    plt.xlabel("Epoch")
    plt.ylabel("PSNR (dB)")
    plt.title("Validation PSNR")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history_ssim)
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("Validation SSIM")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(config.CURVES_PATH)
    print(f"曲线已保存到 {config.CURVES_PATH}")


if __name__ == "__main__":
    main()
