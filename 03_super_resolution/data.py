import random
from pathlib import Path

import config
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(root):
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.suffix.lower() in IMG_EXTS)


def load_image(path):
    return to_tensor(Image.open(path).convert("RGB"))


def modcrop(hr, scale):
    """把宽高裁成 scale 的整数倍，避免缩小/放大后差 1 个像素。"""
    h, w = hr.shape[-2:]
    return hr[:, : h - h % scale, : w - w % scale]


def downsample(hr, scale):
    h, w = hr.shape[-2:]
    lr = F.interpolate(
        hr.unsqueeze(0),
        size=(h // scale, w // scale),
        mode="bicubic",
        antialias=True,
    )
    return lr.squeeze(0)


class DIV2KDataset(Dataset):
    """读取 HR 图片，按需裁 patch 并现场生成 LR。"""

    def __init__(
        self, hr_dir, scale, patch_size=None, random_crop=False, repeat=1, limit=None
    ):
        self.paths = list_images(hr_dir)
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(f"{hr_dir} 里没有图片")

        self.scale = scale
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.repeat = repeat

    def __len__(self):
        return len(self.paths) * self.repeat

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        hr = modcrop(load_image(path), self.scale)

        if self.patch_size is not None:
            ps = self.patch_size
            h, w = hr.shape[-2:]
            if h < ps or w < ps:
                raise ValueError(f"{path} 只有 {h}×{w}，比 patch {ps} 小")

            if self.random_crop:
                top = random.randint(0, h - ps)
                left = random.randint(0, w - ps)
            else:
                top = (h - ps) // 2
                left = (w - ps) // 2
            hr = hr[:, top : top + ps, left : left + ps]

        return downsample(hr, self.scale), hr


def build_datasets():
    """训练用随机 patch，验证用中心裁剪，测试用完整图。"""
    train_set = DIV2KDataset(
        config.TRAIN_HR_DIR,
        config.SCALE,
        patch_size=config.HR_PATCH,
        random_crop=True,
        repeat=config.TRAIN_REPEAT,
    )
    val_set = DIV2KDataset(
        config.VAL_HR_DIR,
        config.SCALE,
        patch_size=config.VAL_PATCH,
        random_crop=False,
        limit=config.VAL_LIMIT,
    )
    test_set = DIV2KDataset(config.TEST_HR_DIR, config.SCALE)
    return train_set, val_set, test_set


def build_full_val_set():
    """完整 DIV2K 验证集，用来跑第 1 步的基线。"""
    return DIV2KDataset(config.VAL_HR_DIR, config.SCALE)


def build_test_set():
    """完整测试集（Set5），用来跑最终成绩。"""
    return DIV2KDataset(config.TEST_HR_DIR, config.SCALE)


def build_loader(dataset, shuffle, batch_size=None):
    return DataLoader(
        dataset,
        batch_size=batch_size or config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda"),
    )


# 检查形状：训练 patch 的低清是 3×48×48，高清是 3×96×96
if __name__ == "__main__":
    train_set, val_set, test_set = build_datasets()
    print("训练集", len(train_set), "验证集", len(val_set), "测试集", len(test_set))

    train_loader = build_loader(train_set, shuffle=True)
    lr, hr = next(iter(train_loader))
    print("训练 batch：低清", lr.shape, "高清", hr.shape)

    val_loader = build_loader(val_set, shuffle=False, batch_size=1)
    lr, hr = next(iter(val_loader))
    print("验证 batch：低清", lr.shape, "高清", hr.shape)

    test_loader = build_loader(test_set, shuffle=False, batch_size=1)
    lr, hr = next(iter(test_loader))
    print("测试 batch：低清", lr.shape, "高清", hr.shape)
