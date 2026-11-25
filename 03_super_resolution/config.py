from pathlib import Path

import torch

# ===== 路径 =====
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"  # 数据集
TRAIN_HR_DIR = DATA_DIR / "DIV2K" / "DIV2K_train_HR"
VAL_HR_DIR = DATA_DIR / "DIV2K" / "DIV2K_valid_HR"
TEST_HR_DIR = DATA_DIR / "Set5"
MODEL_DIR = BASE_DIR / "saved_models"

# ===== 超分设置 =====
SCALE = 2  # 放大倍数
HR_PATCH = 96  # 训练时随机裁的高清 patch 边长，必须是 SCALE 的倍数
VAL_PATCH = 256  # 每轮验证时中心裁剪的边长
BATCH_SIZE = 16
NUM_WORKERS = 2

# ===== 数据使用方式 =====
TRAIN_REPEAT = 10  # 每张训练图每轮随机裁几个 patch
VAL_LIMIT = 20  # 每轮验证只用前多少张 DIV2K valid 图

# ===== 设备 =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
