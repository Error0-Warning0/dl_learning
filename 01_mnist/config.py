from pathlib import Path

import torch

# ===== 路径 =====
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"  # MNIST 数据集
MODEL_DIR = BASE_DIR / "saved_models"  # 模型保存的文件夹
MODEL_PATH = MODEL_DIR / "mnist_mlp.pth"  # 模型文件
CURVES_PATH = MODEL_DIR / "training_curves.png"  # 训练曲线图

# ===== 超参数 =====
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3

# ===== 数据 =====
TRAIN_SIZE = 55000  # 训练集张数
VAL_SIZE = 5000  # 验证集张数（两者加起来必须等于 60000）
SPLIT_SEED = 42  # 划分验证集用的随机种子
MNIST_MEAN = 0.1307  # MNIST 像素均值
MNIST_STD = 0.3081  # MNIST 像素标准差

# ===== 设备 =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
