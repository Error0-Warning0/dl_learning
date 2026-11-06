from pathlib import Path

import torch

MODEL_NAME = "cnn_bn_drop"  # 可选 "cnn"、"cnn_bn"、"cnn_bn_drop"
AUGMENT = True  # 训练集是否做数据增强
# 每次实验的名字，用来给保存的文件起名，比如 cnn_aug
RUN_NAME = f"{MODEL_NAME}_aug" if AUGMENT else MODEL_NAME

# ===== 路径 =====
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"  # CIFAR-10 数据集
MODEL_DIR = BASE_DIR / "saved_models"  # 模型保存的文件夹
MODEL_PATH = MODEL_DIR / f"cifar10_{RUN_NAME}.pth"  # 模型文件
CURVES_PATH = MODEL_DIR / f"curves_{RUN_NAME}.png"  # 训练曲线图

# ===== 超参数 =====
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3

# ===== 数据 =====
TRAIN_SIZE = 45000  # 训练集张数
VAL_SIZE = 5000  # 验证集张数（两者加起来必须等于 50000）
SPLIT_SEED = 42  # 划分验证集用的随机种子
DATASET_MEAN = (0.4914, 0.4822, 0.4465)  # 红绿蓝三个通道均值
DATASET_STD = (0.2470, 0.2435, 0.2616)  # 三个通道标准差
CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# ===== 设备 =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
