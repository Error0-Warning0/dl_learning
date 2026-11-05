import os

import config
import torch
from matplotlib import pyplot as plt
from model import CustomModel
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

# Device configuration (hyperparameters and paths are in config.py)
device = config.DEVICE
print(f"Using device: {device}")

# Data preprocessing (transformations)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((config.MNIST_MEAN,), (config.MNIST_STD,)),
    ]
)

# datasets
full_train_dataset = datasets.MNIST(
    root=config.DATA_DIR, train=True, download=True, transform=transform
)
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [config.TRAIN_SIZE, config.VAL_SIZE],
    generator=torch.Generator().manual_seed(config.SPLIT_SEED),
)
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# model, loss function, and optimizer
model = CustomModel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


# Training function and evaluation function
def train_one_epoch():
    model.train()
    total_loss = 0
    correct = 0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
    train_loss = total_loss / len(train_loader)
    train_acc = correct / len(train_dataset)
    return train_loss, train_acc


def evaluate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_loss += loss.item()
            correct += (preds == labels).sum().item()
        val_loss = total_loss / len(loader)
        val_acc = correct / len(loader.dataset)
        return val_loss, val_acc


# main training loop
best_acc = 0.0
os.makedirs(config.MODEL_DIR, exist_ok=True)
history_train_loss = []
history_val_loss = []
history_train_acc = []
history_val_acc = []

for epoch in range(config.EPOCHS):
    train_loss, train_acc = train_one_epoch()
    val_loss, val_acc = evaluate(val_loader)
    history_train_loss.append(train_loss)
    history_val_loss.append(val_loss)
    history_train_acc.append(train_acc)
    history_val_acc.append(val_acc)
    print(
        f"Epoch {epoch + 1}/{config.EPOCHS}, "
        f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
        f"Train Acc: {train_acc:.2%}, Val Acc: {val_acc:.2%}"
    )
    # Save the best trained model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), config.MODEL_PATH)
        print(f"New Best Model Saved - Validation Accuracy: {best_acc:.2%}")

print(f"Model saved to {config.MODEL_PATH}")
print(f"Best Validation Accuracy: {best_acc:.2%}")

# draw training curves
epochs = range(1, config.EPOCHS + 1)  # 横轴从第 1 轮开始，和打印的轮次对应
plt.figure(figsize=(12, 5))  # 整张图宽 12、高 5（单位是英寸）

plt.subplot(1, 2, 1)  # 1 行 2 列，现在画第 1 个
plt.plot(epochs, history_train_loss, label="Train Loss")
plt.plot(epochs, history_val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()  # 显示图例（label 的文字）
plt.grid(True)  # 显示网格

plt.subplot(1, 2, 2)  # 画第 2 个
plt.plot(epochs, history_train_acc, label="Train Acc")
plt.plot(epochs, history_val_acc, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()  # 自动调整间距，防止文字重叠
plt.savefig(config.CURVES_PATH)
print(f"Training curves saved to {config.CURVES_PATH}")
# plt.show()  # 弹出窗口显示
