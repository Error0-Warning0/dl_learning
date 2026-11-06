import config
import torch
from model import build_model
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Device configuration (paths and settings are in config.py)
device = config.DEVICE
print(f"Using device: {device}")

# Data preprocessing (transformations)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(config.DATASET_MEAN, config.DATASET_STD),
    ]
)

# datasets
test_dataset = datasets.CIFAR10(
    root=config.DATA_DIR, train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# Load the trained model
model = build_model(config.MODEL_NAME).to(device)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))


# Testing function
def evaluate(loader):
    model.eval()
    correct = 0
    class_correct = [0] * 10  # 每个类别识别正确几张
    class_total = [0] * 10  # 每个类别总数
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            for c in range(10):
                mask = labels == c
                class_total[c] += mask.sum().item()
                class_correct[c] += (preds[mask] == c).sum().item()
    return correct / len(loader.dataset), class_correct, class_total


# Run evaluation on test set
test_acc, class_correct, class_total = evaluate(test_loader)
print(f"Test Accuracy: {test_acc:.2%}")
for c in range(10):
    print(f"{config.CLASSES[c]:>10}:{class_correct[c] / class_total[c]:.2%}")
