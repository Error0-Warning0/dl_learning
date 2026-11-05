import config
import torch
from model import MLP
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Device configuration (paths and settings are in config.py)
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
test_dataset = datasets.MNIST(
    root=config.DATA_DIR, train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

# Load the trained model
model = MLP().to(device)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))


# Testing function
def evaluate(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    return correct / len(loader.dataset)


# Run evaluation on test set
test_acc = evaluate(test_loader)
print(f"Test Accuracy: {test_acc:.2%}")
