import os

from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    # from "root/label/image.png" reads the image and label
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []

        for label_name in sorted(os.listdir(root)):
            label_dir = os.path.join(root, label_name)
            if not os.path.isdir(label_dir):
                continue

            for file_name in sorted(os.listdir(label_dir)):
                if file_name.lower().endswith(".png"):
                    path = os.path.join(label_dir, file_name)
                    self.samples.append((path, int(label_name)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# Test CustomDataset
if __name__ == "__main__":
    from torchvision import transforms

    ROOT = "/home/kg/python/DL_PyTorch_new/learn_DL/mnist_images"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    ds = CustomDataset(f"{ROOT}/train", transform=transform)
    print(len(ds))  # 60000
    print(ds.samples[0])  # ('.../train/0/1.png', 0)

    image, label = ds[0]
    print(image.shape, label)  # torch.Size([1, 28, 28]) 0
