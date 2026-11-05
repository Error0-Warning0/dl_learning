# This file is used to predict the images from your own handwritten images using the trained model.
# Usage: python predict.py image path
import sys

import config
import torch
from model import MLP
from PIL import Image, ImageOps
from torchvision import transforms

# Data preprocessing (transformations)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # the values are the same as those used in training (from config.py)
        transforms.Normalize((config.MNIST_MEAN,), (config.MNIST_STD,)),
    ]
)


def preprocess(path):
    image = Image.open(path).convert("L")
    # sys.argv[1] is the path to the image file；"L" is tramsformed to grayscale
    image = ImageOps.invert(image)
    # color inversion, because the model was trained on white digits on black background
    # image binarazation
    threshold = image.getextrema()[1] * 0.6
    image = image.point(lambda p: 255 if p > threshold else 0)
    # crop the image
    image = image.crop(image.getbbox())
    # make the image square
    w, h = image.size
    size = max(w, h)
    square = Image.new("L", (size, size), 0)
    square.paste(image, ((size - w) // 2, (size - h) // 2))
    # resize the image to 20x20 and paste it in a 28x28 black image
    square = square.resize((20, 20))
    result = Image.new("L", (28, 28), 0)
    result.paste(square, (4, 4))
    return result


image = preprocess(sys.argv[1])
x = transform(image).unsqueeze(0)
# (1,28,28) → (1,1,28,28), it is necessary to add a batch dimension for the model input

# Load the trained model
model = MLP()
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Make prediction
with torch.no_grad():
    probs = torch.softmax(model(x), dim=1)[0]  # get the probabilities of each class

print(
    f"Predicted digit: {probs.argmax().item()} Confidence: {probs.max().item():.2%}"
)  # print the predicted digit and its confidence
