import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from PIL import Image

def main(model_path, data_path, labels_path, custom_image_path=None):
    # Load the model
    model = torch.load(model_path)
    model.eval()

    # Load data and labels
    if os.path.isdir(data_path):
        transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
        dataset = ImageFolder(root=data_path, transform=transform)
    else:  # Assuming it's a .pt file with data
        data = torch.load(data_path)
        labels = torch.load(labels_path)
        dataset = torch.utils.data.TensorDataset(data, labels)

    dataloader = DataLoader(dataset, batch_size=64)


    # Run predictions and calculate accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print(f"Accuracy: {accuracy}")

    if custom_image_path:
        image = Image.open(custom_image_path).convert('L')  # Convert to grayscale
        transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
        image = transform(image).unsqueeze(0)  # Transform and add batch dimension
        print(image.size())

        with torch.no_grad():
            custom_pred = model(image)
            predicted_label = torch.argmax(custom_pred, dim=1)
            print(f"Custom image prediction: {predicted_label.item()}")

    # Paths
model_path = 'C:/Users/Emil/Documents/DTU_git/dtu_mlops/dtu_mlops/models/model.pt'
data_path = 'C:/Users/Emil/Documents/DTU_git/dtu_mlops/dtu_mlops/data/processed/train_data.pt'
labels_path = 'C:/Users/Emil/Documents/DTU_git/dtu_mlops/dtu_mlops/data/processed/train_labels.pt'
custom_image_path = 'C:/Users/Emil/Documents/DTU_git/dtu_mlops/dtu_mlops/dtu_mlops/asd.png'  # Optional

main(model_path, data_path, labels_path, custom_image_path)
