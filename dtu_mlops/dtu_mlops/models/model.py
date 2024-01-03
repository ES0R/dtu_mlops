import torch
from torch import nn
import torch.nn.functional as F
class MyCNNModel(nn.Module):
    def __init__(self):
        super(MyCNNModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjust input size based on your actual input size
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)  # Use -1 to handle variable batch sizes
        # Convolutional layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 7 * 7)  # Adjust the size based on your actual input size

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x