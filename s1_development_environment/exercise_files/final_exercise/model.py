from torch import nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
    
    def forward(self, x):
        # Hidden layer with sigmoid activation

        x = F.softmax(self.fc1(x), dim=1)
        return x
