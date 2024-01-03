import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import os
from models.model import MyCNNModel


train_data = torch.load('C:/Users/Emil/Documents/DTU_git/dtu_mlops/dtu_mlops/data/processed/train_data.pt')
train_labels = torch.load('C:/Users/Emil/Documents/DTU_git/dtu_mlops/dtu_mlops/data/processed/train_labels.pt')

train_data = train_data.unsqueeze(1)  # Shape becomes [25000, 1, 28, 28]

batchsize = 64
learning_rate = 1e-3
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

validation_split = 0.2
num_train = len(train_data)
split = int(validation_split * num_train)
train_split, val_split = random_split(range(num_train), [num_train - split, split])

train_dataloader = DataLoader(TensorDataset(train_data, train_labels), batch_size=batchsize, sampler=torch.utils.data.SubsetRandomSampler(train_split))
val_dataloader = DataLoader(TensorDataset(train_data, train_labels), batch_size=batchsize, sampler=torch.utils.data.SubsetRandomSampler(val_split))


print(train_data.shape)

model = MyCNNModel()
model.to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for x, y in train_dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_dataloader))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            val_loss += loss_fn(y_pred, y).item()
    val_losses.append(val_loss / len(val_dataloader))
    print(f"Epoch {epoch}: Train Loss {train_losses[-1]}, Validation Loss {val_losses[-1]}")

# Plotting the training and validation loss
plt.figure()
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('C:/Users/Emil/Documents/DTU_git/dtu_mlops/dtu_mlops/reports/figures/training_validation_loss_curve.png')


torch.save(model, 'C:/Users/Emil/Documents/DTU_git/dtu_mlops/dtu_mlops/models/model.pt')
