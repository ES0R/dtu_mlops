
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

train_data, train_labels = [], []

batchsize = 64
learning_rate = 1e-3
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in range(5):
    train_data.append(torch.load(f"train_images_{i}.pt"))
    train_labels.append(torch.load(f"train_target_{i}.pt"))

train_data = torch.cat(train_data, dim = 0)
train_labels = torch.cat(train_labels,dim = 0)

test_data = torch.load("test_images.pt")
test_labels = torch.load("test_target.pt")

print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

train_data = train_data.unsqueeze(1)
test_data = test_data.unsqueeze(1)

train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batchsize)

test_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_data, test_labels), batch_size=batchsize)

model = nn.Sequential(
    nn.Conv2d(1,32,3),  # [B,1,28,28] -> [B,32,26,26]
    nn.LeakyReLU(),
    nn.Conv2d(32,64,3),  # [B,32,26,26] -> [B,64,24,24]
    nn.MaxPool2d(2),    # [B, 64,12,12]
    nn.Flatten(),       # [B,64*12*12]
    nn.Linear(64*12*12,10)
)

model = model.to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


for epoch in range(epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        x,y = batch
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)

        loss = loss_fn(y_pred,y)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} loss {loss}")

model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for batch in test_dataloader:
        x,y = batch
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        test_preds.append(y_pred.argmax(dim=1).cpu)
        test_labels.append(y.cpu)
test_preds = torch.cat(test_preds, dim=0)
test_labels = torch.cat(test_labels, dim = 0)

print((test_preds == y_pred).mean)