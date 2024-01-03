import torch
import os

def load_and_normalize_data(raw_folder, processed_folder):
    # Load data
    train_data, train_labels = [], []
    for i in range(5):                                          
        train_data.append(torch.load(raw_folder + f"train_images_{i}.pt"))
        train_labels.append(torch.load(os.path.join(raw_folder, f"train_target_{i}.pt")))

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    # Normalize data
    mean = train_data.mean()
    std = train_data.std()
    train_data_normalized = (train_data - mean) / std

    # Save processed data
    torch.save(train_data_normalized, os.path.join(processed_folder, 'train_data.pt'))
    torch.save(train_labels, os.path.join(processed_folder, 'train_labels.pt'))
raw_folder = 'C:/Users/Emil/Documents/DTU_git/dtu_mlops/data/corruptmnist/'
processed_folder = 'C:/Users/Emil/Documents/DTU_git/dtu_mlops/dtu_mlops/data/processed'
load_and_normalize_data(raw_folder, processed_folder)