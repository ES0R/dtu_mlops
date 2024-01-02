import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(data_dir, train=True):
    """
    Loads the MNIST data from the given directory.
    
    Args:
    - data_dir (str): The path to the directory containing the data files.
    - train (bool): If True, load training data; otherwise, load test data.
    
    Returns:
    - DataLoader: A DataLoader object with the MNIST data.
    """
    # Initialize lists to hold the data tensors.
    images_list = []
    targets_list = []
    
    if train:
        # Load all training images and targets
        for i in range(6):  # Assuming there are 6 parts as seen in the image
            images_list.append(torch.load(f'{data_dir}/train_images_{i}.pt'))
            targets_list.append(torch.load(f'{data_dir}/train_target_{i}.pt'))
    else:
        # Load test images and targets
        images_list.append(torch.load(f'{data_dir}/test_images.pt'))
        targets_list.append(torch.load(f'{data_dir}/test_target.pt'))
    
    # Concatenate all parts into a single tensor
    images = torch.cat(images_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    
    # Create a TensorDataset
    dataset = TensorDataset(images, targets)
    
    # Create a DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    return loader

def mnist(data_dir):
    """Return train and test dataloaders for MNIST."""
    train_loader = load_data(data_dir, train=True)
    test_loader = load_data(data_dir, train=False)
    
    return train_loader, test_loader
