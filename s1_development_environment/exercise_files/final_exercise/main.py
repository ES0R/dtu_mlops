import click
from model import MyAwesomeModel
import torch
from torch import nn, optim
from data import mnist

@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist(r"C:\Users\Emil\Documents\DTU_git\dtu_mlops\data\corruptmnist")

    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    epochs = 10
    for e in range(epochs):
        for images, labels in train_set:
            images = images.view(images.shape[0], -1)  # Flatten MNIST images
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    torch.save(model, 'trained_model.pt')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)


    # Load the trained model
    model = torch.load(model_checkpoint)
    model.eval()  # Set the model to evaluation mode

    # Load the test data
    _, test_loader = mnist(r"C:\Users\Emil\Documents\DTU_git\dtu_mlops\data\corruptmnist")

    correct = 0
    total = 0
    with torch.no_grad():  # Turn off gradients for evaluation
        for images, labels in test_loader:
            # Assuming the model outputs log probabilities, use torch.exp to get probabilities
            outputs = torch.exp(model(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy}%')


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
