import torch
from lenet import LeNet
from visualize_utils import train_with_logging
from torchvision import datasets, transforms
import torch.optim as optim

def main():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, 
        transform=transforms.Compose([transforms.ToTensor()])), 
        batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        ])), batch_size=9, shuffle=True)

    test_batch = None
    for x in test_loader:
        test_batch = x[0]
        break

    # Use cpu if no cuda available
    device = torch.device("cuda")
    #device = torch.device("cpu")

    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    train_with_logging(model, device, train_loader, optimizer, 200, 5, test_batch)


if __name__ == '__main__':
    main()
