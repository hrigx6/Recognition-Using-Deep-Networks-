# mnist.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.log_softmax(self.fc2(x), dim=1)
        return x

def train_network(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        train_loss_history.append(train_loss / len(train_loader))
        train_acc = 100 * correct_train / total_train
        train_acc_history.append(train_acc)

        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_loss_history.append(test_loss / len(test_loader))
        test_acc = 100 * correct_test / total_test
        test_acc_history.append(test_acc)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_history[-1]:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Test Loss: {test_loss_history[-1]:.4f}, '
              f'Test Acc: {test_acc:.2f}%')

    return train_loss_history, train_acc_history, test_loss_history, test_acc_history

# Only executed when running this file directly, not when importing
if __name__ == "__main__":
    # Step 2: Load the MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

    # Step 6: Prepare data loaders and train the model
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = MyNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loss_history, train_acc_history, test_loss_history, test_acc_history = train_network(model, train_loader, test_loader, criterion, optimizer)

    # Plotting training and testing error
    plt.plot(train_loss_history, label='Training Error', color='blue')
    plt.plot(test_loss_history, label='Testing Error', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Error over Epochs')
    plt.legend()
    plt.show()

    # Plotting training and testing accuracy
    plt.plot(train_acc_history, label='Training Accuracy', color='blue')
    plt.plot(test_acc_history, label='Testing Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracy over Epochs')
    plt.legend()
    plt.show()

    # Step 7: Save the trained model
    torch.save(model.state_dict(), 'mnist_model.pth')

