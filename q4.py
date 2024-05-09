import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt

# Check CUDA availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the network architecture
class CustomNetwork(nn.Module):
    def _init_(self, num_conv_layers, num_filters, filter_size, num_hidden_nodes, dropout_rate, train_loader):
        super(CustomNetwork, self)._init_()
        self.conv_layers = nn.ModuleList()
        self.num_conv_layers = num_conv_layers
        self.in_channels = 1

        for _ in range(num_conv_layers):
            self.conv_layers.append(nn.Conv2d(self.in_channels, num_filters, kernel_size=filter_size, padding=1).to(device))
            self.conv_layers.append(nn.ReLU().to(device))
            self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2).to(device))
            self.in_channels = num_filters

        # Dynamic calculation of flatten size
        self._calculate_flatten_size(train_loader)

        self.fc1 = nn.Linear(self.flatten_size, num_hidden_nodes).to(device)
        self.fc2 = nn.Linear(num_hidden_nodes, 10).to(device)
        self.dropout = nn.Dropout(dropout_rate).to(device)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(-1, self.flatten_size)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def _calculate_flatten_size(self, train_loader):
        sample_data = next(iter(train_loader))[0].to(device)
        x = sample_data.new_zeros(1, *sample_data.shape[1:]).to(device)
        for layer in self.conv_layers:
            x = layer(x)
        self.flatten_size = x.view(1, -1).size(1)  # Calculate flatten size dynamically

import time

# Function to perform experiments
def perform_experiment(num_conv_layers_list, num_filters_list, filter_size_list,
                       num_hidden_nodes_list, dropout_rate_list, epochs, batch_size, num_variations):
    # Load Fashion MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Randomly select combinations for experimentation
    experiments = []
    for _ in range(num_variations):
        num_conv_layers = random.choice(num_conv_layers_list)
        num_filters = random.choice(num_filters_list)
        filter_size = random.choice(filter_size_list)
        num_hidden_nodes = random.choice(num_hidden_nodes_list)
        dropout_rate = random.choice(dropout_rate_list)

        experiments.append((num_conv_layers, num_filters, filter_size, num_hidden_nodes, dropout_rate))

    # Experimentation loop
    results = []
    for idx, (num_conv_layers, num_filters, filter_size, num_hidden_nodes, dropout_rate) in enumerate(experiments):
        print(f"Experiment {idx + 1}/{num_variations}: Conv Layers={num_conv_layers}, Filters={num_filters}, "
              f"Filter Size={filter_size}, Hidden Nodes={num_hidden_nodes}, Dropout={dropout_rate}")

        start_time = time.time()  # Start time for the experiment

        # Initialize the model
        model = CustomNetwork(num_conv_layers, num_filters, filter_size, num_hidden_nodes, dropout_rate, train_loader).to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        train_losses, test_losses = [], []
        train_accuracies, test_accuracies = [], []
        for epoch in range(epochs):
            train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
            test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

        end_time = time.time()  # End time for the experiment
        experiment_time = end_time - start_time  # Time taken for the experiment
        print(f"Experiment {idx + 1}/{num_variations} completed in {experiment_time:.2f} seconds")

        results.append({
            'num_conv_layers': num_conv_layers,
            'num_filters': num_filters,
            'filter_size': filter_size,
            'num_hidden_nodes': num_hidden_nodes,
            'dropout_rate': dropout_rate,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'test_losses': test_losses,
            'test_accuracies': test_accuracies
        })

    return results


# Function to plot results
def plot_results(results):
    for idx, result in enumerate(results):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(result['train_losses'], label='Train Loss', color='blue')
        plt.plot(result['test_losses'], label='Test Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Experiment {idx + 1}: Training and Test Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(result['train_accuracies'], label='Train Accuracy', color='blue')
        plt.plot(result['test_accuracies'], label='Test Accuracy', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Experiment {idx + 1}: Training and Test Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

# Function to train the network
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total

    return train_loss, train_accuracy

# Function to evaluate the network
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_accuracy = correct / total

    return test_loss, test_accuracy

# Function to save results
import os
import json
import time

def save_results(results):
    # Create directory for experiment results if it doesn't exist
    if not os.path.exists('experiment_results'):
        os.makedirs('experiment_results')

    for idx, result in enumerate(results):
        # Create subdirectory for each experiment
        exp_dir = f'experiment_results/exp_{idx + 1}'
        os.makedirs(exp_dir, exist_ok=True)

        # Save results to a JSON file
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(result, f)

        # Plot and save figures
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(result['train_losses'], label='Train Loss', color='blue')
        plt.plot(result['test_losses'], label='Test Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Experiment {idx + 1}: Training and Test Loss')
        plt.legend()
        plt.savefig(os.path.join(exp_dir, 'loss_plot.png'))  # Save loss plot

        plt.subplot(1, 2, 2)
        plt.plot(result['train_accuracies'], label='Train Accuracy', color='blue')
        plt.plot(result['test_accuracies'], label='Test Accuracy', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Experiment {idx + 1}: Training and Test Accuracy')
        plt.legend()
        plt.savefig(os.path.join(exp_dir, 'accuracy_plot.png'))  # Save accuracy plot

        plt.close()  # Close the plot to free memory

# Function to perform experiments and calculate time taken
def perform_experiment_with_timing(num_conv_layers_list, num_filters_list, filter_size_list,
                       num_hidden_nodes_list, dropout_rate_list, epochs, batch_size, num_variations):
    results = perform_experiment(num_conv_layers_list, num_filters_list, filter_size_list,
                       num_hidden_nodes_list, dropout_rate_list, epochs, batch_size, num_variations)
    return results

if _name_ == "_main_":
    num_conv_layers_list = [2, 3, 4, 5]
    num_filters_list = [32, 64]
    filter_size_list = [3]
    num_hidden_nodes_list = [64,128, 256]
    dropout_rate_list = [0.1,0.3, 0.5]
    epochs = 10
    batch_size = 32
    num_variations = 50
    # Adjust as needed

    results = perform_experiment_with_timing(num_conv_layers_list, num_filters_list, filter_size_list,
                                 num_hidden_nodes_list, dropout_rate_list, epochs, batch_size, num_variations)
    plot_results(results)
    save_results(results)
