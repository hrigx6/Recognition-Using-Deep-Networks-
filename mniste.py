import torch
import torch.nn as nn 
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import os

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

# Load the pretrained model architecture
model = MyNetwork()
# Load the state dictionary into the model
model.load_state_dict(torch.load("mnist_model.pth"))
# Set the model to evaluation mode
model.eval()

# Now you can use the model for inference or any other purposes


# Define the transform for preprocessing the input images
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize
])

# Load the MNIST test set
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

# Initialize a figure to display the handwritten digits
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
fig.suptitle('Handwritten Digits Classification', fontsize=16)

# Iterate over the first 9 examples in the test set
for i in range(9):
    image, label = test_dataset[i]

    # Perform inference using the model
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)

    # Display the handwritten digit and its classified result
    ax = axes[i // 3, i % 3]
    ax.imshow(image.squeeze().numpy(), cmap='gray')
    ax.set_title(f'Predicted: {predicted.item()}')
    ax.axis('off')

    # Print output values and correct label
    print(f"Example {i+1}:")
    print("Output values (probabilities):")
    for j, prob in enumerate(probabilities.squeeze().tolist()):
        print(f"Class {j}: {prob:.2f}")
    print(f"Predicted Label: {predicted.item()}")
    print(f"Correct Label: {label}\n")


plt.tight_layout()
plt.show()
