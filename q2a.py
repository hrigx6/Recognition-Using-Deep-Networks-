import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
from torchvision import datasets, transforms

# Step 1: Load the trained model
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

model = MyNetwork()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Print the model architecture
print(model)

# Step 2: Analyze the first layer
# Get the weights of the first convolutional layer
weights = model.conv1.weight.data.cpu().numpy()

# Print the shape of the weights tensor
print("Shape of the first layer weights:", weights.shape)

# Visualize the ten filters
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(weights[i, 0])#, #cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Filter {i+1}')
plt.show()

# Step 3: Apply the filters to the first training example image
# Load the MNIST training dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=1, shuffle=True)

# Get the first training example image
example_image, _ = next(iter(train_loader))

# Apply the filters to the first training example image
with torch.no_grad():
    filtered_images = [cv2.filter2D(example_image.squeeze().numpy(), -1, weights[i, 0]) for i in range(10)]

# Visualize the filtered images
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(filtered_images[i], cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Filter Result {i+1}')
plt.show()

