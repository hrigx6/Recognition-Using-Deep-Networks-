import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Import the pre-trained MNIST network architecture from Task 1
from mnist1 import MyNetwork

# Load the checkpoint
checkpoint = torch.load('mnist_model.pth')

# Print the keys of the loaded checkpoint
print(checkpoint.keys())

# Define the transform for the Greek dataset
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = transforms.functional.rgb_to_grayscale(x)
        x = transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = transforms.functional.center_crop(x, (28, 28))
        return transforms.functional.invert(x)

# Load the pre-trained MNIST model
network = MyNetwork()

# Freeze the network weights
for param in network.parameters():
    param.requires_grad = False

# Modify the last layer for Greek letter classification
network.fc1 = nn.Linear(320, 50)  # Modify fc1 to match input size of fc2
network.fc2 = nn.Linear(50, 3)  # Change the output size to 3 for Greek letters

# Load the pre-trained model checkpoint for fine-tuning
checkpoint = torch.load('fine_tuned_model.pth')
network.load_state_dict(checkpoint)
print(network)
# Testing with your own examples
def predict_greek_letter(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image).unsqueeze(0)

    # Make the prediction
    network.eval()
    with torch.no_grad():
        output = network(image)
        _, predicted = torch.max(output.data, 1)
        class_labels = ['Alpha', 'Beta', 'Gamma']
        predicted_label = class_labels[predicted.item()]
    return predicted_label

# Specify the paths to your own example images
example_images = ['alpha.jpg', 'beta.jpg', 'gamma.jpg']

# Iterate over the example images and make predictions
for image_path in example_images:
    predicted_label = predict_greek_letter(image_path)
    print(f"Image: {image_path}, Predicted Label: {predicted_label}")
