import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from mnist1 import MyNetwork

# Define the transform for preprocessing the input images
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize
])

# Specify the directory containing the handwritten digit images
images_directory = "final_digits"

# Load the model
model = MyNetwork()
model.load_state_dict(torch.load("mnist_model.pth"))
model.eval()

# Initialize a figure to display the handwritten digits and their classified results
fig, axes = plt.subplots(1, 10, figsize=(20, 5))

# Iterate over the digit images
for i in range(10):
    # Construct the image path
    image_path = os.path.join(images_directory, f"digit_{i}.jpg")

    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found.")
        continue

    # Load and preprocess the image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    preprocessed_image = transform(image).unsqueeze(0)

    # Perform inference using the model
    with torch.no_grad():
        output = model(preprocessed_image)
        _, predicted = torch.max(output, 1)

    # Display the handwritten digit and its classified result
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Predicted: {predicted.item()}')
    axes[i].axis('off')

plt.show()
