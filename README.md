# Recognition using Deep Networks

This project focuses on building, training, analyzing, and modifying a deep neural network for handwritten digit recognition using the MNIST dataset. The project is implemented in Python, utilizing the PyTorch library for building and training convolutional neural networks (CNNs).

## Tasks

The project covers the following main tasks:

1. **Build and Train a Network for Digit Recognition**: Build and train a CNN to recognize handwritten digits from the MNIST dataset, evaluate its performance, and save the trained model.
2. **Examine the Network**: Analyze the structure and weights of the trained network, visualize the learned filters, and observe their effects on input images.
3. **Transfer Learning on Greek Letters**: Adapt the pre-trained MNIST network to recognize Greek letters (alpha, beta, gamma) using transfer learning techniques.
4. **Design Your Own Experiment**: Experiment with various architectural modifications to the network and evaluate their impact on performance and training dynamics.

## Usage

1. Clone the repository:

```
git clone https://github.com/your-username/recognition-using-deep-networks.git
```

2. Install the required dependencies:

```
pip install torch torchvision matplotlib opencv-python
```

3. Run the individual Python scripts for each task:

```
python mnist1.py  # Task 1: Build and Train a Network for Digit Recognition
python mniste.py  # Task 1: Read the Network and Run it on the Test Set
python q1fnew.py  # Task 1: Test the Network on New Inputs
python q2a.py  # Task 2: Examine the Network
python q3a.py  # Task 3: Transfer Learning on Greek Letters
python q4.py  # Task 4: Design Your Own Experiment
python test.py  # Task 4: Execute the Experiment
```

## Project Structure

- `mnist1.py`: Implementation of Task 1 (A-D): Building and training the MNIST digit recognition network.
- `mniste.py`: Implementation of Task 1 (E): Reading the trained network and running it on the test set.
- `q1fnew.py`: Implementation of Task 1 (F): Testing the network on new handwritten digit inputs.
- `q2a.py`: Implementation of Task 2 (A, B): Analyzing the first layer of the network and visualizing the learned filters.
- `q3a.py`: Implementation of Task 3: Transfer learning on Greek letters using the pre-trained MNIST network.
- `q4.py`: Implementation of Task 4: Designing and executing experiments with various network architectural modifications.
- `test.py`: Supporting script for executing the experiments in Task 4.

