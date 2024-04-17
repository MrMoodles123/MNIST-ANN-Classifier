# CSC3022F ML A1
# Mahir Moodaley (MDLMAH007)
# 16 April 2024
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision.transforms as transforms

# constant variables
image_size = 28*28
hidden_size = 100
hidden_size_2 = 50
num_classes = 10
num_epochs = 10
batch_size = 64
learning_rate = 0.001


# Load MNIST from file
DATA_DIR = "."
download_dataset = False
train_mnist = datasets.MNIST(DATA_DIR, train=True, transform = transforms.ToTensor(), download=download_dataset)
test_mnist = datasets.MNIST(DATA_DIR, train=False, transform = transforms.ToTensor(), download=download_dataset)

# Load MNIST from file
DATA_DIR = "."
download_dataset = False
train_mnist = datasets.MNIST(DATA_DIR, train=True, transform = transforms.ToTensor(), download=download_dataset)
test_mnist = datasets.MNIST(DATA_DIR, train=False, transform = transforms.ToTensor(), download=download_dataset)

print(train_mnist)
print(test_mnist)

# Create variables for MNIST data
X_train = train_mnist.data.float()
y_train = train_mnist.targets
X_test = test_mnist.data.float()
y_test = test_mnist.targets

""" Split training data into training and validation (let validation be the size of test) """

# Sample random indices for validation
test_size = X_test.shape[0]
indices = np.random.choice(X_train.shape[0], test_size, replace=False)

# Create validation set
X_valid = X_train[indices]
y_valid = y_train[indices]

# Remove validation set from training set
X_train = np.delete(X_train, indices, axis=0)
y_train = np.delete(y_train, indices, axis=0)

# Reshape the data
X_train = X_train.reshape(-1, 28*28)
X_valid = X_valid.reshape(-1, 28*28)
X_test = X_test.reshape(-1, 28*28)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class NeuralNetwork(nn.Module):
    def __init__(self, img_size, hidden_size, num_classes):
        super(NeuralNetwork,self).__init__()
        self.linear1 = nn.Linear(img_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out

def read_file(filename):
    return

def main():
    while True:
        input_str = input("Please enter a filepath:\n> ")
        if input_str.strip().lower() == "exit":
            print("Exiting...")
            break
        
if __name__ == "__main__":
    main()
