# CSC3022F ML A1
# Mahir Moodaley (MDLMAH007)
# 16 April 2024
import torch
import torch.nn as nn
from torchvision import datasets
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision.transforms as transforms

# constant variables
image_size = 28*28
hidden_size = 100
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001


# Load MNIST from file
DATA_DIR = "."
download_dataset = False
train_mnist = datasets.MNIST(DATA_DIR, train=True, transform = transforms.ToTensor(), download=download_dataset)
test_mnist = datasets.MNIST(DATA_DIR, train=False, transform = transforms.ToTensor(), download=download_dataset)

print(train_mnist)
print(test_mnist)

# Create variables for MNIST data
train_loader = torch.utils.data.DataLoader(dataset=train_mnist, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_mnist, batch_size=batch_size, shuffle=False)

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
