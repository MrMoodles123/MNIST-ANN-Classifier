# CSC3022F ML A1
# Mahir Moodaley (MDLMAH007)
# 11 April 2024
#%matplotlib inline
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
hidden_size_1 = 100
hidden_size_2 = 50
num_classes = 10
num_epochs = 5
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

class NeuralNetwork(nn.Module):
    def __init__(self, img_size, hidden_size_1, hidden_size_2, num_classes):
        super(NeuralNetwork,self).__init__()
        self.linear1 = nn.Linear(img_size, hidden_size_1)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.final = nn.Linear(hidden_size_2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = x.view(-1, 28*28)
        out = self.relu(self.linear1(out))
        out = self.relu(self.linear2(out))
        out = self.final(out)
        return out
    
model = NeuralNetwork(image_size, hidden_size_1, hidden_size_2, num_classes)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# training
num_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (image, label) in enumerate(train_loader):
        image = image.reshape(-1, 28*28)
        output = model(image)
        loss = loss_func(output, label)
        loss.backward()
        
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            print(f"epoch {epoch + 1}/{num_epochs}, step {i + 1}/{num_steps}, loss = {loss.item()}")
        
#testing
with torch.no_grad():
    num_correct = 0
    num_samples = 0
    
    for image, label in test_loader:
        #image = image.reshape(-1, 28*28)
        output = model(image.view(-1, 28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == label[idx]:
                num_correct += 1
            num_samples += 1
        #_,prediction = torch.max(output, 1)
        #num_samples += label.shape[0]
        #num_correct += (prediction == label).sum().item()
        
    accuracy = 100.00 * num_correct / num_samples
    print(f"accuracy = {accuracy}")

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