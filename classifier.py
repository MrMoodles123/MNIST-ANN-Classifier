# CSC3022F ML A1
# Mahir Moodaley (MDLMAH007)
# 21 April 2024

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import random
from PIL import Image

# constants
image_size = 28*28
num_classes = 10
weight_decay = 1e-5  # L2 regularization weight decay

# hyperparameters
batch_size = 64
hidden_size = 512
hidden_size_2 = 64
dropout_prob = 0.3 # probability of dropout 
num_epochs = 15
learning_rate = 0.0001

# Load MNIST from file
DATA_DIR = "."
download_dataset = False
train_mnist = datasets.MNIST(DATA_DIR, train=True, transform = transforms.ToTensor(), download=download_dataset)
test_mnist = datasets.MNIST(DATA_DIR, train=False, transform = transforms.ToTensor(), download=download_dataset)

print(train_mnist)
print(test_mnist)
print()

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
X_train = X_train.reshape(-1, image_size)
X_valid = X_valid.reshape(-1, image_size)
X_test = X_test.reshape(-1, image_size)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_valid, y_valid)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class NeuralNetwork(nn.Module):
    def __init__(self, img_size, hidden_size, hidden_size_2, num_classes, dropout_prob):
        super(NeuralNetwork, self).__init__()
        
        self.linear1 = nn.Linear(img_size, hidden_size) # linear layer 1
        self.dropout1 = nn.Dropout(dropout_prob) # dropout layer 1
        self.linear2 = nn.Linear(hidden_size, hidden_size_2) # linear layer 2
        self.dropout2 = nn.Dropout(dropout_prob) # dropout layer 2
        self.final = nn.Linear(hidden_size_2, num_classes) # output layer
        self.relu = nn.ReLU() # ReLU layer       
        
    def forward(self, x):   
        out = x.view(-1, 28*28)
        out = self.relu(self.linear1(out)) # apply ReLU accross hidden layer 1
        out = self.dropout1(out) # apply dropout accross hidden layer 1
        out = self.relu(self.linear2(out)) # apply ReLU accross hidden layer 2
        out = self.dropout2(out) # apply dropout accross hidden layer 2
        out = self.final(out)    
        out = torch.softmax(out, dim = 1) # apply softmax accross last layer
        return out
    
model = NeuralNetwork(image_size, hidden_size, hidden_size_2, num_classes, dropout_prob) # instantiate model
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay) # create Adam Optimization func

# training
num_steps = len(train_loader)
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i, (image, label) in enumerate(train_loader):
        image = image.reshape(-1, 28*28)
        output = model(image)
        loss = F.cross_entropy(output, label) # cross entropy loss function
        
        # calculate L2 regularization penalty
        regularization_loss = 0.0
        for param in model.parameters():
            regularization_loss += torch.norm(param, 2) 
        total_loss = loss + weight_decay * regularization_loss
        epoch_loss += total_loss.item()
        
        # compute back propagation and step the optimizer forward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # periodically print out loss and validation accuracy
        if (i + 1) % batch_size == 0:
            rounded_loss = "{:.4f}".format(total_loss.item())
            print(f"Epoch: {epoch + 1}/{num_epochs}, Step: {i + 1}/{num_steps}, Loss: {rounded_loss}")
            
            # validation accuracy testing
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in val_loader:
                    images = images.reshape(-1, 28*28)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    
                accuracy = 100.00 * correct.item() / total
                print(f"Validation Accuracy: {accuracy}%")
                model.train()
            
    # accuracy and loss after each epoch
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0            
        
        for images, labels in val_loader:
            images = images.reshape(-1, 28*28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            
        accuracy = 100.00 * correct.item() / total
        print(f"Validation Accuracy after epoch {epoch + 1}: {accuracy}%") 
        ave_loss = "{:.4f}".format(epoch_loss / num_steps)
        print(f"Average Loss for epoch {epoch + 1}: {ave_loss}\n")
        model.train()
        
# testing
model.eval()
with torch.no_grad():
    num_correct = 0
    total = 0
    
    for image, label in test_loader:
        image = image.reshape(-1, 28*28)
        output = model(image)
        
        _,prediction = torch.max(output.data, 1)
        total += label.size(0)
        num_correct += (prediction == label).sum().item()
        
    accuracy = 100.00 * num_correct / total
    print(f"Test Accuracy on 10000 images: {accuracy}%")
    print("Done!\n")

# loop to take in img file path from user and predict the label (i.e what the number is)
while True:
        input_str = input("Please enter a filepath:\n> ")
        if input_str.strip().lower() == "exit":
            print("Exiting...")
            break
        # open jpeg image
        try:
            image = Image.open(input_str) 
            transform = transforms.ToTensor() 
            tensor_image = transform(image) # transform image into a tensor
            tensor_image = tensor_image.unsqueeze(0)

            # pass img through model
            model.eval()
            with torch.no_grad():
                output = model(torch.tensor(tensor_image).view(-1, 1))
            
                # get and print prediction
                _, prediction = torch.max(output, 1)
                print("Classifier: ", prediction.item())    
        except:
            print(f"File '{input_str}' Not Found!")
            continue
