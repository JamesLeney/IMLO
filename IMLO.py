import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.optim as optim
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32

total_data = datasets.Flowers102(root='data', split='train', download=True, transform=transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
]))

#create data loader
total_loader = DataLoader(total_data, batch_size=batch_size, shuffle=True)

#calculate the mean and std
def mean_std(loader):
    num_pixels = 0
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for images, _ in loader:
        batch_size, channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)) * batch_size * height * width
        std += images.std(axis=(0, 2, 3)) * batch_size * height * width

    mean /= num_pixels
    std /= num_pixels

    return mean, std

mean, std = mean_std(total_loader)
print(f"Mean: {mean}")
print(f"Std: {std}")

#create data loader
total_loader = DataLoader(total_data, batch_size=batch_size, shuffle=True)

#calculate the mean and std
def mean_std(loader):
    num_pixels = 0
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for images, _ in loader:
        batch_size, channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.sum(axis=[0, 2, 3])
        std += images.pow(2).sum(axis=[0, 2, 3])

    mean /= num_pixels
    std = torch.sqrt(std / num_pixels - mean.pow(2))

    return mean, std

mean, std = mean_std(total_loader)
print(f"Mean: {mean}")
print(f"Std: {std}")

#Load the data into the train, val and test splits and transforming them
#to suit our needs
train_data = datasets.Flowers102(root = 'data', split = 'train', download = True, transform = transforms.Compose([
    #Resize the images to be all the same size
    transforms.Resize((64,64)),
    #Turn the data into a tensor
    transforms.ToTensor(),
]))

val_data = datasets.Flowers102(root = 'data', split = 'val', transform = transforms.Compose([
    #Resize the images to be all the same size
    transforms.Resize((64,64)),
    #Turn the data into a tensor
    transforms.ToTensor(),
]))

test_data = datasets.Flowers102(root = 'data', split = 'test', transform = transforms.Compose([
    #Resize the images to be all the same size
    transforms.Resize((64,64)),
    #Turn the data into a tensor
    transforms.ToTensor(),
]))

# Create a data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


# Function to show images
def show_images(loader):
    # Get one batch of data
    for images, labels in loader:
        fig, axes = plt.subplots(1, 6, figsize=(15, 6))
        for i in range(6):
            ax = axes[i]
            ax.imshow(images[i].permute(1, 2, 0))
            ax.axis('off')
        plt.show()
        break  # Only display one batch

# Display some images from the training set
show_images(train_loader)

class NeuralNetwork(nn.Module):
    
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.convolutional1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.convolutional2 = nn.Conv2d(32, 64, 5)
        self.convolutional3 = nn.Conv2d(64, 128, 5)
        self.Fully1 = nn.Linear(128 * 4 * 4, 102)
        
    def forward(self, x):
        x = self.pool(F.relu(self.convolutional1(x)))
        x = self.pool(F.relu(self.convolutional2(x)))
        x = self.pool(F.relu(self.convolutional3(x)))
        x = x.view(x.size(0), -1)
        x = self.Fully1(x)
        return x
    
model = NeuralNetwork().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    val_acc = 0.0
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Validation Accuracy: {val_acc:.2f}%")

    # Evaluate the model on the test dataset
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

test_acc = 0.0
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total

print(f"Test Accuracy: {test_acc:.2f}%")