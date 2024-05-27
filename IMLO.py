import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time

#Create Perameters
num_epochs = 2000
batch_size = 128
learning_rate = 0.001
best_acc = 0
start_time = 0
time_taken = 0

#Allow GPU use if it is supported
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset with resizing transform
total_data = datasets.Flowers102(root='data', split='train', download=True, transform=transforms.Compose([
    transforms.Resize((256, 256)),
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
        mean += images.mean(axis=(0,2,3)) * batch_size * height * width
        std += images.std(axis=(0,2,3)) * batch_size * height * width

    mean /= num_pixels
    std /= num_pixels

    return mean, std

mean, std = mean_std(total_loader)
print("Mean:", mean)
print("Std:", std)

#Load the data into the train, val and test splits and transforming them
#to suit our needs
train_data = datasets.Flowers102(root='data', split='train', download=True, transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std= std)
]))

val_data = datasets.Flowers102(root='data', split='val', transform=transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std= std)
]))

test_data = datasets.Flowers102(root='data', split='test', transform=transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std= std)
]))

#Create dataloaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


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


# Create the Convolutional Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, num_classes=102):
        super(NeuralNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 14 * 14, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



# Instantiate and move model to device
model = NeuralNetwork().to(device)

# Set up loss function and Optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    start_time = time.time()
    running_loss = 0
    model.train()  # Set the model to training mode
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    model.eval()  # Set the model to evaluation mode
    n_correct = 0
    n_samples = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

    val_acc = 100 * n_correct / n_samples
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

    time_taken += time.time() - start_time
    print(f"{n_correct}/{n_samples}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {val_acc:.2f}%, Time: {time_taken:.2f}s")

# Load the best model for testing
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate on the test set
model.eval()  # Set the model to evaluation mode
n_correct = 0
n_samples = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

test_acc = 100 * n_correct / n_samples
print(f"Test Accuracy: {test_acc:.2f}%")