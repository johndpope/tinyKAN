
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model.kan import KAN
# Assuming the KANLayer and KAN classes are defined as in the previous example

# Define the hyperparameters
input_size = 784  # 28x28 pixels
hidden_size = 128
output_size = 10  # 10 classes
num_knots = 10
spline_degree = 3
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Create the KAN model
layer_shapes = [input_size, hidden_size, output_size]
kan_model = KAN(layer_shapes, num_knots, spline_degree)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(kan_model.parameters(), lr=learning_rate)

# Load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    kan_model.train()
    for images, labels in train_loader:
        # Flatten the images
        images = images.view(images.size(0), -1)
        
        # Forward pass
        outputs = kan_model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate on the test set
    kan_model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.view(images.size(0), -1)
            outputs = kan_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%")