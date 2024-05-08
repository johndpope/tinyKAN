import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

# Define the SplineActivation module
class SplineActivation(nn.Module):
    def __init__(self, in_features, out_features, knot_count=5, order=3, extra_knots=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.knot_count = knot_count
        self.order = order
        self.extra_knots = extra_knots

        # Initialize knot positions and spline coefficients
        self.knots = nn.Parameter(torch.linspace(-1, 1, knot_count))
        self.coeffs = nn.Parameter(torch.randn(knot_count + order, out_features, in_features))

    def forward(self, x):
        knots = self.knots.unsqueeze(-1).repeat(1, self.out_features, self.in_features)
        coeffs = self.coeffs.permute(2, 0, 1)  # (in_features, knot_count + order, out_features)

        # Evaluate the spline activation functions
        activations = []
        for i in range(self.in_features):
            spline = torch.tensor([]).to(x.device)
            for j in range(self.out_features):
                knot_values = knots[:, j, i]
                coeff_values = coeffs[i, :, j]
                spline_j = torch.utils.interpolate.spline_basis(knot_values, coeff_values, x[:, i], order=self.order, anti_alias=True, extract_kernel=False)
                spline = torch.cat((spline, spline_j.unsqueeze(1)), dim=1)
            activations.append(spline)

        activations = torch.cat(activations, dim=2)
        return activations.sum(dim=2)

# Define the KAN model
class KANNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, knot_count=5, order=3, extra_knots=2):
        super().__init__()
        self.input_layer = SplineActivation(input_size, hidden_size, knot_count, order, extra_knots)
        self.hidden_layer = SplineActivation(hidden_size, output_size, knot_count, order, extra_knots)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        return x

# Load MNIST dataset
transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the KAN model
input_size = 28 * 28
hidden_size = 256
output_size = 10
model = KANNet(input_size, hidden_size, output_size)

# Training loop
num_epochs = 10
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_loss = 0.0
    for images, labels in train_loader:
        images = images.view(-1, input_size)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss = train_loss / len(train_loader.dataset)

    # Evaluate on test set
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, input_size)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
