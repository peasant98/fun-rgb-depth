import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from depth_model import DepthEstimationNet

from depth_dataset import NYUDepthDataset

# Parameters
learning_rate = 0.001
batch_size = 16
num_epochs = 100

# Model, Loss and Optimizer
model = DepthEstimationNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Data loaders
transform = transforms.Compose([transforms.ToTensor()])  # Add necessary transforms
train_dataset = NYUDepthDataset (root_dir='nyu2_train', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        # Unpack the data
        inputs, depth_maps = data
        inputs, depth_maps = inputs.to(device), depth_maps.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        print(outputs.shape, 'i value', i, 'out of', len(train_loader))

        # Compute loss
        loss = criterion(outputs, depth_maps)

        print('Loss:', loss)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print statistics
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
    torch.save(model.state_dict(), f'model_state_dict_{epoch+1}.pth')
print('Finished Training')
