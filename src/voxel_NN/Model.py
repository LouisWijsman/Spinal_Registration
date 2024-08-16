import torch
import torch.nn.functional as F
import torch.nn as nn

class VoxelCNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4):
        super(VoxelCNN, self).__init__()
        self.convs = nn.ModuleList()
        
        # First layer
        self.convs.append(nn.Conv3d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, voxel_data):
        # Forward pass through 3D convolutional layers
        x = voxel_data
        for conv in self.convs:
            x = conv(x)
            x = F.relu(x)
        
        # Global average pooling
        x = torch.mean(x, dim=(2, 3, 4))  # Assuming the input shape is (batch_size, channels, depth, height, width)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
