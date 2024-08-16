import os
import random
import signal
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from Model import VoxelCNN  # Ensure this is the correct path to your VoxelCNN class
from Data_loader import PickleDataset  # Ensure this is the correct path to your PickleDataset class



# Set working directory and parameters
working_dir = "/media/volume/Training_data/experiments/experiment_09"

print(f"Training model on {working_dir}\n")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set a random seed
set_seed(42)

# Create the model
model = VoxelCNN(in_channels=1, hidden_channels=32, out_channels=3, num_layers=4)  # Adjust the channels and layers as needed

# Check if CUDA is available and move the model to GPU if it is
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Print whether CUDA is being used
if device.type == 'cuda':
    print("CUDA is available and being used.")
else:
    print("CUDA is not available. Using CPU.")

# Define the loss function and the optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Directory containing pickle files (data for training)
pickle_folder = f'{working_dir}/training_data/pickle_files/'

# Create the dataset and dataloader
dataset = PickleDataset(pickle_folder=pickle_folder)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

# Initialize TensorBoard SummaryWriter
log_dir = f'{working_dir}/logs'
writer = SummaryWriter(log_dir=log_dir)

# Flag to stop training
stop_training = False

# Signal handler for graceful interruption
def signal_handler(sig, frame):
    global stop_training
    print("\nTraining interrupted. Saving the model...")
    stop_training = True

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

# Training loop
epochs = 1000
for epoch in range(epochs):
    if stop_training:
        break
    model.train()  # Set the model to training mode
    epoch_loss = 0  # To accumulate loss over all files
    
    # Iterate through the data loader
    for batch in dataloader:
        if stop_training:
            break
        
        # Unpack the batch
        voxel_data, ground_truth = batch

        # Move data to the same device as the model
        voxel_data = voxel_data.to(device)
        ground_truth = ground_truth.to(device)
        
        optimizer.zero_grad()  # Clear gradients

        # Forward pass
        outputs = model(voxel_data)
        
        # Compute loss
        loss = criterion(outputs, ground_truth)  # Ensure the shapes match
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}')  # Print loss for every epoch
    
    # Log the loss to TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, epoch)

# Save the trained model
torch.save(model.state_dict(), f'{working_dir}/SavedModels/model_weights.pth')
print("Model saved successfully.")

# Close the TensorBoard SummaryWriter
writer.close()
