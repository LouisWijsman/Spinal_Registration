import os
import re
import pickle
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from Model import VoxelCNN  # Assuming you are now using the VoxelCNN model
from Data_loader import load_data_from_pickle

working_dir = "/media/volume/Training_data/experiments/experiment_09"







# Define a function to extract the numerical part of the filename
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

# Create the model instance
model = VoxelCNN(in_channels=1, hidden_channels=32, out_channels=3, num_layers=4)   # Adjust based on your VoxelCNN definition

# Load the saved model weights
model.load_state_dict(torch.load(f'{working_dir}/SavedModels/model_weights.pth'))
model.eval()  # Set the model to evaluation mode

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the function to evaluate the model and print outputs
def evaluate_model(model, test_pickle_folder, device, decimal_places=4):
    model.eval()  # Set the model to evaluation mode
    losses = []
    results = {}  # Dictionary to store results
    
    with torch.no_grad():  # No need to track gradients during evaluation
        pickle_files = [f for f in os.listdir(test_pickle_folder) if f.endswith('.pkl')]
        pickle_files.sort(key=extract_number)  # Sort files based on numerical order
        for pickle_file in pickle_files:
            if pickle_file.endswith('.pkl'):
                file_path = os.path.join(test_pickle_folder, pickle_file)
                
                # Load test data from the pickle file
                voxel_data, ground_truth = load_data_from_pickle(file_path)
                
                # Create a DataLoader for the current data
                test_loader = DataLoader([(voxel_data, ground_truth)], batch_size=1)
                
                for voxel_data, ground_truth in test_loader:
                    # Move data to the device
                    voxel_data = voxel_data.to(device)
                    ground_truth = ground_truth.to(device)
                    
                    # Forward pass
                    outputs = model(voxel_data)
                    
                    # Compute loss
                    ground_truth = ground_truth.view(outputs.shape)  # Ensure the shapes match
                    loss = criterion(outputs, ground_truth)  # Compute loss

                    # Convert to NumPy arrays and round for better readability
                    ground_truth_rounded = np.round(ground_truth.cpu().numpy(), decimals=decimal_places).flatten()
                    outputs_rounded = np.round(outputs.cpu().numpy(), decimals=decimal_places).flatten()
                    
                    # Format the output without scientific notation
                    ground_truth_str = ' '.join(f'{x:.{decimal_places}f}' for x in ground_truth_rounded)
                    outputs_str = ' '.join(f'{x:.{decimal_places}f}' for x in outputs_rounded)
                    
                    # Print the outputs and the corresponding ground truth
                    print(f'\nFile: {pickle_file}')
                    print(f'Ground Truth: [{ground_truth_str}]')
                    print(f'Model Output: [{outputs_str}]')
                    
                    losses.append(loss.item())
                    
                    # Store the results in the dictionary
                    results[pickle_file] = {
                        'ground_truth': ground_truth_str,
                        'model_output': outputs_str
                    }
                
    avg_loss = np.mean(losses)
    std_dev_loss = np.std(losses)
    
    # Save the results to a pickle file
    with open('model_outputs.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return avg_loss, std_dev_loss

# Define the loss function
criterion = torch.nn.MSELoss()

# Folder containing test pickle files
test_pickle_folder = f'{working_dir}/testing_data/pickle_files/'  # folder with testing files

# Evaluate the model and print outputs
avg_loss, std_dev_loss = evaluate_model(model, test_pickle_folder, device)
print(f'\nTest Loss avg: {avg_loss}')
print(f'Test Loss Standard Deviation: {std_dev_loss}')
