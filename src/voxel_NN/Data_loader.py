import os
import pickle
import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset





def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Deformed mesh
    #deformed_vertices = np.array(data['deformed_vertices'], dtype=np.float32)
    #deformed_triangles = np.array(data['deformed_triangles'], dtype=np.int32)

    # get voxel grid from pickle file
    voxel_grid_np = data['deformed_voxels_np']
    
    """ # Pickle the numpy array to a file
    pickle_file_path = "VOXEL_grid.pkl"
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(voxel_grid_np, f)
    print(f"Voxel grid numpy array has been pickled to {pickle_file_path}")
 """
    
    
    voxel_tensor = torch.tensor(voxel_grid_np, dtype=torch.float).unsqueeze(0)  # Add channel dimension
    
    # Adjusting ground_truth to be a 3-dimensional vector
    ground_truth = torch.tensor(data['force_vector'][:3], dtype=torch.float)
    
    return voxel_tensor, ground_truth

class PickleDataset(Dataset):
    def __init__(self, pickle_folder):
        super().__init__()
        self.pickle_folder = pickle_folder
        
        self.pickle_files = sorted([f for f in os.listdir(pickle_folder) if f.endswith('.pkl')])
    
    def __len__(self):
        return len(self.pickle_files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.pickle_folder, self.pickle_files[idx])
        voxel_tensor, ground_truth = load_data_from_pickle(file_path)
        return voxel_tensor, ground_truth
