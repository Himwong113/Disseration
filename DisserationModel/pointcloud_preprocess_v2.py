import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import DisserationConfig as Configure

config =Configure.Para()

def pcNormalization(arr, top):
    arr = np.array(arr)
    max= np.max(arr)
    min= np.min(arr)
    norm_arr = np.floor(top*((arr-min)/(max-min))).astype(int)
    return norm_arr

class PointCloudDataset(Dataset):
    def __init__(self, data_paths, label_paths, transform=None):
        self.data_paths = data_paths  # List of file paths to point cloud data
        self.label_paths = label_paths  # List of file paths to segment label data
        self.transform = transform  # Optional data transformation

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Load and preprocess a point cloud from a file
        point_cloud = np.load(self.data_paths)  # Load the point cloud data (as a NumPy array)
        
        # Load the corresponding segment label data
        segment_label = np.load(self.label_paths)  # Load the label data
        
        # Apply a transformation (e.g., normalization) if provided
        if self.transform:
            point_cloud = self.transform(point_cloud)

        # Convert to PyTorch tensors
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        segment_label = torch.tensor(segment_label, dtype=torch.long)  # Assuming integer labels

        return point_cloud, segment_label

# Create a DataLoader

def create_dataloader(data_paths, label_paths):
    dataset = PointCloudDataset(data_paths, label_paths)
    data_loader = DataLoader(   dataset=dataset,
                                batch_size=config.batchsize,
                                shuffle=True)
    
    return data_loader

