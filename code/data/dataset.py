import os
import numpy as np
import torch
from torch.utils.data import Dataset

class LabColorizationDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (string): Directory containing 'L.npy' and 'ab.npy' files.
        """
        self.L_dir = os.path.join(root_dir, 'L.npy')
        self.ab_dir = os.path.join(root_dir, 'ab.npy')
        
        # Load dataset files
        self.L_data = np.load(self.L_dir).astype(np.float32)  # (N, 1, H, W)
        self.ab_data = np.load(self.ab_dir).astype(np.float32)  # (N, 2, H, W)
        
        # Ensure data consistency
        assert self.L_data.shape[0] == self.ab_data.shape[0], "Mismatch between L and ab data!"
        
    def __len__(self):
        return self.L_data.shape[0]

    def __getitem__(self, idx):
        # Extract L and ab tensors
        L = self.L_data[idx] / 255.0  # Normalize to [0, 1]
        ab = (self.ab_data[idx] / 128.0) - 1  # Normalize to [-1, 1]
        
        # Convert to PyTorch tensors
        L = torch.tensor(L, dtype=torch.float32)
        ab = torch.tensor(ab, dtype=torch.float32)
        
        return L, ab
